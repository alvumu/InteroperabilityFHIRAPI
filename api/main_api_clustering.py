# main_api_clustering.py

import os
import json
import re
import time
import numpy as np
from collections import defaultdict, Counter
import spacy
import nltk
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, SpectralClustering
from sklearn.model_selection import ParameterGrid
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
# Remove unused transformer imports if not used in clustering anymore
# from transformers import AutoTokenizer, AutoModel
import openai
import torch # Ensure torch is imported if used by transformers or sentence-transformers implicitly
import traceback # For detailed error logging
import requests  # For making HTTP requests

# --- Haystack Imports ---
# Make sure haystack is installed: pip install farm-haystack[openai]
# If using older openai (<1.0), compatibility might be needed, otherwise ensure haystack supports openai v1+
try:
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers import InMemoryEmbeddingRetriever
    from haystack.components.preprocessors import DocumentPreprocessor
    from haystack import Document # Import Document schema
    print("Haystack components imported successfully.")
except ImportError as e:
    print(f"ERROR: Failed to import Haystack components. Make sure Haystack is installed ('pip install farm-haystack[openai]'). Details: {e}")
    # Exit or disable RAG functionality if Haystack is critical and missing
    # sys.exit(1)


from fastapi import FastAPI, HTTPException, Query
import uvicorn

# --- Configuration and Initialization ---

# 1. OpenAI API Key Check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable not set. OpenAI-dependent features will fail.")
    # Keep openai.api_key as None or handle it in functions that need it
    openai.api_key = None
else:
    # Check OpenAI version and set key accordingly
    # For openai v0.x
    if hasattr(openai, 'Embedding'):
         openai.api_key = OPENAI_API_KEY
         print("OpenAI API key configured (v0.x style).")
    # For openai v1.x (key is usually read automatically from env var, but check client init)
    # Needs `from openai import OpenAI` and `client = OpenAI()` later
    else:
         print("OpenAI API key found (v1.x expected, ensure client uses it).")


# 2. NLTK Stopwords Download
try:
    nltk.data.find('corpora/stopwords')
    print("NLTK stopwords found.")
except nltk.downloader.DownloadError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 3. SpaCy Model Loading
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Downloading...")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm')
        print("SpaCy model 'en_core_web_sm' downloaded and loaded.")
    except Exception as e:
        print(f"ERROR: Failed to download or load SpaCy model: {e}")
        nlp = None # Set to None to indicate failure

# 4. Embedding Model Loading (Load once at startup - for BOTH Clustering & RAG)
print("Loading embedding models...")
embedding_models_instances = {}
# Models needed for BOTH Clustering and RAG CombinedRetriever
required_model_keys = [
    'neuml/pubmedbert-base-embeddings', # Clustering
    'abhinand/MedEmbed-large-v0.1',    # Clustering & RAG
    'sentence-transformers/all-MiniLM-L12-v2', # Clustering & RAG
    'sentence-transformers/all-mpnet-base-v2' # Clustering & RAG
    # Add other models if needed by either process
]
try:
    for key in required_model_keys:
         print(f"  Loading model: {key}...")
         embedding_models_instances[key] = SentenceTransformer(key)
         print(embedding_models_instances[key])

    # Function references for OpenAI and Combined (used by both potentially)
    embedding_models_instances["openai_embedding"] = lambda text: compute_openai_embedding(text) # Pass the function itself

    # Combined embedding configuration (primarily for clustering, RAG uses its own class)
    # Ensure models referenced here are loaded above
    embedding_models_instances["combined_embedding_config"] = {
        "model_general": embedding_models_instances.get('sentence-transformers/all-MiniLM-L12-v2'),
        "model_general2": embedding_models_instances.get('sentence-transformers/all-mpnet-base-v2'),
        "model_medico": embedding_models_instances.get('abhinand/MedEmbed-large-v0.1'),
        "model_openai": embedding_models_instances["openai_embedding"] # Reference the openai function
    }
    print("All required embedding models loaded successfully.")

except Exception as e:
    print(f"ERROR loading embedding models: {e}")
    print(traceback.format_exc())
    # Decide how to handle model loading failure


# --- Utility Functions (Common, Clustering, RAG) ---

def preprocess_text(text):
    # (Keep existing preprocess_text function)
    if not nlp:
        print("Warning: SpaCy model not available, performing basic lowercasing only.")
        return text.lower() if isinstance(text, str) else str(text).lower()
    if not isinstance(text, str): text = str(text)
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_punct]
    filtered_tokens = [w for w in tokens if w not in stop_words and w.strip() != '']
    return " ".join(filtered_tokens)

def detect_value_type(values):
    # (Keep existing detect_value_type function)
    if not values: return 'Unknown'
    valid_values = [v for v in values if v is not None]
    if not valid_values: return 'Unknown'
    if all(isinstance(v, (int, float)) for v in valid_values): return 'Numeric'
    if all(isinstance(v, str) for v in valid_values):
        if all(('-' in v or '/' in v) and len(v) > 5 for v in valid_values): return 'Date/Time'
        else: return 'Categorical'
    if all(isinstance(v, bool) for v in valid_values): return 'Boolean'
    types = set(type(v) for v in valid_values)
    return 'Mixed' if len(types) > 1 else 'Complex/Other'

def load_json(path):
    # (Keep existing load_json function)
    try:
        with open(path, 'r', encoding='utf-8') as file: data = json.load(file)
        return data
    except FileNotFoundError: print(f"Error: File not found at {path}"); return None
    except json.JSONDecodeError: print(f"Error: Could not decode JSON from {path}"); return None
    except Exception as e: print(f"Error loading JSON from {path}: {e}"); return None

def load_ndjson(path):
    # (Keep existing load_ndjson function)
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    if line.strip(): data.append(json.loads(line))
                except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line in {path}: {line.strip()}")
        return data
    except FileNotFoundError: print(f"Error: File not found at {path}"); return None
    except Exception as e: print(f"Error loading NDJSON from {path}: {e}"); return None

# --- Embedding Functions (Used by Clustering & RAG) ---

def compute_openai_embedding(text):
    """Calculates embedding using OpenAI API (text-embedding-3-large). Handles v0.x and basic v1.x client."""
    # --- START FIX 1: Use text-embedding-3-large and correct dimension ---
    openai_model = "text-embedding-3-large"
    openai_dim = 3072 # Dimension for text-embedding-3-large
    # --- END FIX 1 ---

    if not OPENAI_API_KEY:
        print(f"Warning: OpenAI API key not set. Cannot compute OpenAI embedding. Returning zero vector of dim {openai_dim}.")
        return np.zeros(openai_dim)

    text = str(text) if not isinstance(text, str) else text
    if not text.strip():
        print(f"Warning: Empty text received for OpenAI embedding. Returning zero vector of dim {openai_dim}.")
        return np.zeros(openai_dim)

    try:
        # Attempt v0.x style first
        if hasattr(openai, 'Embedding'):
             # Ensure correct model name is passed
            response = openai.Embedding.create(
                input=text,
                model=openai_model # Use the correct model variable
            )
            embedding = response["data"][0]["embedding"]
        # Basic check for v1.x (requires 'openai' v1+ and client initialized elsewhere or globally)
        # You might need to explicitly initialize 'client = OpenAI()' globally if using v1+
        elif 'client' in globals() and hasattr(globals()['client'], 'embeddings'):
             client = openai.OpenAI(api_key=OPENAI_API_KEY)
             response = client.embeddings.create(input=text, model=openai_model) # Use the correct model variable
             embedding = response.data[0].embedding
        else:
             print(f"ERROR: OpenAI client/method not recognized. Returning zero vector of dim {openai_dim}.")
             return np.zeros(openai_dim)

        embedding_array = np.array(embedding)
        if embedding_array.shape[0] != openai_dim:
            print(f"ERROR: OpenAI embedding dimension mismatch. Expected {openai_dim}, got {embedding_array.shape[0]}. Returning zero vector.")
            return np.zeros(openai_dim)
        return embedding_array

    # Add specific exception handling for OpenAI if possible (depends on version)
    except Exception as e:
        # Catch potential openai.error specific errors if using v0.x
        auth_error_type = getattr(openai.error, 'AuthenticationError', None) if hasattr(openai, 'error') else None
        rate_limit_error_type = getattr(openai.error, 'RateLimitError', None) if hasattr(openai, 'error') else None

        if auth_error_type and isinstance(e, auth_error_type):
             print(f"ERROR: OpenAI Authentication failed. Check your API key. Details: {e}")
        elif rate_limit_error_type and isinstance(e, rate_limit_error_type):
            print(f"ERROR: OpenAI Rate Limit exceeded. Please wait. Details: {e}")
        else:
            print(f"ERROR computing OpenAI embedding ({openai_model}): {e}")
            # print(traceback.format_exc()) # Uncomment for more debug info

        return np.zeros(openai_dim) # Return zero vector on any error


# This version is used by RAG's CombinedRetriever specifically
def compute_combined_embedding_for_rag(text, model_general, model_general2, model_medico, openai_func):
    """
    Computes combined embedding for RAG using MiniLM, MPNet, MedEmbed, and OpenAI (text-embedding-3-large).
    Ensure models passed are loaded SentenceTransformer instances and openai_func works.
    """
    try:
        # --- START FIX 2: Use correct dimensions based on models ---
        # Expected dimensions (use actual model hidden sizes if possible, but hardcoding based on known defaults)
        dim_gen = 384   # sentence-transformers/all-MiniLM-L12-v2
        dim_gen2 = 768  # sentence-transformers/all-mpnet-base-v2
        dim_med = 1024  # abhinand/MedEmbed-large-v0.1
        dim_openai = 3072 # text-embedding-3-large (Changed from 1536)
        total_dim = dim_gen + dim_gen2 + dim_med + dim_openai # Should be 5248
        # --- END FIX 2 ---

        # Encode using SentenceTransformers
        emb_gen = model_general.encode(text, convert_to_numpy=True) if model_general else np.zeros(dim_gen)
        emb_gen2 = model_general2.encode(text, convert_to_numpy=True) if model_general2 else np.zeros(dim_gen2)
        emb_med = model_medico.encode(text, convert_to_numpy=True) if model_medico else np.zeros(dim_med)

        # Compute OpenAI embedding
        emb_openai = openai_func(text) if openai_func else np.zeros(dim_openai)

        # Ensure vectors have correct shapes, fill with zeros if encoding failed or model missing
        emb_gen = emb_gen if emb_gen.shape == (dim_gen,) else np.zeros(dim_gen)
        emb_gen2 = emb_gen2 if emb_gen2.shape == (dim_gen2,) else np.zeros(dim_gen2)
        emb_med = emb_med if emb_med.shape == (dim_med,) else np.zeros(dim_med)
        emb_openai = emb_openai if emb_openai.shape == (dim_openai,) else np.zeros(dim_openai)

        # Apply weights (currently all 1.0) - Consistent with script
        emb_gen_weighted = 1.0 * emb_gen
        emb_gen2_weighted = 1.0 * emb_gen2
        emb_med_weighted = 1.0 * emb_med
        emb_openai_weighted = 1.0 * emb_openai

        # Concatenate (Order matches script implied order)
        combined = np.concatenate([
            emb_gen_weighted,
            emb_gen2_weighted,
            emb_med_weighted,
            emb_openai_weighted
        ], axis=0)

        # Normalization
        norm = np.linalg.norm(combined)
        if norm > 0: combined = combined / norm
        return combined

    except Exception as e:
        print(f"Error computing combined embedding for RAG text: {text[:100]}... Error: {e}")
        # --- START FIX 2b: Use correct total_dim for zero vector ---
        return np.zeros(total_dim)
        # --- END FIX 2b ---

# --- Clustering Specific Functions ---
# (Keep cluster_and_evaluate, select_best_clustering, normalize_embeddings)
# Note: cluster_attributes uses a different combined embedding function if needed

def cluster_and_evaluate(attribute_embeddings, max_k: int):
    # (Keep existing cluster_and_evaluate function, ensuring it uses max_k correctly)
    # ... (code as provided previously) ...
    clustering_results = []
    min_k = 5
    if max_k < min_k: max_k = min_k
    print(f"Cluster evaluation will explore up to {max_k} clusters (starting from {min_k}).")
    cluster_range = range(min_k, max_k + 1)
    clustering_algorithms = { # Using the more extensive list from the user's provided clustering script
        'KMeans': {'model': KMeans,'params': {'n_clusters': cluster_range, 'random_state': [42]}},
        'AgglomerativeClustering': {'model': AgglomerativeClustering,'params': {'n_clusters': cluster_range, 'metric': ['euclidean', 'cosine'],'linkage': ['ward', 'complete', 'average', 'single']}},
        'DBSCAN': {'model': DBSCAN,'params': {'eps': [0.5, 0.3, 0.1, 0.05, 0.01], 'min_samples': [6, 7, 8, 10, 15],'metric': ['euclidean', 'cosine']}},
        'Birch': {'model': Birch,'params': {'n_clusters': cluster_range, 'threshold': [0.05, 0.1, 0.5, 1.0]}},
        'OPTICS': {'model': OPTICS,'params': {'min_samples': [6, 7, 8], 'metric': ['euclidean', 'cosine']}},
        'SpectralClustering': {'model': SpectralClustering,'params': {'n_clusters': cluster_range, 'affinity': ['nearest_neighbors', 'rbf'],'random_state': [42]}}
    }
    print(f"Starting clustering evaluation with {len(attribute_embeddings)} items.")
    if len(attribute_embeddings) < 2: return []
    # ... (rest of the clustering loop, fitting, evaluation, filtering as before) ...
    # Ensure the filtering conditions (min_clusters=3, max_cluster_size=25, min_cluster_size=3) are applied
    for algo_name, algo in clustering_algorithms.items():
        Model = algo['model']
        param_grid = ParameterGrid(algo['params'])
        for params in param_grid:
            if algo_name == 'AgglomerativeClustering':
                linkage = params.get('linkage'); metric = params.get('metric')
                if linkage == 'ward' and metric != 'euclidean': continue
                if linkage == 'single' and metric == 'cosine': continue # Added based on user's original clustering script

            # print(f"Trying {algo_name} with params: {params}") # Keep print for debugging
            try:
                current_params = params.copy()
                is_density_based = Model in [DBSCAN, OPTICS]
                if 'n_clusters' in current_params and is_density_based: del current_params['n_clusters']
                model = Model(**current_params)
                if hasattr(model, 'fit_predict'): cluster_labels = model.fit_predict(attribute_embeddings)
                else: model.fit(attribute_embeddings); cluster_labels = model.labels_

                unique_labels = set(cluster_labels)
                n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)

                if n_clusters_ < 3: continue # Min 3 clusters condition
                cluster_counts = Counter(cluster_labels)
                noise_points = cluster_counts.pop(-1, 0)
                if not cluster_counts: continue # Skip if only noise

                max_cluster_size = 25; min_cluster_size = 3 # Conditions from user script
                if any(size > max_cluster_size for size in cluster_counts.values()): continue
                if any(size < min_cluster_size for size in cluster_counts.values()): continue

                try: # Calculate scores
                    if n_clusters_ >= 2 and len(attribute_embeddings) > n_clusters_:
                         silhouette_avg = silhouette_score(attribute_embeddings, cluster_labels)
                         ch_score = calinski_harabasz_score(attribute_embeddings, cluster_labels)
                         db_score = davies_bouldin_score(attribute_embeddings, cluster_labels)
                    else: silhouette_avg, ch_score, db_score = -1, -1, -1
                except ValueError: silhouette_avg, ch_score, db_score = -1, -1, -1

                clustering_results.append({
                    'algorithm': algo_name, 'params': params, 'n_clusters': n_clusters_,
                    'cluster_sizes': dict(cluster_counts), 'noise_points': noise_points,
                    'silhouette_score': silhouette_avg, 'calinski_harabasz_score': ch_score,
                    'davies_bouldin_score': db_score,
                    'labels': cluster_labels.tolist() if isinstance(cluster_labels, np.ndarray) else cluster_labels
                })
                # print(f"  Success: ...") # Keep print for debugging
            except Exception as e: print(f"ERROR clustering {algo_name} {params}: {e}") # Keep error print

    return clustering_results


def select_best_clustering(clustering_results, weights={'silhouette': 1, 'calinski': 1, 'davies_positive_contribution': 1}):
    # (Keep existing select_best_clustering function)
    # ... (code as provided previously) ...
    if not clustering_results: return None, None
    valid_results = [res for res in clustering_results if res['silhouette_score'] != -1]
    if not valid_results: return None, None
    # ... (normalization and scoring logic) ...
    silhouette_scores = [r['silhouette_score'] for r in valid_results]
    calinski_scores = [r['calinski_harabasz_score'] for r in valid_results]
    davies_scores = [r['davies_bouldin_score'] for r in valid_results]
    def normalize(vals, rev=False):
        min_v, max_v = min(vals), max(vals); rng = max_v - min_v
        if rng == 0: return [0.5] * len(vals)
        norm = [(v - min_v) / rng for v in vals]
        return [1 - n for n in norm] if rev else norm
    sil_norm = normalize(silhouette_scores); cal_norm = normalize(calinski_scores); dav_norm = normalize(davies_scores, rev=True)
    overall = [(weights.get('silhouette', 0)*sil_norm[i] + weights.get('calinski', 0)*cal_norm[i] + weights.get('davies_positive_contribution', 0)*dav_norm[i]) for i in range(len(valid_results))]
    if not overall: return None, None
    best_idx = np.argmax(overall)
    best_result = valid_results[best_idx]
    # ... (print statements for best result) ...
    return best_result['labels'], best_result


def normalize_embeddings(embeddings):
    # (Keep existing normalize_embeddings function)
    if not isinstance(embeddings, np.ndarray): embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return embeddings / norms


def cluster_attributes(attributes, loaded_embedding_models, max_k: int):
    # (Keep existing cluster_attributes function, ensure it passes max_k down)
    # It should iterate through models defined in loaded_embedding_models
    # Use the correct combined embedding function if 'combined_embedding_config' is chosen
    # ... (code as provided previously, adapted for current loaded_embedding_models structure) ...
    if not attributes: return {}, None, {}
    attribute_texts = []
    valid_attributes = []
    for i, attr in enumerate(attributes): # Basic text prep
        try:
            attr_name = attr.get('Attribute name', f'Unknown {i}'); desc = attr.get('Description', ''); vals = attr.get('Values', [])
            v_type = detect_value_type(vals); combined = f"Attribute: {attr_name}. Description: {desc}. Type: {v_type}."
            attribute_texts.append(combined); valid_attributes.append(attr)
        except Exception as e: print(f"Warn: Skip attr {i} text prep: {e}")
    if not attribute_texts: return {}, None, {}
    preprocessed_texts = [preprocess_text(text) for text in attribute_texts]

    all_clustering_results = []
    for model_name, model_instance in loaded_embedding_models.items():
        # Skip the config dict itself, only process actual models/functions
        if model_name == "combined_embedding_config": continue

        print(f"\n--- Clustering with Embedding Model: {model_name} ---")
        attribute_embeddings = []
        try:
            if isinstance(model_instance, SentenceTransformer):
                attribute_embeddings = model_instance.encode(preprocessed_texts, show_progress_bar=False)
            # Handle the combined embedding case using the config dict
            elif model_name == "combined_embedding": # Check for specific key if using combined approach for clustering
                config = loaded_embedding_models.get("combined_embedding_config")
                if config:
                     # Use a dedicated function or inline logic for clustering's combined embedding if different from RAG's
                     print("Generating combined embeddings for CLUSTERING...")
                     # Example: Reuse RAG's function if identical logic is intended
                     attribute_embeddings = np.array([compute_combined_embedding_for_rag(text,
                                                                    config["model_general"], config["model_general2"],
                                                                    config["model_medico"], config["model_openai"])
                                        for text in preprocessed_texts])
                else: print("Warning: Combined embedding config not found."); continue
            elif model_name == "openai_embedding":
                 if not OPENAI_API_KEY: print("Skipping OpenAI: key not set."); continue
                 openai_func = model_instance
                 attribute_embeddings = np.array([openai_func(text) for text in preprocessed_texts])
            else: print(f"Warn: Unknown model type for clustering '{model_name}'. Skip."); continue

            if len(attribute_embeddings) == 0 or attribute_embeddings.shape[0] < 3: continue
            attribute_embeddings = normalize_embeddings(attribute_embeddings)
            print(f"Generated {attribute_embeddings.shape[0]}x{attribute_embeddings.shape[1]} embeddings.")

            # Pass max_k here
            clustering_results = cluster_and_evaluate(attribute_embeddings, max_k) # Pass only embeddings and max_k

            for result in clustering_results: result['embedding_model'] = model_name
            all_clustering_results.extend(clustering_results)
        except Exception as e: print(f"ERROR processing CLUSTERING model {model_name}: {e}\n{traceback.format_exc()}")

    if not all_clustering_results: return {}, None, {}
    selected_labels, best_result_info = select_best_clustering(all_clustering_results)
    if selected_labels is None: return {}, None, {}

    cluster_embedding_model_name = best_result_info['embedding_model']
    clusters = defaultdict(list)
    for idx, label in enumerate(selected_labels): # Group valid attributes
        if idx < len(valid_attributes) and label != -1: clusters[label].append(valid_attributes[idx])
    embedding_models_info = {k: type(v).__name__ for k, v in loaded_embedding_models.items() if k != "combined_embedding_config"}
    return clusters, cluster_embedding_model_name, embedding_models_info


# --- RAG Specific Functions ---

def create_passages(resources, json_schemas):
    """Creates a dictionary mapping resource ID to preprocessed text content."""
    passages = {}
    num_resources = len(resources)
    num_schemas = len(json_schemas)

    if num_resources == 0:
        print("Warning: No resources provided to create passages.")
        return passages
    if num_resources != num_schemas:
        print(f"Warning: Mismatch between resources ({num_resources}) and schemas ({num_schemas}). Attempting to proceed, but results may be inaccurate.")
        # Decide on fallback: use only resources, only schemas, or try to align? For now, let's try to align by index.
        max_common = min(num_resources, num_schemas)
    else:
        max_common = num_resources

    for i in range(max_common):
        res = resources[i]
        schema = json_schemas[i]
        try:
            # Extract ID: Prefer 'id', fallback to 'resourceType', then generate one.
            res_id_keys = ['id', 'resourceType']
            id_val = next((res.get(key) for key in res_id_keys if res.get(key)), f"generated_id_{i}")

            # Extract text: Combine description and purpose if available.
            desc = res.get('description', '')
            purpose = res.get('purpose', '')
            res_text_parts = [desc, purpose, str(schema)] # Combine all available info
            full_text = " ".join(part for part in res_text_parts if part) # Join non-empty parts

            if not full_text.strip():
                print(f"Warning: Skipping resource {id_val} (index {i}) due to empty text content.")
                continue

            passages[str(id_val)] = preprocess_text(full_text) # Preprocess the combined text

        except Exception as e:
            print(f"Error processing resource at index {i}: {e}")
            # Optionally assign a placeholder ID if you need to track errors
            # passages[f"error_id_{i}"] = f"Error processing resource: {e}"
            continue # Skip this resource on error

    if not passages:
         print("Warning: No passages were successfully created from resources and schemas.")

    return passages


def create_documents_from_passages_with_chunks(passages, chunk_size=512, chunk_overlap=50):
    """Generates Haystack Documents from passages using PreProcessor."""
    # Use smaller chunk_size suitable for embedding models
    if not passages:
        print("No passages provided to create documents.")
        return []

    try:
        preprocessor = DocumentPreprocessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False, # Assume no headers/footers in combined text
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
            split_respect_sentence_boundary=True # Try to respect sentences
        )
    except Exception as e:
        print(f"Error initializing Haystack PreProcessor: {e}. Returning empty list.")
        return []


    haystack_documents = []
    for resource_id, text in passages.items():
        if not text or not isinstance(text, str):
            print(f"Warning: Skipping passage for resource_id '{resource_id}' due to invalid content.")
            continue
        try:
            # Haystack process expects a list of dicts
            # Use Haystack's Document schema directly for clarity
            initial_doc = Document(content=text, meta={"resource_id": resource_id})
            # Process returns a list of Document objects
            docs_processed = preprocessor.process([initial_doc])

            for doc in docs_processed:
                # Ensure meta field is preserved or re-added if needed
                if "resource_id" not in doc.meta:
                    doc.meta["resource_id"] = resource_id # Add back if preprocessor removes it
                haystack_documents.append(doc)
        except Exception as e:
            print(f"Error processing passage for resource_id '{resource_id}': {e}")
            print(traceback.format_exc())
            # Continue processing other passages

    print(f"Generated {len(haystack_documents)} Haystack documents (chunks).")
    return haystack_documents

# Define the custom retriever class
class CombinedRetriever(InMemoryEmbeddingRetriever):
    """Custom Haystack Retriever using the combined embedding function."""
    def __init__(self, document_store, model_general, model_general2, model_medico, model_openai, **kwargs):
        # Pass a model name Haystack recognizes for super init
        super().__init__(document_store=document_store, embedding_model="sentence-transformers/all-MiniLM-L12-v2", **kwargs)
        print("Initializing CombinedRetriever...")
        self.model_general = model_general
        self.model_general2 = model_general2
        self.model_medico = model_medico
        self.model_openai = model_openai # This should be the compute_openai_embedding function
        print("CombinedRetriever initialized with models.")

    def embed_documents(self, docs: list[Document]) -> np.ndarray:
        embeddings = []
        print(f"Embedding {len(docs)} documents using CombinedRetriever...")
        # --- START FIX 2c: Use correct total_dim ---
        total_dim = 5248
        # --- END FIX 2c ---
        for i, doc in enumerate(docs):
            if not doc.content or not isinstance(doc.content, str):
                 print(f"Warning: Document {i} has invalid content. Generating zero embedding.")
                 embeddings.append(np.zeros(total_dim))
                 continue
            emb = compute_combined_embedding_for_rag(doc.content, self.model_general, self.model_general2, self.model_medico, self.model_openai)
            embeddings.append(emb)
        print("Document embedding finished.")
        return np.array(embeddings)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        embeddings = []
        print(f"Embedding {len(queries)} queries using CombinedRetriever...")
        # --- START FIX 2d: Use correct total_dim ---
        total_dim = 5248
        # --- END FIX 2d ---
        for query in queries:
            if not query or not isinstance(query, str):
                 print("Warning: Invalid query string received. Generating zero embedding.")
                 embeddings.append(np.zeros(total_dim))
                 continue
            processed_query = preprocess_text(query)
            emb = compute_combined_embedding_for_rag(processed_query, self.model_general, self.model_general2, self.model_medico, self.model_openai)
            embeddings.append(emb)
        print("Query embedding finished.")
        return np.array(embeddings)

    # Keep the retrieve method from the parent class, it uses embed_queries and the document store
    # Or override if custom retrieval logic beyond embedding is needed
    # def retrieve(self, query, top_k=5, filters=None):
    #     print(f"Retrieving top-{top_k} for query: {query[:100]}...")
    #     query_emb = self.embed_queries([query])[0] # Embed single query
    #     results = self.document_store.query_by_embedding(query_emb, top_k=top_k, filters=filters)
    #     print(f"Retrieved {len(results)} documents from store.")
    #     return results


# --- Core Logic Functions ---

def run_clustering_only_pipeline(attribute_path: str, max_k: int, embedding_models_instances: dict):
    # (Keep existing run_clustering_only_pipeline function)
    # ... (code as provided previously) ...
    print(f"Starting clustering pipeline (max_k={max_k})")
    # ... (checks, load attributes, call cluster_attributes, format results) ...
    attributes = load_json(attribute_path)
    if attributes is None: return {"error": f"Failed load: {attribute_path}"}
    if not isinstance(attributes, list) or not attributes: return {"error": "Attribute data invalid/empty."}
    print(f"Loaded {len(attributes)} attributes.")
    try:
        clusters, model_name, models_info = cluster_attributes(attributes, embedding_models_instances, max_k)
    except Exception as e: return {"error": f"Clustering error: {e}"}
    if not clusters: return {"message": "Clustering OK, but no clusters met criteria.", "clusters": {}}
    clusters_dict = {}
    labels_sorted = sorted(clusters.keys())
    for i, label in enumerate(labels_sorted):
        attrs = clusters[label]; key = f"Cluster {i + 1} (Label: {label})"
        clusters_dict[key] = {str(a.get("Attribute name", f"Unk_{label}_{j}")): a.get("Description", "") for j, a in enumerate(attrs)}
    clusters_dict["Winner Embedding Model"] = model_name
    clusters_dict["Embeddings Models Used"] = models_info
    print("Clustering pipeline finished.")
    return clusters_dict


# NEW: Core function for the RAG pipeline
def run_rag_pipeline(cluster_file_path: str, resource_path: str, json_schemas_path: str, top_k: int, threshold: float):
    """
    Loads clustered data, resources, schemas, performs RAG retrieval for each cluster,
    and returns the results matching the expected format.
    """
    print(f"Starting RAG pipeline: clusters='{cluster_file_path}', resources='{resource_path}', schemas='{json_schemas_path}', top_k={top_k}, threshold={threshold}")

    # ... (1. Load Input Data and Validation remains the same) ...
    print("Loading RAG input data...")
    clusters_data = load_json(cluster_file_path)
    resources = load_ndjson(resource_path)
    json_schemas = load_ndjson(json_schemas_path)
    if clusters_data is None: return {"error": f"Failed load cluster data: {cluster_file_path}."}
    if not isinstance(clusters_data, dict): return {"error": "Cluster data not dict."}
    if resources is None: return {"error": f"Failed load resource data: {resource_path}."}
    if not isinstance(resources, list): return {"error": "Resource data not list."}
    if json_schemas is None: return {"error": f"Failed load schema data: {json_schemas_path}."}
    if not isinstance(json_schemas, list): return {"error": "Schema data not list."}
    print(f"Loaded data.")

    # ... (2. Prepare Documents for Haystack remains the same) ...
    print("Creating passages and documents...")
    passages = create_passages(resources, json_schemas)
    if not passages: return {"error": "Failed to create passages."}
    haystack_docs_processed = create_documents_from_passages_with_chunks(passages)
    if not haystack_docs_processed: return {"error": "Failed to create document chunks."}

    # 3. Initialize Haystack Components
    print("Initializing Haystack...")
    try:
        # --- START FIX 3: Use correct embedding_dim ---
        embedding_dim = 5248 # Correct dimension for MiniLM + MPNet + MedEmbed + OpenAI v3 Large
        # --- END FIX 3 ---
        document_store = InMemoryDocumentStore(
            similarity="cosine",
            embedding_dim=embedding_dim
        )

        # Retrieve globally loaded models
        model_gen = embedding_models_instances.get('sentence-transformers/all-MiniLM-L12-v2')
        model_gen2 = embedding_models_instances.get('sentence-transformers/all-mpnet-base-v2')
        model_med = embedding_models_instances.get('abhinand/MedEmbed-large-v0.1')
        openai_func = embedding_models_instances.get('openai_embedding')

        print(openai_func) # Debug print to check if function is loaded correctly

        if not all([model_gen, model_gen2, model_med, openai_func]):
             missing = [name for name, model in [('MiniLM', model_gen), ('MPNet', model_gen2), ('MedEmbed', model_med), ('OpenAI', openai_func)] if not model]
             return {"error": f"Cannot init CombinedRetriever. Missing models: {missing}"}

        combined_retriever = CombinedRetriever(
            document_store=document_store,
            model_general=model_gen, model_general2=model_gen2,
            model_medico=model_med, model_openai=openai_func,
            use_gpu=torch.cuda.is_available()
        )

        # Convert Haystack Document objects to dictionaries for writing
        document_dicts = [doc.to_dict() for doc in haystack_docs_processed]
        document_store.write_documents(document_dicts)
        print(f"Wrote {document_store.get_document_count()} documents to store.")

        print("Updating embeddings...")
        document_store.update_embeddings(retriever=combined_retriever)
        print("Embeddings updated.")

    except Exception as e:
        print(f"ERROR initializing Haystack: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed Haystack setup: {e}"}

    # 4. Perform Retrieval for Each Cluster
    print("Performing retrieval...")
    rag_results = {}
    for cluster_key, cluster_content in clusters_data.items():
        if cluster_key in ["Winner Embedding Model", "Embeddings Models Used", "Embeddings Models"]:
            continue

        print(f"--- Processing {cluster_key} ---")
        if not isinstance(cluster_content, dict):
             print(f"Warn: Skip {cluster_key}, content not dict.")
             rag_results[cluster_key] = {"Attributes": "Invalid Content", "Top Resources": [], "Error": "Invalid cluster content"}
             continue

        # --- START FIX 4: Use cluster_content (dict) directly for "Attributes" ---
        # Construct query from the values (descriptions) in the cluster_content dict
        query_parts = [f"Attribute: {name}. Description: {desc}" for name, desc in cluster_content.items()]
        cluster_query_text = " ".join(query_parts)
        # --- (Keep the rest of the query processing) ---

        if not cluster_query_text.strip():
             print(f"Warn: Skip {cluster_key}, empty query.")
             # Use the original dict for attributes even if query fails
             rag_results[cluster_key] = {"Attributes": cluster_content, "Top Resources": [], "Error": "Empty query"}
             continue

        processed_query = preprocess_text(cluster_query_text)

        try:
            retrieved_docs = combined_retriever.retrieve(query=processed_query, top_k=top_k)
            filtered_docs = [doc for doc in retrieved_docs if hasattr(doc, 'score') and doc.score is not None and doc.score >= threshold]
            print(f"  Retrieved {len(retrieved_docs)}, Filtered {len(filtered_docs)}.")

            # Extract unique resource IDs from filtered documents, maintaining order by score implicitly
            top_resources_dict = {} # Use dict to handle potential duplicate resource_ids from different chunks
            for doc in filtered_docs:
                resource_id = doc.meta.get("resource_id", "Unknown_ID")
                # Store the highest score found for a resource_id to allow sorting later if needed,
                # but we only output the name.
                if resource_id not in top_resources_dict:
                     top_resources_dict[resource_id] = doc.score # Store score temporarily if needed for sorting unique IDs

            # If maintaining the order from Haystack (which is by score) is desired:
            unique_ordered_resources = []
            seen_ids = set()
            for doc in filtered_docs:
                 resource_id = doc.meta.get("resource_id", "Unknown_ID")
                 if resource_id not in seen_ids:
                     unique_ordered_resources.append({"resource": resource_id})
                     seen_ids.add(resource_id)


            rag_results[cluster_key] = {
                "Attributes": cluster_content, # Use the original dict here
                "Top Resources": unique_ordered_resources, # Use the list formatted without score
            }
        except Exception as e:
            print(f"ERROR retrieving for {cluster_key}: {e}\n{traceback.format_exc()}")
            rag_results[cluster_key] = {
                "Attributes": cluster_content, # Still include attributes on error
                "Top Resources": [],
                "Error": f"Retrieval failed: {e}"
            }

    print("RAG pipeline finished.")
    return rag_results

# --- Configuration and Initialization ---

# 1. OpenAI API Key Check (Used for Embeddings and potentially Azure GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Key for text-embedding models
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") # Separate key for Azure endpoint
GPT4V_ENDPOINT_FHIR = os.getenv("GPT4V_ENDPOINT_FHIR") # Azure Endpoint URL

# 2. Together AI API Key Check
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# 3. NLTK Downloads (Ensure punkt is also downloaded)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt') # Check for punkt
    print("NLTK resources (stopwords, punkt) found.")
except nltk.downloader.DownloadError:
    print("Downloading NLTK resources (stopwords, punkt)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True) # Needed for word_tokenize
stop_words = set(stopwords.words('english'))

# 4. SpaCy Model Loading (Keep as is)
# ... (SpaCy loading code) ...
try:
    nlp = spacy.load('en_core_web_sm')
    print("SpaCy model 'en_core_web_sm' loaded.")
except OSError:
    # ... (Download and load logic) ...
    print("ERROR: Failed to load or download SpaCy model 'en_core_web_sm'.")
    nlp = None


# 6. Initialize Together Client (if key exists)
# together_client = None
# if TOGETHER_API_KEY:
#     try:
#         together_client = Together(api_key=TOGETHER_API_KEY)
#         print("Together AI client initialized.")
#     except Exception as e:
#         print(f"WARNING: Failed to initialize Together AI client: {e}")
# else:
#     print("WARNING: TOGETHER_API_KEY not set. Llama provider will not be available.")

# --- Utility Functions (Common, Clustering, RAG, LLM Identification) ---
# Keep: preprocess_text, detect_value_type, load_json, load_ndjson, normalize_embeddings
# Keep/Adapt: compute_openai_embedding (ensure model name matches needs)
# Keep Clustering Functions: cluster_and_evaluate, select_best_clustering, cluster_attributes
# Keep RAG Functions if endpoint exists: create_passages, create_documents_from_passages_with_chunks, compute_combined_embedding_for_rag, CombinedRetriever

# Add/Adapt functions from the LLM identification script:

# Use load_ndjson as it seems appropriate for the resource file format
def load_dataset_llm(filePath):
    """Loads dataset from NDJSON file."""
    return load_ndjson(filePath)

def create_corpus_llm(resources, schemas):
    """Creates corpus strings for LLM context, one string per resource+schema."""
    corpus = []
    num_resources = len(resources)
    num_schemas = len(schemas)
    if num_resources != num_schemas:
        print(f"Warning (create_corpus_llm): Mismatch resources ({num_resources}) vs schemas ({num_schemas}). Using min count.")
        max_common = min(num_resources, num_schemas)
    else:
        max_common = num_resources

    for i in range(max_common):
        res = resources[i]
        schema = schemas[i]
        try:
            # Use lowercasing as in the original script
            res_type = res.get("resourceType", f"unknown_resource_{i}").lower()
            # Combine all available string values from schema, handle non-dict schemas gracefully
            schema_str = str(schema).lower() if schema else ""
            content = f"Resource name : {res_type} Attribute definitions : {schema_str}"
            # Preprocess here if needed, script preprocesses later in generate_response context
            # For consistency let's NOT preprocess here, match script's context injection
            # preprocessed_text = preprocess_text(content) # Original script preprocesses context *later*
            corpus.append(content)
        except Exception as e:
             print(f"Error creating corpus item for index {i}: {e}")
    return corpus # Returns list of strings

def filter_texts_and_schemas_by_resources(target_resource_names, all_resources, all_schemas):
    """Filters resources and schemas based on a list of target resource names."""
    filtered_resources = []
    filtered_schemas = []
    target_set = set(r.lower() for r in target_resource_names) # Case-insensitive matching

    if len(all_resources) != len(all_schemas):
         print(f"Warning (filter): Mismatch all_resources ({len(all_resources)}) vs all_schemas ({len(all_schemas)}).")
         # Handle mismatch or return empty? For now, proceed with min length
         max_common = min(len(all_resources), len(all_schemas))
    else:
         max_common = len(all_resources)


    for i in range(max_common):
        res = all_resources[i]
        res_type = res.get('resourceType', '').lower()
        if res_type in target_set:
            filtered_resources.append(res)
            filtered_schemas.append(all_schemas[i])

    print(f"Filtered down to {len(filtered_resources)} resources matching target list.")
    return filtered_resources, filtered_schemas


def send_request(payload, endpoint):
        """Sends request to Azure OpenAI endpoint."""
        if not AZURE_OPENAI_API_KEY or not endpoint:
             raise ValueError("Azure OpenAI API Key or Endpoint not configured.")
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_OPENAI_API_KEY, # Use the specific Azure key
        }
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json() # Use .json() method
        except requests.exceptions.RequestException as e:
             print(f"Error sending request to Azure endpoint {endpoint}: {e}")
             # Return a mock error structure or raise exception
             return {"error": f"Request failed: {e}", "choices": [{"message": {"content": ""}}]}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Azure: {e}")
            print(f"Response content: {response.text}")
            return {"error": f"JSON decode failed: {e}", "choices": [{"message": {"content": response.text}}]} # Include raw text

# --- LLM Response Generation ---
# Updated to handle errors and structure differences
def generate_response_llm(query: str, context: list, json_schemas: list, iterations: int = 1):
    """
    Generates attribute mapping response using either Azure GPT or Together Llama.

    Args:
        query: The detailed query including attribute info and task instructions.
        context: List of strings representing the resource/schema context.
        json_schemas: List of schema dictionaries used for function/tool definitions.
        iterations: Number of reflection iterations (currently only implemented for GPT).

    Returns:
        The final response content (string, potentially JSON) from the LLM.
    """
    final_response_content = "" # Initialize
    model_name = "GPT" # Default to GPT for now
    if model_name == "GPT":
        if not AZURE_OPENAI_API_KEY or not GPT4V_ENDPOINT_FHIR:
            raise ValueError("Azure OpenAI API Key or Endpoint not configured for GPT.")

        # --- Prepare Azure Functions ---
        functions = []
        for i, schema in enumerate(json_schemas):
             # Ensure schema is a dict and has 'properties'
             if isinstance(schema, dict) and 'properties' in schema:
                 schema_title = schema.get("title", f"Mapping_{i + 1}")
                 schema_title_formatted = schema_title.replace(" ", "_").replace("-", "_").replace(".", "_") # Sanitize name
                 functions.append({
                     "name": schema_title_formatted[:64], # Max 64 chars for name
                     "description": f"Generate mapping for attributes potentially related to {schema_title}", # Keep description concise
                     "parameters": { "type": "object", "properties": schema['properties'] } # Standard JSON schema format
                 })
             else:
                  print(f"Warning: Skipping schema index {i} for function definition due to invalid format.")

        print(f"Prepared {len(functions)} functions for Azure call.")

        # --- Initial Azure Call ---
        payload = {
              "messages": [
                {
                    "role": "system",
                    "content": (
                        f"""
                        ##################### CONTEXT ####################  
                        You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4).  
                        You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.  

                        Review the given attributes thoroughly and choose the mappings that most accurately reflect the semantic meaning and structure of the table attributes in the context of FHIR. Reason about the potential matches internally, but do not include the reasoning steps in the final answer.
                        The documents you have to consider to answer the query is: {context}
                        ##################### TASK ####################  
                        Based on the provided table attribute information and standard FHIR attribute definitions, determine the top 3 most specific FHIR attributes that correspond to each table attribute, listed in order of relevance (most specific first).
                    
                        ##################### OUTPUT FORMAT ####################  
                        Provide your answer as a JSON object, using the following structure:

                        
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                        

                        Each key corresponds to the table attribute name, and its value is an array of exactly three strings. If fewer than three matches are found, use "No additional attribute found" to fill the empty slots.

                        ##################### EXAMPLE ####################  
                        For example, if the table attribute is "patient_birthDate":

                        
                            "table_attribute_name": "patient_birthDate",
                            "fhir_attribute_name": "Patient.birthDate", "Encounter.birthDate", "Patient.anchorAge"
                        

                        ##################### ADDITIONAL INSTRUCTIONS ####################  
                        - Consider the attribute name, description, and sample values when determining the best FHIR attributes.
                        - Return only the final JSON object without additional commentary.
                        - Use the FHIR R4 specification as the reference (https://www.hl7.org/fhir/).
                        - Double-check your mappings before returning the final result.
                        """
                        
                    )
                },
                {
                    "role": "user",
                    "content": f"{query}" # Query contains the attribute details and explicit task/format instructions
                }
            ],
            "temperature": 2, # Slightly higher temp might encourage finding alternatives
            "top_p": 0, # Allow full vocab exploration initially
            "functions": functions, # Add functions only if they were successfully created
            "function_call": "auto"
        }

        print("Sending initial request to Azure GPT...")
        response_data = send_request(payload, GPT4V_ENDPOINT_FHIR)

        if "error" in response_data:
             print(f"Initial Azure GPT call failed: {response_data['error']}")
             return json.dumps({"error": "Initial Azure GPT call failed.", "details": response_data['error']})

        if not response_data or 'choices' not in response_data or not response_data['choices']:
            print("Error: Invalid or empty response from Azure GPT.")
            return json.dumps({"error": "Invalid or empty response from Azure GPT initial call."})

        response_message = response_data['choices'][0]['message']
        current_response_content = ""
        if response_message and 'function_call' in response_message and response_message['function_call']:
            # Handle potential parsing errors for function arguments
            try:
                 current_response_content = json.dumps(json.loads(response_message['function_call']['arguments']))
            except json.JSONDecodeError:
                 print("Warning: Could not parse function call arguments as JSON.")
                 current_response_content = response_message['function_call']['arguments'] # Keep as string
        elif response_message and 'content' in response_message:
            current_response_content = response_message['content']
        else:
             print("Warning: Could not extract content or function call from Azure response.")
             current_response_content = json.dumps({"error": "Could not extract response content."}) # Default error JSON

        print(f"Initial GPT response received (length {len(current_response_content)}).")
        time.sleep(20) # Reduced sleep time

        # --- Reflection Loop (GPT Only) ---
        for i in range(iterations):
            print(f"Starting reflection iteration {i+1}/{iterations}...")
            reflection_payload = {
                "messages": [
                    {"role": "system", "content": f"Iteration {i+1}: Initial mapping of top 3 attributes for each column."},
                    {"role": "assistant", "content": current_response_content},
                    {
                        "role": "system",
                        "content": (
                            f"""
                            #######CONTEXT#######
                            You are a domain expert assistant specializing in mapping clinical tabular data to FHIR resources (FHIR R4). 
                            You have been given information about table attributes (including their names, descriptions, and sample values) and your task is to determine the 3 most specific FHIR attributes that best correspond to each table attribute. You may choose attributes from any FHIR resource as you see fit.
                            You previously provided a mapping between table column names and the top 3 FHIR resource attributes based on similarity.
                            {current_response_content}
                            #######INPUT_DATA#######
                            The documents you have to consider to answer the query is: {context}
                            #######TASK#######
                            Review the mapping above for completeness and accuracy. Ensure the three best-matched attributes are selected.
                            #######OUTPUT_FORMAT#######
                            Provide your answer as a JSON object in the following format:
                                    {{
                                    "table_attribute_name": "attributeName",
                                    "fhir_attribute_name": "FHIRResource.attributeName 1", "FHIRResource.attributeName 2", "FHIRResource.attributeName 3"
                                    }}
                            """
                            

                        )
                    },
                ],
                "temperature": 2,
                "top_p": 0,
                "functions": functions,
                "function_call": "auto"
            }

            response_data = send_request(reflection_payload, GPT4V_ENDPOINT_FHIR)

            if "error" in response_data:
                 print(f"Azure GPT reflection call {i+1} failed: {response_data['error']}")
                 # Keep the last valid response if reflection fails
                 break

            if not response_data or 'choices' not in response_data or not response_data['choices']:
                 print(f"Error: Invalid or empty response from Azure GPT reflection {i+1}.")
                 break # Keep last valid response

            response_message = response_data['choices'][0]['message']
            new_response_content = ""
            if response_message and 'function_call' in response_message and response_message['function_call']:
                try:
                     new_response_content = json.dumps(json.loads(response_message['function_call']['arguments']))
                except json.JSONDecodeError:
                     new_response_content = response_message['function_call']['arguments']
            elif response_message and 'content' in response_message:
                new_response_content = response_message['content']
            else:
                 print(f"Warning: Could not extract content/function call from reflection response {i+1}.")
                 # Keep the previous response if extraction fails
                 new_response_content = current_response_content


            # Basic check if response changed significantly (optional)
            if len(new_response_content) > 0 and new_response_content != current_response_content:
                 print(f"Reflection {i+1} yielded changes.")
                 current_response_content = new_response_content
            else:
                 print(f"Reflection {i+1} did not yield significant changes or failed extraction.")
                 # Optionally break early if no changes or errors occurred

            time.sleep(20) # Reduced sleep time

        final_response_content = current_response_content


    # elif model_name == "Llama":
    #     if not together_client:
    #         raise ValueError("Together AI client not initialized (API Key missing?).")

    #     # --- Prepare Together Tools ---
    #     tools = []
    #     for i, schema in enumerate(json_schemas):
    #          if isinstance(schema, dict) and 'properties' in schema:
    #              schema_title = schema.get("title", f"Mapping_{i + 1}")
    #              schema_title_formatted = schema_title.replace(" ", "_").replace("-", "_").replace(".","_") # Sanitize
    #              tools.append({
    #                  "type": "function",
    #                  "function": {
    #                      "name": schema_title_formatted[:64],
    #                      "description": f"Generate FHIR mapping for attributes potentially related to {schema_title}",
    #                      "parameters": {"type": "object", "properties": schema['properties']}
    #                  }
    #              })
    #          else:
    #               print(f"Warning: Skipping schema index {i} for tool definition due to invalid format.")

    #     print(f"Prepared {len(tools)} tools for Llama call.")

    #     # --- Llama Call (Only one call, script's reflection for Llama wasn't well-defined) ---
    #     try:
    #         # Choose a capable Llama model available via Together API
    #         # Meta-Llama-3.1-405B-Instruct-Turbo might be overkill/expensive, consider 70B or 8B.
    #         # llama_model = "meta-llama/Llama-3-70b-chat-hf"
    #         llama_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" # Or 405B if needed and available

    #         print(f"Sending request to Together Llama ({llama_model})...")
    #         response = together_client.chat.completions.create(
    #             model=llama_model,
    #             messages=[
    #                 {   'role': 'system',
    #                     'content': ( # Keep the system prompt concise but complete
    #                         f"You are an expert FHIR R4 data mapper. Map the user-provided table attributes to the top 3 most specific FHIR attributes (Resource.attribute format). "
    #                         f"Consider attribute names, descriptions, and sample values. Use the provided context documents and FHIR R4 specs. "
    #                         f"Output ONLY a JSON list where each item is {{'table_attribute_name': '...', 'fhir_attribute_name': ['Res.attr1', 'Res.attr2', 'Res.attr3']}}. "
    #                         f"Use 'No additional attribute found' if fewer than 3 matches exist. Context Documents: {context_str}"
    #                     )
    #                 },
    #                 {'role': 'user', 'content': query} # Query contains the attribute details and task
    #             ],
    #             temperature=0.7, # Adjusted temperature
    #             # Add tools only if they exist
    #              **({"tools": tools, "tool_choice": "auto"} if tools else {})
    #         )

    #         response_message = response.choices[0].message
    #         if response_message.tool_calls:
    #             # Preference tool call arguments if available
    #             try:
    #                  # Assuming only one tool call based on script logic
    #                  tool_arguments = response_message.tool_calls[0].function.arguments
    #                  # Attempt to parse/format as JSON string for consistency
    #                  final_response_content = json.dumps(json.loads(tool_arguments))
    #             except (IndexError, AttributeError, json.JSONDecodeError) as e:
    #                  print(f"Warning: Could not extract/parse Llama tool call arguments: {e}")
    #                  final_response_content = response_message.content or json.dumps({"error": "Failed to process tool call."}) # Fallback to content or error
    #         elif response_message.content:
    #             final_response_content = response_message.content
    #         else:
    #              print("Error: Llama response has no content or tool calls.")
    #              final_response_content = json.dumps({"error": "Empty response from Llama."})

    #     except Exception as e:
    #         print(f"Error during Together Llama API call: {e}")
    #         print(traceback.format_exc())
    #         final_response_content = json.dumps({"error": f"Llama API call failed: {e}"})

    #     # No reflection loop implemented for Llama here, matching simplified script logic for it.

    # else:
    #     raise ValueError(f"Unsupported modelName: {model_name}. Choose 'GPT' or 'Llama'.")

    return final_response_content


# --- Core Logic Function for LLM Identification ---
def run_attribute_identification_pipeline(
    #llm_provider: str,
    #iterations: int,
    cluster_file_path: str,
    structured_attributes_path: str,
    resource_path: str,
    schema_path: str):
    """
    Runs the LLM-based attribute identification pipeline for all clusters.
    """
    #print(f"Starting LLM Attribute Identification Pipeline: Provider={llm_provider}, Iterations={iterations}, ClusterFile={cluster_file_path}")

    # 1. Load Required Data
    print("Loading data for LLM pipeline...")
    clusters_data = load_json(cluster_file_path) # Load cluster definitions
    data_structured = load_json(structured_attributes_path) # Load detailed attribute info
    all_resources_data = load_dataset_llm(resource_path) # Load base resources (NDJSON)
    all_schemas_data = load_ndjson(schema_path) # Load schemas (NDJSON)

    # --- Data Validation ---
    if clusters_data is None: return {"error": f"Failed to load cluster data: {cluster_file_path}"}
    if data_structured is None: return {"error": f"Failed to load structured attributes: {structured_attributes_path}"}
    if all_resources_data is None: return {"error": f"Failed to load resources: {resource_path}"}
    if all_schemas_data is None: return {"error": f"Failed to load schemas: {schema_path}"}
    if not isinstance(clusters_data, dict): return {"error": "Cluster data is not dict."}
    if not isinstance(data_structured, list): return {"error": "Structured attributes not list."}
    if not isinstance(all_resources_data, list): return {"error": "Resource data not list."}
    if not isinstance(all_schemas_data, list): return {"error": "Schema data not list."}
    # ------------------------
    print("Data loaded successfully.")

    # 2. Process Each Cluster
    output_data = [] # List to store results for each cluster
    print(clusters_data.items())
    for cluster_name, cluster_info in clusters_data.items():
        # Skip metadata keys from clustering output
        if cluster_name in ["Winner Embedding Model", "Embeddings Models Used", "Embeddings Models"]:
            continue

        print(f"\n--- Processing Cluster: {cluster_name} ---")
        # Ensure cluster_info structure is as expected from the file
        if not isinstance(cluster_info, dict):
             print(f"Warning: Skipping cluster '{cluster_name}', invalid format.")
             output_data.append({"Cluster": cluster_name, "Error": "Invalid cluster format"})
             continue

        # Extract attribute names and resource names from the cluster file
        # Adapt key names based on the actual cluster file structure
        # Assuming format like: {"Cluster 1": {"Attributes": {"name": "desc",...}, "Top Resources": [{"resource": "Name"},...]}}
        attributes_dict = cluster_info.get('Attributes') # Get the dict {name: desc}
        top_resources_list = cluster_info.get('Top Resources', [])

        print(f"Attributes: {attributes_dict}")
        if not attributes_dict or not isinstance(attributes_dict, dict):
            print(f"Warning: No valid attributes found for cluster '{cluster_name}'. Skipping.")
            output_data.append({"Cluster": cluster_name, "Error": "No attributes found"})
            continue
        if not top_resources_list:
             print(f"Warning: No top resources listed for cluster '{cluster_name}'. Using all resources as context.")
             # Decide fallback: error out, use all resources, or skip? Using all for now.
             resource_names = [res.get("resourceType") for res in all_resources_data if res.get("resourceType")]
        else:
            # Extract resource names correctly from list of dicts
            resource_names = [res_info.get('resource') for res_info in top_resources_list if res_info.get('resource')]

        print(f"Target Resources: {resource_names}")
        print(f"Attributes to map: {list(attributes_dict.keys())}")

        # Find detailed info for attributes in this cluster from the structured data file
        attr_details_for_query = []
        for attr_name in attributes_dict.keys():
            # Find the detailed dict in data_structured
            attr_detail = attributes_dict[attr_name]
            if attr_detail:
                attr_details_for_query.append({"Attribute name" : attr_name, "Description": attr_detail, "Values":[]})
            else:
                 print(f"Warning: Details for attribute '{attr_name}' not found in structured data. Using name only.")
                 # Add a basic entry if details are missing but name exists
                 attr_details_for_query.append({"Attribute name": attr_name, "Description": "N/A", "Values": []})

        if not attr_details_for_query:
             print(f"Error: Could not retrieve details for any attributes in cluster '{cluster_name}'. Skipping.")
             output_data.append({"Cluster": cluster_name, "Error": "Attribute details missing"})
             continue

        # --- Construct Query for LLM ---
        query = "##################### INPUT DATA ##################\n"
        for detail in attr_details_for_query:
            query += f"Attribute Name: {detail.get('Attribute name', 'N/A')}\n"
            query += f"Description: {detail.get('Description', 'N/A')}\n"
            #query += f"Sample Values: {detail.get('Values', [])}\n" # Assuming 'Values' key exists
            query += "----------------------------------------\n"
        # Append Task and Output Format instructions (simplified)
        query += (
            "##################### TASK & OUTPUT ##################\n"
            "Map each table attribute above to its top 3 most specific FHIR R4 attributes (Resource.attribute). "
            "Output ONLY a JSON list: [{'table_attribute_name': '...', 'fhir_attribute_name': ['Res.attr1', 'Res.attr2', 'Res.attr3']}, ...]"
        )
        # print(f"Generated Query:\n{query[:500]}...") # Optional: Log query snippet


        # --- Prepare Context ---
        filtered_resources, filtered_schemas = filter_texts_and_schemas_by_resources(
            resource_names, all_resources_data, all_schemas_data
        )
        # Ensure schemas are valid for function/tool generation (list of dicts)
        valid_filtered_schemas = [s for s in filtered_schemas if isinstance(s, dict)]
        if len(valid_filtered_schemas) != len(filtered_schemas):
             print("Warning: Some filtered schemas were not valid dictionaries.")

        context_list = create_corpus_llm(filtered_resources, valid_filtered_schemas)
        if not context_list:
            print(f"Warning: No context generated for cluster '{cluster_name}'. Proceeding without context.")
        # print(f"Generated Context (first item):\n{context_list[0][:500] if context_list else 'N/A'}...") # Optional: Log context snippet


# --- Call LLM ---
        print("Generating response...")
        try:
            llm_response_str = generate_response_llm(
                query,
                context_list,
                valid_filtered_schemas  # Pass valid schemas for function/tool defs
            )
            print(llm_response_str)
        except Exception as e:
            print(f"Error calling generate_response_llm for cluster {cluster_name}: {e}")
            print(traceback.format_exc())
            output_data.append({"Cluster": cluster_name, "Error": f"LLM call failed: {e}"})
            continue  # Skip to next cluster

        # --- Remover bloques Markdown y limpiar la respuesta ---
        # Se asume que la respuesta viene en un bloque marcado con ```json y termina con ```
        pattern = r'```json\s*(.*?)\s*```'
        match = re.search(pattern, llm_response_str, re.DOTALL)
        if match:
            json_string = match.group(1)
        else:
            # En caso de no encontrar los delimitadores, se usa la respuesta completa
            json_string = llm_response_str

        print(f"LLM Response String (raw, first 500 chars): {json_string[:500]}")

        # --- Parse y Almacenamiento del Resultado ---
        parsed_response = None
        try:
            # Convierte la cadena JSON en un objeto Python (lista de diccionarios)
            parsed_response = json.loads(json_string)
            
            # Validación básica: comprobar si es una lista
            if not isinstance(parsed_response, list):
                print(f"Warning: LLM response for {cluster_name} was not a JSON list. Storing raw string.")
                parsed_response = {"raw_response": llm_response_str, "parsing_error": "Expected JSON list"}
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse LLM response for {cluster_name} as JSON. Storing raw string. Error: {e}")
            parsed_response = {"raw_response": llm_response_str, "parsing_error": "Invalid JSON"}
        except Exception as e:
            print(f"Error processing LLM response for {cluster_name}: {e}")
            parsed_response = {"raw_response": llm_response_str, "processing_error": str(e)}

        output_data.append({
            "Cluster": cluster_name,
            # Se almacena la lista parseada o el diccionario de error
            "LLM_Mappings": parsed_response
        })

    print("\nLLM Attribute Identification Pipeline finished.")
    return output_data

# --- FastAPI Application ---
# (Keep app definition and existing endpoints)
app = FastAPI(
    title="Attribute Clustering, RAG & LLM Identification API", # Updated title
    description="Runs pipelines for clustering, resource retrieval (RAG), and LLM-based attribute identification.",
    version="1.3.0", # Incremented version
)

# --- API Endpoints ---

@app.get("/cluster-attributes", tags=["Clustering"])
async def cluster_attributes_endpoint(
    max_k: int = Query(default=40, ge=5, le=100, description="Max number of clusters to explore (inclusive).")
):
    # (Keep existing /cluster-attributes endpoint code as provided previously)
    """
    Triggers the attribute clustering pipeline. Reads attributes, clusters them up to max_k,
    selects the best, and returns the results. Writes results to a predefined file.
    """
    print("AAAAAAAAAAAAAAAAAAAAAAA")
    print(OPENAI_API_KEY, nlp, embedding_models_instances) # Debugging line
    if not all([OPENAI_API_KEY, nlp, embedding_models_instances]): # Basic checks
        raise HTTPException(status_code=503, detail="Server prerequisites not met (OpenAI Key, SpaCy, Models).")
    try:
        
        attribute_path = "data/enriched_attribute_descriptions_SK.json" # Input for clustering
        # Define the output path where clustering results will be saved (and RAG will read from)
        cluster_output_path = "data/only_clusters_resQ_sk.json"

        print(f"Cluster API: Attr Path='{attribute_path}', Output Path='{cluster_output_path}', Max K={max_k}")
        if not os.path.exists(attribute_path):
            raise HTTPException(status_code=500, detail=f"Attribute file not found: {attribute_path}")

        # Run clustering
        results = run_clustering_only_pipeline(attribute_path, max_k,embedding_models_instances)

        if isinstance(results, dict) and "error" in results: raise HTTPException(status_code=500, detail=results["error"])
        # Save results to file for RAG endpoint to use
        try:
            os.makedirs(os.path.dirname(cluster_output_path), exist_ok=True)
            with open(cluster_output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Clustering results saved to: {cluster_output_path}")
        except Exception as e:
            print(f"ERROR saving clustering results to {cluster_output_path}: {e}")
            # Decide if this should be a fatal error for the endpoint
            raise HTTPException(status_code=500, detail=f"Clustering succeeded, but failed to save results: {e}")

        print("Clustering API request processed successfully.")
        return results # Also return results directly from API

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        print(f"Unexpected error in /cluster-attributes: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# NEW: RAG Endpoint
@app.get("/retrieve-resources", tags=["Retrieval (RAG)"])
async def retrieve_resources_endpoint(
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top documents to retrieve per cluster."),
    threshold: float = Query(default=0.70, ge=0.0, le=1.0, description="Minimum similarity score threshold for retrieved documents.")
):
    # (Keep the endpoint logic, ensuring it calls the corrected run_rag_pipeline)
    """
    Triggers the RAG pipeline using pre-computed clusters. Reads cluster data, resources, schemas,
    indexes them using Haystack with a combined embedding model (using text-embedding-3-large),
    and retrieves the top_k relevant resources for each cluster, filtered by score threshold.
    Output matches the specified format (attribute dictionary, resources without scores).
    """
    if not all([OPENAI_API_KEY, nlp, embedding_models_instances]):
        raise HTTPException(status_code=503, detail="Server prerequisites not met.")
    try:
        cluster_file_path = "data/clusters_resQ_Sk_API_corrected.json"
        resource_path = "data/datasetRecursosBase.ndjson"
        json_schemas_path = "data/enriched_dataset_schemasBase.ndjson"
        print(f"RAG API triggered: K={top_k}, Threshold={threshold}")

        # --- File existence checks ---
        if not os.path.exists(cluster_file_path): raise HTTPException(status_code=404, detail=f"Cluster file missing: {cluster_file_path}. Run /cluster-attributes first.")
        if not os.path.exists(resource_path): raise HTTPException(status_code=500, detail=f"Resource file missing: {resource_path}")
        if not os.path.exists(json_schemas_path): raise HTTPException(status_code=500, detail=f"Schema file missing: {json_schemas_path}")

        results = run_rag_pipeline(cluster_file_path, resource_path, json_schemas_path, top_k, threshold)

        if isinstance(results, dict) and "error" in results: raise HTTPException(status_code=500, detail=results["error"])

        # Optional: Save RAG results to file
        rag_output_path = "data/clusters_resQ_Sk_API_corrected.json" # New name
        try:
            os.makedirs(os.path.dirname(rag_output_path), exist_ok=True)
            with open(rag_output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"RAG results saved: {rag_output_path}")
        except Exception as e: print(f"Warn: Failed save RAG results: {e}") # Non-fatal

        print("RAG API request processed successfully.")
        return results

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        print(f"Unexpected error in /retrieve-resources: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error during RAG: {str(e)}")

@app.get("/identify-attributes-llm", tags=["LLM Identification"])
async def identify_attributes_llm_endpoint(
    #llm_provider: str = Query(default="Llama", description="LLM provider to use ('GPT' for Azure, 'Llama' for Together)."),
    #iterations: int = Query(default=1, ge=0, le=5, description="Number of reflection iterations (currently only for GPT)."),
    #cluster_file_name: str = Query(default="only_clusters_resQ_sk.json", description="Filename of the cluster results JSON within 'data/'.")
):
    """
    Triggers the LLM-based attribute identification pipeline.

    Reads cluster data, structured attributes, resources, and schemas from predefined paths.
    For each cluster, it constructs a prompt with attribute details and context from relevant resources/schemas,
    then calls the specified LLM provider (Azure GPT or Together Llama) to generate FHIR attribute mappings.
    """
    #print(f"LLM Identification API triggered: Provider={llm_provider}, Iterations={iterations}, ClusterFile={cluster_file_name}")

    # --- Prerequisite Checks ---
    if (not AZURE_OPENAI_API_KEY or not GPT4V_ENDPOINT_FHIR):
         raise HTTPException(status_code=503, detail="Azure OpenAI API Key or Endpoint not configured for GPT provider.")
    #if llm_provider == "Llama" and not together_client: # Check if client initialized
    #    raise HTTPException(status_code=503, detail="Together AI API Key not configured or client failed to initialize for Llama provider.")
    if not nlp:
         raise HTTPException(status_code=503, detail="Server configuration error: SpaCy model failed to load.")
    # Add checks for other necessary components if needed

    # --- Define File Paths ---
    base_data_path = "data"
    base_output_path = "/output"
    llm_output_base = "" # Separate output dir for LLM results
    cluster_file_name = "clusters_resQ_Sk_API_corrected.json"
    cluster_file_path = os.path.join(base_output_path, cluster_file_name)
    # Path to the structured attributes file used by the script
    structured_attributes_path = "data/filtered_data_attributes.json" # Based on script variable 'path' + 'filename'
    resource_path = os.path.join(base_data_path, "datasetRecursosBase.ndjson") # Based on script
    schema_path = os.path.join(base_data_path, "enriched_dataset_schemasBase.ndjson") # Based on script


    # --- File Existence Checks ---
    required_files = {
        "Cluster File": cluster_file_path,
        "Structured Attributes": structured_attributes_path,
        "Resources": resource_path,
        "Schemas": schema_path
    }
    for name, path in required_files.items():
         if not os.path.exists(path):
              raise HTTPException(status_code=404, detail=f"{name} file not found at expected path: {path}")

    # --- Run Pipeline ---
    try:
        results = run_attribute_identification_pipeline(
            #llm_provider=llm_provider,
            #iterations=iterations,
            cluster_file_path=cluster_file_path,
            structured_attributes_path=structured_attributes_path,
            resource_path=resource_path,
            schema_path=schema_path
        )

        if isinstance(results, dict) and "error" in results:
            # Pipeline itself reported an error (e.g., loading failed)
            raise HTTPException(status_code=500, detail=f"Pipeline error: {results['error']}")

        # --- Optional: Save results to file ---
        output_filename = f"llm_results_iter_{cluster_file_name}"
        output_path = os.path.join(llm_output_base, output_filename)
        try:
            os.makedirs(llm_output_base, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                # Results should be a list of dicts here
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"LLM identification results saved to: {output_path}")
        except Exception as e:
             print(f"Warning: Failed to save LLM results to {output_path}: {e}")
             # Do not fail the request if saving fails

        print("LLM Identification API request processed successfully.")
        return results # Return the list of results

    except HTTPException as http_exc:
        raise http_exc # Re-raise specific HTTP exceptions
    except ValueError as ve: # Catch config errors etc.
         print(f"Configuration or Value Error: {ve}")
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"An unexpected error occurred in the LLM endpoint: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error during LLM identification: {str(e)}")


# --- Run the API (for local development) ---
if __name__ == "__main__":
    print("Starting Uvicorn server for local development...")
    uvicorn.run("main_api_clustering:app", host="0.0.0.0", port=8001, reload=True)