# main_api_clustering.py

import math
import os
import json
import logging
import logging.config
import sys
import re
import time
from typing import Dict, List
import numpy as np
from collections import defaultdict, Counter
from functools import wraps
import spacy
import nltk
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, SpectralClustering
from sklearn.model_selection import ParameterGrid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from together import Together
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, CrossEncoder
# Remove unused transformer imports if not used in clustering anymore
# from transformers import AutoTokenizer, AutoModel
from openai import OpenAI, embeddings
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

LOG_FILE = os.getenv("LOG_FILE", "log.out")

# --- Rate Limiter Configuration ---
OPENAI_RATE_LIMIT_CONFIG = {
    "max_retries": 5,
    "base_delay": 1,  # segundos
    "max_delay": 60,  # segundos máximo
    #"batch_size": 100,  # Máximo 100 inputs por request
}

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": LOG_FILE,
            "mode": "w",
            "encoding": "utf-8",
            "formatter": "default",
        }
    },
    "root": {"handlers": ["file"], "level": "INFO"},
    "loggers": {
        "uvicorn":         {"handlers": ["file"], "level": "INFO", "propagate": False},
        "uvicorn.error":   {"handlers": ["file"], "level": "INFO", "propagate": False},
        "uvicorn.access":  {"handlers": ["file"], "level": "INFO", "propagate": False},
    },
}

logging.config.dictConfig(LOGGING_CONFIG)

# Redirigir print() a logging (INFO) y errores a logging (ERROR)
class _StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
    def write(self, message):
        message = message.strip()
        if message:
            self.logger.log(self.level, message)
    def flush(self):  # para compatibilidad con streams
        pass

sys.stdout = _StreamToLogger(logging.getLogger("STDOUT"), logging.INFO)
sys.stderr = _StreamToLogger(logging.getLogger("STDERR"), logging.ERROR)

# --- Configuration and Initialization ---

# 1. OpenAI API Key Check (Used for Embeddings and potentially Azure GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Key for text-embedding models
print(f"OpenAI API Key found: {'Yes' if OPENAI_API_KEY else 'No'}")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API") # Separate key for Azure endpoint
print(f"Azure OpenAI API Key found: {'Yes' if AZURE_OPENAI_API_KEY else 'No'}")
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

import re

_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")

def preprocess_text_dense(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _CTRL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text
from typing import Any, Dict, List

def extract_fhir_elements_from_schema(schema: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    def deep_get(d: Dict[str, Any], keys: List[str], default=None):
        cur = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    base_props = deep_get(
        schema,
        ["properties", "mappings", "items", "properties", "fhir_attributes", "properties"],
        default={}
    )
    if not isinstance(base_props, dict):
        return out

    # Heurística de corrección: si estamos dentro de un objeto "X.Y" y vemos claves "X.something"
    # que NO empiezan por "X.Y.", asumimos que faltó el prefijo "Y".
    def maybe_fix_child_path(parent_path: str, child_key: str) -> str:
        if not parent_path or "." not in parent_path:
            return child_key
        res_prefix = parent_path.split(".", 1)[0] + "."  # "Encounter."
        if child_key.startswith(res_prefix) and not child_key.startswith(parent_path + "."):
            # e.g. parent_path="Encounter.reason", child_key="Encounter.use"
            suffix = child_key[len(res_prefix):]  # "use"
            return parent_path + "." + suffix      # "Encounter.reason.use"
        return child_key

    def walk(props: Dict[str, Any], parent_path: str = ""):
        if not isinstance(props, dict):
            return
        for k, v in props.items():
            if not isinstance(v, dict):
                continue

            # Si k parece path FHIR
            if "." in k:
                fixed_k = maybe_fix_child_path(parent_path, k)
                out.append({
                    "path": fixed_k,
                    "type": str(v.get("type", "")).strip(),
                    "description": str(v.get("description", "")).strip(),
                })
                # Si este nodo tiene properties, recorre con parent_path = fixed_k
                if isinstance(v.get("properties"), dict):
                    walk(v["properties"], parent_path=fixed_k)
            else:
                # nodo no-path: puede contener properties
                if isinstance(v.get("properties"), dict):
                    walk(v["properties"], parent_path=parent_path)

    walk(base_props, parent_path="")

    # Dedup por path manteniendo orden
    seen = set()
    uniq = []
    for e in out:
        p = e["path"]
        if p and p not in seen:
            seen.add(p)
            uniq.append(e)
    return uniq
import re
from typing import Dict, Any, Optional, Tuple

FIELD_RE = re.compile(r'^\s*\"(?P<field>[^\"]+)\"\s*:\s*(?P<val>.+?)(?:,)?\s*(?://\s*(?P<comment>.*))?$')
ALLOWED_RE = re.compile(r'\b([a-zA-Z][\w-]*\s*(?:\|\s*[a-zA-Z][\w-]*){1,})\b')  # "a | b | c"

def normalize_fhir_datatype(val_fragment: str) -> str:
    v = val_fragment.strip()
    # tokens comunes
    if "<dateTime>" in v: return "dateTime"
    if "<code>" in v: return "code"
    if "<string>" in v: return "string"
    if "<boolean>" in v: return "boolean"
    if "<positiveInt>" in v: return "positiveInt"
    if "{ Duration" in v: return "Duration"
    if "{ Period" in v: return "Period"
    if "{ Age" in v: return "Age"
    if "{ Range" in v: return "Range"
    if "{ Timing" in v: return "Timing"
    if "{ Identifier" in v: return "Identifier"
    if "CodeableConcept" in v: return "CodeableConcept"
    if "Reference(" in v or "{ Reference" in v: return "Reference"
    if "Annotation" in v: return "Annotation"
    return ""
def extract_comments_from_structure_dict(struct_obj: Any, resource_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Busca un dict bajo clave '_comments' y lo transforma a:
      Resource.field -> {comment, allowed, datatype:''}
    """
    def find_comments(o) -> Optional[dict]:
        if isinstance(o, dict):
            if "_comments" in o and isinstance(o["_comments"], dict):
                return o["_comments"]
            for v in o.values():
                found = find_comments(v)
                if found:
                    return found
        elif isinstance(o, list):
            for it in o:
                found = find_comments(it)
                if found:
                    return found
        return None

    comments = find_comments(struct_obj) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for field, comment in comments.items():
        if not isinstance(field, str):
            continue
        full_path = f"{resource_type}.{field}"
        out[full_path] = {
            "comment": str(comment).strip(),
            "allowed": [],
            "datatype": "",
        }
    return out
FIELD_RE = re.compile(r'^\s*\"(?P<field>[^\"]+)\"\s*:\s*(?P<val>.+?)(?:,)?\s*(?://\s*(?P<comment>.*))?$')
ALLOWED_RE = re.compile(r'\b([a-zA-Z][\w-]*\s*(?:\|\s*[a-zA-Z][\w-]*){1,})\b')

def parse_structure(structure: Any, resource_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Soporta:
      - structure como dict (Patient-style) -> usa _comments
      - structure como str (Encounter-style) -> parsea líneas
    """
    result: Dict[str, Dict[str, Any]] = {}
    if not structure:
        return result

    # Caso 1: dict -> extraer _comments (mejor señal)
    if isinstance(structure, dict) or isinstance(structure, list):
        return extract_comments_from_structure_dict(structure, resource_type)

    # Caso 2: string -> parsear
    structure_str = structure if isinstance(structure, str) else str(structure)
    lines = structure_str.splitlines()

    for line in lines:
        m = FIELD_RE.match(line)
        if not m:
            continue

        field = m.group("field").strip()
        val = (m.group("val") or "").strip()
        comment = (m.group("comment") or "").strip()

        dtype = normalize_fhir_datatype(val)

        allowed = []
        if comment:
            mm = ALLOWED_RE.search(comment)
            if mm:
                allowed = [x.strip() for x in mm.group(1).split("|") if x.strip()]

        full_path = f"{resource_type}.{field}"
        result[full_path] = {
            "comment": comment,
            "allowed": allowed,
            "datatype": dtype,
        }

    return result
def truncate(text: str, max_chars: int = 600) -> str:
    t = (text or "").strip()
    return t if len(t) <= max_chars else t[:max_chars] + "..."

def build_fhir_passages_from_schema_and_structure(
    schema_json: dict,
    resource_json: Optional[dict] = None
) -> Dict[str, str]:
    passages: Dict[str, str] = {}

    res_type = schema_json.get("title") or (resource_json or {}).get("resourceType") or "Unknown"
    schema_desc = schema_json.get("description", "") or ""
    res_desc = (resource_json or {}).get("description", "") or ""

    # 1) Elementos desde schema
    elements = extract_fhir_elements_from_schema(schema_json)

    # 2) Señales desde structure (si existe)
    #structure_str = ""
    if resource_json:
        structure_obj = (resource_json or {}).get("structure")
        structure_info = parse_structure(structure_obj, res_type)

    # Resource-level: key elements top-level
    top_level = []
    for e in elements:
        p = e["path"]
        if p.startswith(res_type + ".") and p.count(".") == 1:
            top_level.append(p.split(".", 1)[1])
    top_level = sorted(set(top_level))

    resource_text = "\n".join([
        f"fhir_resource: {res_type}",
        f"schema_description: {truncate(schema_desc, 500)}" if schema_desc else "schema_description:",
        f"resource_description: {truncate(res_desc, 500)}" if res_desc else "resource_description:",
        "key_elements: " + ", ".join(top_level) if top_level else "key_elements:"
    ])
    passages[f"RESOURCE::{res_type}"] = preprocess_text_dense(resource_text)

    # Element-level
    for e in elements:
        path = e["path"]
        sch_type = e.get("type", "")
        sch_desc = e.get("description", "")

        # Enriquecimiento con structure
        st = structure_info.get(path, {})
        st_comment = st.get("comment", "")
        st_allowed = st.get("allowed", [])
        st_dtype = st.get("datatype", "")

        # Elegir datatype: structure > schema
        dtype = st_dtype or sch_type

        # Elegir description: schema + comment (si aporta)
        desc = sch_desc.strip()
        if st_comment and st_comment not in desc:
            desc = (desc + " " + st_comment).strip() if desc else st_comment.strip()

        elem_lines = [
            f"fhir_resource: {res_type}",
            f"fhir_path: {path}",
            f"datatype: {dtype}" if dtype else "datatype:",
            f"description: {truncate(desc, 700)}" if desc else "description:",
        ]
        if st_allowed:
            elem_lines.append("allowed_values: " + ", ".join(st_allowed[:30]))

        passages[f"ELEMENT::{path}"] = preprocess_text_dense("\n".join(elem_lines))

    return passages

def build_fhir_resource_passage_only(
    schema_json: dict,
    resource_json: Optional[dict] = None,
    max_fields: int = 60
) -> Dict[str, str]:
    """
    Devuelve SOLO un pasaje por recurso:
      passages["RESOURCE::<ResourceType>"] = texto
    """
    passages: Dict[str, str] = {}

    res_type = schema_json.get("title") or (resource_json or {}).get("resourceType") or "Unknown"
    schema_desc = schema_json.get("description", "") or ""
    res_desc = (resource_json or {}).get("description", "") or ""

    # Elementos (solo para enriquecer el pasaje del recurso, no para indexar elementos)
    elements = extract_fhir_elements_from_schema(schema_json)

    # Señales desde structure (enums / comments), si existe
    structure_info: Dict[str, Dict[str, Any]] = {}
    if resource_json:
        structure_obj = (resource_json or {}).get("structure")
        structure_info = parse_structure(structure_obj, res_type) if structure_obj else {}

    # Prioriza top-level fields
    top_level = [e for e in elements if e["path"].startswith(res_type + ".") and e["path"].count(".") == 1]
    nested = [e for e in elements if e not in top_level]
    chosen = (top_level + nested)[:max_fields]

    lines = [
        f"fhir_resource: {res_type}",
        f"schema_description: {truncate(schema_desc, 600)}" if schema_desc else "schema_description:",
        f"resource_description: {truncate(res_desc, 600)}" if res_desc else "resource_description:",
        "fields:"
    ]

    for e in chosen:
        path = e["path"]
        dtype = e.get("type", "")
        desc = e.get("description", "")

        st = structure_info.get(path, {})
        st_comment = st.get("comment", "")
        st_allowed = st.get("allowed", [])
        st_dtype = st.get("datatype", "")

        # Preferimos datatype del structure si existe
        dtype_final = st_dtype or dtype

        # Enriquecemos descripción con comment/enums si aporta
        desc_final = (desc or "").strip()
        if st_comment and st_comment not in desc_final:
            desc_final = (desc_final + " " + st_comment).strip() if desc_final else st_comment.strip()

        field_line = f"- {path}"
        if dtype_final:
            field_line += f" ({dtype_final})"
        if desc_final:
            field_line += f": {truncate(desc_final, 180)}"
        if st_allowed:
            field_line += f" | allowed: {', '.join(st_allowed[:15])}"

        lines.append(field_line)

    resource_text = preprocess_text_dense("\n".join(lines))
    passages[res_type] = resource_text
    return passages


# 4. Embedding Model Loading (Load once at startup - for BOTH Clustering & RAG)
print("Loading embedding models...")
embedding_models_instances = {}
# Models needed for BOTH Clustering and RAG CombinedRetriever
BASE_EMBEDDING_SPECS = {
    "general":  {"model_key": "sentence-transformers/all-MiniLM-L12-v2"   },
    "general2": {"model_key": "sentence-transformers/all-mpnet-base-v2"},
    "medico":   {"model_key": "abhinand/MedEmbed-large-v0.1"             },
    "medico2":  {"model_key": "NeuML/pubmedbert-base-embeddings"},
    # #"openai":   {"model_key": "openai_embedding"},
    # #"biobert":  {"model_key": "dmis-lab/biobert-base-cased-v1.1"}, 
    "biobert": {"model_key": "pritamdeka/S-BioBert-snli-multinli-stsb"},
    "gemma":    {"model_key": "google/embeddinggemma-300m"  },
    "modern_medico": {"model_key": "lokeshch19/ModernPubMedBERT"},

}
try:
    for key, obj in BASE_EMBEDDING_SPECS.items():
        print(f"  Loading model: {key}...")
        embedding_models_instances[key] = SentenceTransformer(obj["model_key"],similarity_fn_name='euclidean')
        embedding_models_instances[key].max_seq_length = embedding_models_instances[key].get_max_seq_length()
        print(embedding_models_instances[key])

    # Function references for OpenAI and Combined (used by both potentially)
    #embedding_models_instances["openai_embedding"] = lambda text: compute_openai_embedding(text) # Pass the function itself

    # Combined embedding configuration (primarily for clustering, RAG uses its own class)
    # Ensure models referenced here are loaded above
    print("All required embedding models loaded successfully.")

except Exception as e:
    print(f"ERROR loading embedding models: {e}")
    print(traceback.format_exc())
    # Decide how to handle model loading failure


# Configuraciones de embedding a probar en el CLUSTERING
EMBEDDING_CONFIGS = {
    #"general_only":             {"components": ["general"]},
    "general2_only":            {"components": ["general2"]},
    "medico_only":              {"components": ["medico"]},
    "medico2_only":             {"components": ["medico2"]},
    # #"openai_only":              {"components": ["openai"]},
    "biobert_only":             {"components": ["biobert"]},
    "gemma_only":              {"components": ["gemma"]},
    "modern_medico_only":     {"components": ["modern_medico"]},

    # "general+medico":           {"components": ["general", "medico"]},
    # "general+medico2":          {"components": ["general", "medico2"]},
    # "general+biobert":         {"components": ["general", "biobert"]},
    # "general+modern_medico":   {"components":["general","modern_medico"]},

    # "general2+medico":          {"components": ["general2", "medico"]},
    # "general2+medico2":         {"components": ["general2", "medico2"]},
    # "general2+biobert":        {"components": ["general2", "biobert"]},
    # "general2+modern_medico":   {"components": ["general2", "modern_medico"]},


    # "gemma+medico":            {"components": ["gemma", "medico"]},
    # "gemma+medico2":           {"components": ["gemma", "medico2"]},
    # "gemma+biobert":           {"components": ["gemma", "biobert"]},
    # "general+medico+gemma":     {"components": ["general", "medico", "gemma"]},

    # "medico+medico2":           {"components": ["medico","medico2"]},
    # "medico+modern_medico":     {"components": ["medico","modern_medico"]},
    # "medico+biobert":           {"components": ["medico", "biobert"]},

    # "medico2+modern_medico":    {"components": ["medico2","modern_medico"]},
    # "medico2+biobert":          {"components": ["medico2", "biobert"]}
    # "medico+gemma":            {"components": ["medico", "gemma"]},
    # "medico+biobert+gemma":    {"components": ["medico", "biobert", "gemma"]},

    # "general+biobert":         {"components": ["general", "biobert"]},
    # "general+biobert+gemma":   {"components": ["general", "biobert", "gemma"]},
    # "general2+biobert":        {"components": ["general2", "biobert"]},
    # "general2+biobert+gemma":  {"components": ["general2", "biobert", "gemma"]},
    # "general+medico+biobert":   {"components": ["general", "medico", "biobert"]},

    # "all_models_with_biobert":  {"components": ["general", 
    #                                             "general2",
    #                                              "medico", "biobert", "gemma"]},
}

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
                print(f"Reading line: {line.strip()}")
                try:
                    if line.strip(): data.append(json.loads(line))
                except json.JSONDecodeError: print(f"Warning: Skipping invalid JSON line in {path}: {line.strip()}")
        return data
    except FileNotFoundError: print(f"Error: File not found at {path}"); return None
    except Exception as e: print(f"Error loading NDJSON from {path}: {e}"); return None

# --- Embedding Functions (Used by Clustering & RAG) ---

def retry_with_exponential_backoff(max_retries=5, base_delay=1, max_delay=60):
    """Decorator con backoff exponencial para manejar rate limits de OpenAI."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = base_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Detectar rate limit errors
                    if any(err in error_str for err in ['rate_limit', '429', 'too many requests', 'quota']):
                        retries += 1
                        if retries >= max_retries:
                            print(f"ERROR: Max retries ({max_retries}) exceeded for rate limit. Returning empty list.")
                            raise
                        delay = min(delay * 2, max_delay)  # Exponential backoff
                        print(f"Rate limit hit. Retry {retries}/{max_retries} after {delay}s delay...")
                        time.sleep(delay)
                    else:
                        # No es rate limit, propagar error
                        raise
            return None
        return wrapper
    return decorator

@retry_with_exponential_backoff(**OPENAI_RATE_LIMIT_CONFIG)
def compute_openai_embedding_batch(texts):
    """Computes embeddings for batch of texts using OpenAI API con rate limiting.
    
    Args:
        texts: List[str] - Máximo 100 textos por request
    
    Returns:
        List[np.ndarray] - Una embedding por texto
    """
    openai_model = "text-embedding-3-large"
    openai_dim = 3072
    
    if not OPENAI_API_KEY:
        print(f"Warning: OpenAI API key not set. Returning zero vectors.")
        return [np.zeros(openai_dim) for _ in texts]
    
    # Validar y limpiar textos
    texts = [str(t).strip() if isinstance(t, str) else str(t) for t in texts]
    if not any(texts):
        print("Warning: All texts are empty. Returning zero vectors.")
        return [np.zeros(openai_dim) for _ in texts]
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.embeddings.create(
            input=texts,
            model=openai_model
        )
        embeddings_dict = {item.index: item.embedding for item in response.data}
        # Retornar en el mismo orden
        embeddings = [np.array(embeddings_dict.get(i, np.zeros(openai_dim))) for i in range(len(texts))]
        return embeddings
    except Exception as e:
        print(f"ERROR computing OpenAI embeddings batch: {e}")
        return [np.zeros(openai_dim) for _ in texts]

def compute_openai_embedding(text):
    """Wrapper para un único texto usando batch."""
    result = compute_openai_embedding_batch([text])
    return result[0] if result else np.zeros(3072)


# This version is used by RAG's CombinedRetriever specifically
def compute_combined_embedding_for_rag(text, model_general,
                                        model_general2,
                                        model_medico, model_biobert, model_gemma):
    """
    Computes combined embedding for RAG using MiniLM, MPNet, MedEmbed, and OpenAI (text-embedding-3-large).
    Ensure models passed are loaded SentenceTransformer instances and openai_func works.
    """
    try:
        dim_gen = 384   # sentence-transformers/all-MiniLM-L12-v2
        dim_gen2 = 768  # sentence-transformers/all-mpnet-base-v2
        dim_med = 1024  # abhinand/MedEmbed-large-v0.1
        dim_biobert = 768 # dmis-lab/biobert-base-cased-v1.1
        dim_gemma = 768 # google/embeddinggemma-300m
        total_dim = dim_gen + dim_gen2 + dim_med + dim_biobert + dim_gemma # Should be 3712

        # Encode using SentenceTransformers
        emb_gen = model_general.encode(text, convert_to_numpy=True) if model_general else np.zeros(dim_gen)
        emb_gen2 = model_general2.encode(text, convert_to_numpy=True) if model_general2 else np.zeros(dim_gen2)
        emb_med = model_medico.encode(text, convert_to_numpy=True) if model_medico else np.zeros(dim_med)
        emb_gemma = model_gemma.encode(text, convert_to_numpy=True) if model_gemma else np.zeros(dim_gemma)
        emb_biobert = model_biobert.encode(text, convert_to_numpy=True) if model_biobert else np.zeros(dim_biobert)

        # Compute OpenAI embedding
        #emb_openai = openai_func(text) if openai_func else np.zeros(dim_openai)

        # Ensure vectors have correct shapes, fill with zeros if encoding failed or model missing
        emb_gen = emb_gen if emb_gen.shape == (dim_gen,) else np.zeros(dim_gen)
        emb_gen2 = emb_gen2 if emb_gen2.shape == (dim_gen2,) else np.zeros(dim_gen2)
        emb_med = emb_med if emb_med.shape == (dim_med,) else np.zeros(dim_med)
        emb_biobert = emb_biobert if emb_biobert.shape == (dim_biobert,) else np.zeros(dim_biobert)
        emb_gemma = emb_gemma if emb_gemma.shape == (dim_gemma,) else np.zeros(dim_gemma)
        #emb_openai = emb_openai if emb_openai.shape == (dim_openai,) else np.zeros(dim_openai)

        # Apply weights (currently all 1.0) - Consistent with script
        emb_gen_weighted = 1.0 * emb_gen
        emb_gen2_weighted = 1.0 * emb_gen2
        emb_med_weighted = 1.0 * emb_med
        emb_biobert_weighted = 1.0 * emb_biobert
        emb_gemma_weighted = 1.0 * emb_gemma

        #emb_openai_weighted = 1.0 * emb_openai

        # Concatenate (Order matches script implied order)
        combined = np.concatenate([
            emb_gen_weighted,
            emb_gen2_weighted,
            emb_med_weighted,
            emb_biobert_weighted,
            emb_gemma_weighted,
            #emb_openai_weighted
        ], axis=0)

        # Normalization
        norm = np.linalg.norm(combined)
        if norm > 0: combined = combined / norm
        return combined

    except Exception as e:
        print(f"Error computing combined embedding for RAG text: {text[:100]}... Error: {e}")
        return np.zeros(total_dim)

# --- Clustering Specific Functions ---
# (Keep cluster_and_evaluate, select_best_clustering, normalize_embeddings)
# Note: cluster_attributes uses a different combined embedding function if needed

def cluster_and_evaluate(attribute_embeddings, max_k: int):
    X = normalize(attribute_embeddings, norm="l2")
    X = PCA(n_components=min(50, X.shape[1]), random_state=42).fit_transform(X)
    X = normalize(X, norm="l2")
    clustering_results = []
    min_k = 3
    if max_k < min_k:
        max_k = min_k

    cluster_range = range(min_k, max_k + 1)

    clustering_algorithms = {
        "KMeans": {
            "model": KMeans,
            "params": {"n_clusters": cluster_range, "random_state": [42], "n_init": [20]},
        },
        "Agglomerative": {
            "model": AgglomerativeClustering,
            "params": {
                "n_clusters": cluster_range,
                "metric": ["cosine", "euclidean", "l2"],
                "linkage": ["average", "complete"],  # ward solo con euclidean
            },
        },
    }

    if X.shape[0] < 3:
        return []

    for algo_name, algo in clustering_algorithms.items():
        Model = algo["model"]
        for params in ParameterGrid(algo["params"]):
            # Restricción de ward
            if algo_name == "Agglomerative":
                if params["linkage"] == "ward":
                    continue
                # ward no está en la lista; si lo añades, exige metric euclidean

            try:
                model = Model(**params)
                cluster_labels = model.fit_predict(X)

                # Excluir ruido para evaluación
                mask = cluster_labels != -1
                if mask.sum() < 3:
                    continue
                X_eval = X[mask]
                y_eval = cluster_labels[mask]

                n_clusters_ = len(set(y_eval))
                if n_clusters_ < 2 or len(X_eval) <= n_clusters_:
                    continue

                cluster_counts = Counter(y_eval)
                if any(s < 2 for s in cluster_counts.values()):
                    continue

                # Métricas internas (coherentes con X normalizado)
                metric_for_eval = params.get("metric", "euclidean")
                try:
                    sil = silhouette_score(X_eval, y_eval, metric=metric_for_eval)
                except Exception:
                    sil = -1

                # CH/DB: solo tienen sentido en euclidean
                try:
                    ch = calinski_harabasz_score(X_eval, y_eval)
                    db = davies_bouldin_score(X_eval, y_eval)
                except Exception:
                    ch, db = -1, -1

                clustering_results.append({
                    "algorithm": algo_name,
                    "params": params,
                    "n_clusters": n_clusters_,
                    "cluster_sizes": dict(cluster_counts),
                    "silhouette_score": sil,
                    "calinski_harabasz_score": ch,
                    "davies_bouldin_score": db,
                    "labels": cluster_labels.tolist(),
                })

            except Exception as e:
                print(f"ERROR clustering {algo_name} {params}: {e}")

    return clustering_results

def consensus_score(labels, top1_resource_idx):
    labels = np.array(labels)
    score_hits = 0
    total = 0
    for c in set(labels):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        total += len(idx)
        maj = Counter(top1_resource_idx[idx]).most_common(1)[0][0]
        score_hits += np.sum(top1_resource_idx[idx] == maj)
    return score_hits / total if total > 0 else 0.0


def select_best_clustering_silhouette(clustering_results, weights={'silhouette': 1, 'calinski': 1, 'davies_positive_contribution': 1}):
    best_result = max(clustering_results, key=lambda x: x['silhouette_score'])
    return best_result['labels'], best_result

 # --- util: ranking (0 = mejor) ---
def rank_indices(values, higher_is_better=True):
    values = np.asarray(values, dtype=float)
    order = np.argsort(-values) if higher_is_better else np.argsort(values)
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(len(values))
    return ranks, order

def select_best_clustering(
    clustering_results,
    weights=None,
    top_frac=0.2,        # “mejores” = top 20% por métrica
    top_n=None,          # si lo pasas, sobrescribe top_frac
    use_intersection=False,  # True = intersección de top por métrica; False = unión
    rrf_k=60,            # parámetro estándar de RRF
):
    """
    Selecciona el mejor clustering en 2 fases:
      (1) shortlist: top-N por cada métrica
      (2) fusión: agrega rankings de las métricas y elige el mejor

    Espera que cada elemento tenga:
      'silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'labels'
    """

    if weights is None:
        weights = {"silhouette": 1.0, "calinski": 1.0, "davies": 1.0}

    if not clustering_results:
        return None, None

    # --- 1) filtrar válidos y finitos ---
    valid = []
    for r in clustering_results:
        try:
            sil = float(r.get("silhouette_score", -1))
            ch  = float(r.get("calinski_harabasz_score", np.nan))
            db  = float(r.get("davies_bouldin_score", np.nan))
            labs = r.get("labels", None)
        except Exception:
            continue

        if labs is None:
            continue
        if sil == -1:
            continue
        if not (np.isfinite(sil) and np.isfinite(ch) and np.isfinite(db)):
            continue

        valid.append(r)

    if not valid:
        return None, None

    n = len(valid)
    m = top_n if top_n is not None else max(1, int(math.ceil(top_frac * n)))

   
    sil_vals = [r["silhouette_score"] for r in valid]
    ch_vals  = [r["calinski_harabasz_score"] for r in valid]
    db_vals  = [r["davies_bouldin_score"] for r in valid]

    sil_rank, sil_order = rank_indices(sil_vals, higher_is_better=True)
    ch_rank,  ch_order  = rank_indices(ch_vals,  higher_is_better=True)
    db_rank,  db_order  = rank_indices(db_vals,  higher_is_better=False)  # menor DB = mejor

    top_sil = set(sil_order[:m].tolist())
    top_ch  = set(ch_order[:m].tolist())
    top_db  = set(db_order[:m].tolist())

    if use_intersection:
        candidates = top_sil & top_ch & top_db
        if not candidates:
            # fallback sensato: si la intersección es vacía, usar unión
            candidates = top_sil | top_ch | top_db
    else:
        candidates = top_sil | top_ch | top_db

    # --- 2) fusión de rankings (RRF) sobre candidatos ---
    best_idx = None
    best_score = -1.0

    for i in candidates:
        score = 0.0
        score += weights.get("silhouette", 0.0) * (1.0 / (rrf_k + sil_rank[i] + 1))
        score += weights.get("calinski",   0.0) * (1.0 / (rrf_k + ch_rank[i]  + 1))
        score += weights.get("davies",     0.0) * (1.0 / (rrf_k + db_rank[i]  + 1))

        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        best_idx = max(range(len(valid)), key=lambda i: valid[i].get("silhouette_score", -1))

    best_result = valid[best_idx]
    # opcional: adjuntar info útil para depurar
    best_result = dict(best_result)
    best_result["selector_debug"] = {
        "shortlist_m": m,
        "candidate_pool_size": len(candidates),
        "ranks": {
            "silhouette": int(sil_rank[best_idx]),
            "calinski": int(ch_rank[best_idx]),
            "davies": int(db_rank[best_idx]),
        },
        "rrf_score": float(best_score),
    }

    return best_result["labels"], best_result

def select_best_clustering_borda(clustering_results, weights=None):
    """
    Selecciona el mejor clustering por agregación de rankings (Borda count).
    - silhouette: mayor es mejor
    - calinski: mayor es mejor
    - davies: menor es mejor
    """
    if not clustering_results:
        return None, None

    valid = [r for r in clustering_results if r.get("silhouette_score", -1) != -1]
    if not valid:
        return None, None

    if weights is None:
        weights = {"silhouette": 1.0, "calinski": 1.0, "davies": 1.0}

    sil = np.array([r["silhouette_score"] for r in valid], dtype=float)
    ch  = np.array([r["calinski_harabasz_score"] for r in valid], dtype=float)
    db  = np.array([r["davies_bouldin_score"] for r in valid], dtype=float)

    # ranks: 0 = mejor
    sil_rank = sil.argsort()[::-1].argsort()
    ch_rank  = ch.argsort()[::-1].argsort()
    db_rank  = db.argsort().argsort()

    score = (
        weights.get("silhouette", 0) * sil_rank +
        weights.get("calinski", 0)   * ch_rank +
        weights.get("davies", 0)     * db_rank
    )

    best_idx = int(np.argmin(score))
    best = valid[best_idx]
    return best["labels"], best


def compute_base_embeddings(preprocessed_texts, embedding_models_instances, embedding_list):
    """
    Calcula embeddings base (general, general2, medico, biobert, openai) UNA sola vez
    para todos los textos de atributos.
    Devuelve: dict base_name -> np.array [n_atributos, dim]
    """
    base_embs = {}
    embedding_list_default = ["general", 
                         "general2",
                         "medico", "medico2", "biobert", "gemma", "modern_medico"]
    n = len(preprocessed_texts)
    print("Preprocessed texts :",n)

    if not embedding_list:
        embedding_list = embedding_list_default
    print("Embedding list:")
    print(embedding_list)
    if n == 0:
        return base_embs

    for base_name, spec in BASE_EMBEDDING_SPECS.items():
        model_key = spec["model_key"]
        #dim = spec["dim"]
        model_inst = embedding_models_instances[base_name]
        print(f"Computing base embeddings for '{base_name}' using '{model_key}'")

        if base_name in embedding_list:
            if model_inst is None:
                print(f"Warning: model '{model_key}' not loaded. Using zeros.")
                base_embs[base_name] = np.zeros((n, model_inst.get_max_seq_length()), dtype=np.float32)
            else:
                print(base_name)
                print("Computing embeddings...")
                base_embs[base_name] = model_inst.encode(
                    preprocessed_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ).astype(np.float32)
        else:
            print(f"Warning: unknown base embedding name '{base_name}'. Skipping.")
    print(base_embs)
    return base_embs


def build_embeddings_for_config(base_embs, config):
    """
    A partir de un dict con embeddings base construye el embedding combinado
    según la configuración (lista de componentes).
    """
    parts = []
    for comp_name in config["components"]:
        if comp_name not in base_embs:
            print(f"Warning: component '{comp_name}' not found in base_embs. Skipping config.")
            return None
        parts.append(base_embs[comp_name])

    if not parts:
        return None

    combined = np.concatenate(parts, axis=1)
    #combined = normalize(combined,norm="l2")
    return combined


def compact_values(vals, max_items=15, max_chars=500):
    if not vals:
        return ""
    # fuerza string simple
    vals_str = [str(v) for v in vals[:max_items]]
    vals = sorted(vals_str)
    s = ", ".join(vals_str)
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s



def cluster_attributes(attributes, loaded_embedding_models, max_k: int):
    """
    Prueba múltiples configuraciones de embeddings (individuales y combinadas),
    evalúa clustering con multi-métrica y devuelve:
      - clusters,
      - nombre de la configuración ganadora,
      - info de la configuración ganadora.
    """
    if not attributes:
        return {}, None, {}
    # 1) Preparar textos
    attribute_texts = []
    valid_attributes = []
    for i, attr in enumerate(attributes):
        try:
            attr_name = attr.get('Attribute name', f'Unknown {i}')
            desc = attr.get('Description', '')
            vals = attr.get('Values', [])
            v_type = detect_value_type(vals)
            vals_preview = compact_values(vals, max_items=15)
            combined_text = f"Attribute: {attr_name}. Description: {desc}. ValueType: {v_type}."
            attribute_texts.append(combined_text)
            valid_attributes.append(attr)
        except Exception as e:
            print(f"Warn: Skip attr {i} text prep: {e}")
    if not attribute_texts:
        return {}, None, {}
    preprocessed_texts = [preprocess_text_dense(text) for text in attribute_texts]
    # 2) Embeddings base una sola vez
    base_embs = compute_base_embeddings(preprocessed_texts, embedding_models_instances, loaded_embedding_models)
    if not base_embs:
        print("ERROR: No base embeddings computed.")
        return {}, None, {}
    # 3) Probar todas las configuraciones de embedding
    all_clustering_results = []
    for config_name, config in EMBEDDING_CONFIGS.items():
        print(f"\n=== Clustering with Embedding Config: {config_name} ({config['components']}) ===")
        attribute_embeddings = build_embeddings_for_config(base_embs, config)
        if attribute_embeddings is None:
            print(f"Skipping config '{config_name}' due to missing components.")
            continue
        if attribute_embeddings.shape[0] < 3:
            print(f"Skipping config '{config_name}': not enough samples.")
            continue
        clustering_results = cluster_and_evaluate(attribute_embeddings, max_k)                 
        for res in clustering_results:
            res["embedding_config"] = config_name
            res["embedding_components"] = config["components"]
        all_clustering_results.extend(clustering_results)
    if not all_clustering_results:
        print("No clustering results for any embedding config.")
        return {}, None, {}
    # 4) Seleccionar el mejor clustering (multi-métrica)
    selected_labels, best_result_info = select_best_clustering_silhouette(all_clustering_results)
    if selected_labels is None:
        return {}, None, {}
    best_embedding_config = best_result_info.get("embedding_config", None)
    best_components = best_result_info.get("embedding_components", [])
    print(f"\n*** BEST EMBEDDING CONFIG: {best_embedding_config} -> {best_components} ***")
    print(f"Best clustering: algo={best_result_info['algorithm']}, params={best_result_info['params']}")
    # 5) Construir clusters
    clusters = defaultdict(list)
    for idx, label in enumerate(selected_labels):
        if idx < len(valid_attributes) and label != -1:
            clusters[label].append(valid_attributes[idx])
    # Normalizar cluster_sizes para JSON
    raw_cluster_sizes = best_result_info.get("cluster_sizes", {})
    cluster_sizes_json = {int(k): int(v) for k, v in raw_cluster_sizes.items()}
    embedding_config_info = {
        "Winner Embedding Config": best_embedding_config,
        "Components": best_components,
        "Best Clustering": {
            "algorithm": best_result_info["algorithm"],
            "params": best_result_info["params"],  # normalmente ya es JSON-safe
            "n_clusters": int(best_result_info["n_clusters"]),
            "cluster_sizes": cluster_sizes_json,
            "scores": {
                "silhouette": float(best_result_info["silhouette_score"]),
                "calinski_harabasz": float(best_result_info["calinski_harabasz_score"]),
                "davies_bouldin": float(best_result_info["davies_bouldin_score"]),
            },
        },
    }
    return clusters, best_embedding_config, embedding_config_info


# --- RAG Specific Functions ---
def schema_to_natural_text(resource_type, schema):
    if not isinstance(schema, dict):
        return ""
    props = schema.get("properties", {})
    lines = [f"FHIR resource {resource_type}. It defines the following attributes:"]
    for name, info in props.items():
        desc = info.get("description", "")
        typ  = info.get("type", "")
        if desc:
            lines.append(f"- {name}: {desc} (type: {typ}).")
        else:
            lines.append(f"- {name} (type: {typ}).")
    return " ".join(lines)

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
        print(type(res))
        print(type(schema))
        passages.update(build_fhir_resource_passage_only(schema, res))


    if not passages:
         print("Warning: No passages were successfully created from resources and schemas.")

    return passages


import math
from typing import List, Union

def chunk_text_by_hf_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int = 0,
    return_token_ids: bool = False,
) -> List[Union[str, List[int]]]:
    if not text:
        return []

    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    L = len(ids)

    max_tokens = int(max_tokens)
    overlap_tokens = int(overlap_tokens)

    if max_tokens <= 0:
        return [ids] if return_token_ids else [text]

    if L <= max_tokens:
        return [ids] if return_token_ids else [text]

    step = max(1, max_tokens - overlap_tokens)

    chunks = []
    for start in range(0, L, step):
        window = ids[start:start + max_tokens]
        if not window:
            break
        chunks.append(window if return_token_ids else tokenizer.decode(window, skip_special_tokens=True))
        if start + max_tokens >= L:
            break

    return chunks



from typing import Dict
from haystack import Document  # o la ruta correcta en tu proyecto
from collections import defaultdict

def build_cluster_top_resources_hybrid(
    cluster_content: dict,
    cluster_query_text: str,
    retriever,
    top_k: int = 5,
    base_top_k: int = 30,          # recoger más candidatos del cluster
    per_attr_top_k: int = 12,      # retrieval por atributo
    vote_m: int = 8,               # usa top-8 como “evidencia” por atributo
    alpha: float = 1.0,            # peso de ranking base
    beta: float = 0.35,            # peso de votos por atributo
    chunk_size: int = 256,
    overlap: int = 40,
):
    # 1) Ranking base por cluster (tu método que ya funcionaba)
    base_docs = retriever.retrieve_chunked(
        cluster_query_text,
        top_k=base_top_k,
        per_chunk_top_k=max(20, base_top_k),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # base_score por recurso (usa score del doc devuelto; si es RRF, mejor)
    base_score = defaultdict(float)
    for rank, d in enumerate(base_docs):
        rid = (d.meta or {}).get("resource_id")
        if not rid:
            continue
        s = float(getattr(d, "score", 0.0) or 0.0)
        # si no hay score útil, usa un prior por rank
        if s <= 0:
            s = 1.0 / (rank + 1)
        base_score[rid] = max(base_score[rid], s)

    # 2) Votos por atributo (Borda sobre top-m)
    vote_score = defaultdict(float)
    for name, desc in cluster_content.items():
        q = f"Attribute: {name}. Description: {desc}."
        docs = retriever.retrieve(q, top_k=per_attr_top_k)

        # dedup por resource_id en este atributo (mejor score)
        best = {}
        for d in docs:
            rid = (d.meta or {}).get("resource_id")
            if not rid:
                continue
            s = float(getattr(d, "score", 0.0) or 0.0)
            if (rid not in best) or (s > best[rid]):
                best[rid] = s

        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:vote_m]
        for r, _s in ranked:
            # Borda: más puntos cuanto más arriba
            vote_score[r] += (vote_m - ranked.index((r, _s)))

    # 3) Combinar scores
    candidates = set(base_score.keys()) | set(vote_score.keys())
    combined = []
    for r in candidates:
        combined.append((r, alpha * base_score.get(r, 0.0) + beta * vote_score.get(r, 0.0)))

    combined.sort(key=lambda x: x[1], reverse=True)
    return [{"resource": r} for r, _ in combined[:top_k]]

def create_documents_from_passages_with_token_chunks(
    passages: Dict[str, str],
    retriever,
    chunk_tokens: int = 256,
    overlap_tokens: int = 30
):
    """
    Crea Documents por recurso a partir de 'passages', chunkizando por tokens
    con tamaño 'chunk_tokens' y solapamiento 'overlap_tokens'. Cada chunk es un Document.

    Nota: NO hacemos packing agresivo; en retrieval semántico suele rendir mejor
    tener chunks más "puros" y luego agregar por resource_id.
    """
    haystack_documents = []

    models = retriever.active_models
    if not models:
        raise ValueError("active_models está vacío.")

    # Selección de modelo referencia para tokenización
    key0 = models[0]
    ref_model = {
        "biobert": retriever.model_biobert,
        "medico": retriever.model_medico,
        "medico2": retriever.model_medico2,
        "general": retriever.model_general,
        "general2": retriever.model_general2,
        "modern_medico": retriever.modern_medico,
        "gemma": retriever.model_gemma,
    }.get(key0, None)

    if ref_model is None:
        raise ValueError("No hay modelo disponible para extraer tokenizer de referencia.")

    tokenizer = getattr(ref_model, "tokenizer", None)
    if tokenizer is None:
        tokenizer = ref_model._first_module().tokenizer

    model_max_tokens = getattr(ref_model, "max_seq_length", None) or getattr(tokenizer, "model_max_length", 512)

    # margen para especiales
    special_margin = 2
    hard_limit = max(8, model_max_tokens - special_margin)

    # chunk_tokens debe respetar el hard_limit
    chunk_tokens = min(int(chunk_tokens), hard_limit)
    overlap_tokens = int(overlap_tokens)

    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens // 4)  # fallback seguro

    for resource_id, text in passages.items():
        if not text or not isinstance(text, str):
            continue

        chunk_ids_list = chunk_text_by_hf_tokens(
            text=text,
            tokenizer=tokenizer,
            max_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            return_token_ids=True,
        )

        for chunk_id, ids in enumerate(chunk_ids_list):
            content = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # ID ESTABLE para RRF/fusión
            doc_id = f"{resource_id}::{chunk_id}"

            haystack_documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    meta={
                        "resource_id": resource_id,
                        "chunk_id": chunk_id,
                        "chunk_tokens": len(ids),
                        "overlap_tokens": overlap_tokens,
                        "model_limit": hard_limit,
                        "chunk_tokens_param": chunk_tokens,
                    },
                )
            )

    return haystack_documents


def rrf_fuse_resource_rankings(rankings, k=60):
    fused = defaultdict(float)
    for rlist in rankings:
        for rank, item in enumerate(rlist):
            rid = item[0]
            fused[rid] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)

def build_attr_query(attr_name: str, desc: str, v_type: str = "") -> str:
    # recorta descripción
    desc = (desc or "").strip()
    desc_short = " ".join(desc.split())  # 18 tokens aprox
    parts = [f"field: {attr_name}"]
    if v_type:
        parts.append(f"type: {v_type}")
    if desc_short:
        parts.append(f"meaning: {desc_short}")
    return ". ".join(parts) + "."

class CombinedRetriever(InMemoryEmbeddingRetriever):
    """Custom Haystack Retriever using a combination of embedding models."""
    def __init__(
        self,
        document_store,
        model_general=None,
        model_general2=None,
        model_medico=None,
        model_medico2=None,
        #model_openai=None,
        model_biobert=None, 
        model_gemma=None,
        modern_medico=None,
        active_models=None,
        num_models=None,
        **kwargs
    ):
        super().__init__(document_store=document_store, **kwargs)
        print("Initializing CombinedRetriever...")

        self.model_general = model_general
        self.model_general2 = model_general2
        self.model_medico = model_medico
        self.model_medico2 = model_medico2
        #self.model_openai = model_openai
        self.model_biobert = model_biobert  
        self.model_gemma = model_gemma    
        self.modern_medico = modern_medico    



        default_order = ["gemma", "modern_medico"]

        # Resolver qué modelos están activos
        if active_models is not None:
            # Normalizamos nombres
            self.active_models = [m.lower() for m in active_models]
        elif num_models is not None:
            if not (1 <= num_models <= 4):
                raise ValueError("num_models debe estar entre 1 y 4.")
            self.active_models = default_order[:num_models]
        else:
            # Por defecto, usar los 4
            self.active_models = default_order

        self.total_dim = self.get_total_dim(self.active_models)

        print(f"CombinedRetriever initialized with models: {self.active_models}")
        print(f"Total embedding dimension: {self.total_dim}")

    def get_total_dim(self,active_models):
        self.model_dims = {
            "general": embedding_models_instances["general"].get_sentence_embedding_dimension() if self.model_general is not None else 0,
            "general2": embedding_models_instances["general2"].get_sentence_embedding_dimension() if self.model_general2 is not None else 0,
            "medico": embedding_models_instances["medico"].get_sentence_embedding_dimension() if self.model_medico is not None else 0,
            "medico2": embedding_models_instances["medico2"].get_sentence_embedding_dimension() if self.model_medico2 is not None else 0,
            "biobert": embedding_models_instances["biobert"].get_sentence_embedding_dimension() if self.model_biobert is not None else 0,     
            "gemma": embedding_models_instances["gemma"].get_sentence_embedding_dimension() if self.model_gemma is not None else 0,
            "modern_medico": embedding_models_instances["modern_medico"].get_sentence_embedding_dimension() if self.modern_medico is not None else 0,      
            #"modern_medico": 1024,
           # "openai": 3072,
        }

        if active_models is not None:
            # Normalizamos nombres
            self.active_models = [m.lower() for m in active_models]
        self.total_dim = sum(self.model_dims[m] for m in self.active_models)

        return self.total_dim
    
    def embed_documents(self, docs: list[Document]) -> np.ndarray:
        """Embeds documents using the selected combination of models."""
        embeddings = []
        print(f"Embedding {len(docs)} documents using CombinedRetriever...")
        
        # Separar textos válidos
        valid_texts = []
        valid_indices = []

        for i, doc in enumerate(docs):
            if not getattr(doc, "content", None) or not isinstance(doc.content, str):
                print(f"Warning: Document {i} has invalid content. Generating zero embedding.")
                embeddings.append((i, np.zeros(self.total_dim, dtype=np.float32)))
            else:
                valid_texts.append(doc.content)
                valid_indices.append(i)

        if valid_texts:
            print("Processing selected embedding models:", self.active_models)
            for idx, (doc_idx, text) in enumerate(zip(valid_indices, valid_texts)):
                parts = []
                # Modelo general
                if "general" in self.active_models:
                    if self.model_general is not None:
                        emb_gen = self.model_general.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_gen = np.zeros(self.model_dims["general"], dtype=np.float32)
                    parts.append(emb_gen)

                # Modelo general2
                if "general2" in self.active_models:
                    if self.model_general2 is not None:
                        emb_gen2 = self.model_general2.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_gen2 = np.zeros(self.model_dims["general2"], dtype=np.float32)
                    parts.append(emb_gen2)

                # Modelo médico
                if "medico" in self.active_models:
                    if self.model_medico is not None:
                        emb_med = self.model_medico.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_med = np.zeros(self.model_dims["medico"], dtype=np.float32)
                    parts.append(emb_med)
                
                if "medico2" in self.active_models:
                    if self.model_medico2 is not None:
                        emb_med2 = self.model_medico2.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_med2 = np.zeros(self.model_dims["medico2"], dtype=np.float32)
                    parts.append(emb_med2)

                if "biobert" in self.active_models:
                    if self.model_biobert is not None:

                        emb_bio = self.model_biobert.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_bio = np.zeros(self.model_dims["biobert"], dtype=np.float32)
                    parts.append(emb_bio)
                
                if "gemma" in self.active_models:
                    if self.model_gemma is not None:
                        emb_gemma = self.model_gemma.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_gemma = np.zeros(self.model_dims["gemma"], dtype=np.float32)
                    parts.append(emb_gemma)

                if "modern_medico" in self.active_models:
                    if self.modern_medico is not None:
                        emb_modern_medico = self.modern_medico.encode(text, convert_to_numpy=True,normalize_embeddings=True)
                    else:
                        emb_modern_medico = np.zeros(self.model_dims["modern_medico"], dtype=np.float32)
                    parts.append(emb_modern_medico)

                # Concatenar
                combined = self._combine_parts(parts)
                embeddings.append((doc_idx, combined))

        # Ordenar por índice original
        embeddings.sort(key=lambda x: x[0])

        # Cada e[1] es ahora 1D (total_dim,)
        result = np.vstack([e[1] for e in embeddings]).astype(np.float32)
        print("Document embedding finished.")
        return result
    
    def _get_chunking_model(self):
        # elige el modelo más “restrictivo” / representativo
        if "biobert" in self.active_models and self.model_biobert is not None:
            return self.model_biobert
        if "medico2" in self.active_models and self.model_medico2 is not None:
            return self.model_medico2
        if "medico" in self.active_models and self.model_medico is not None:
            return self.model_medico
        if "general2" in self.active_models and self.model_general2 is not None:
            return self.model_general2
        if "general" in self.active_models and self.model_general is not None:
            return self.model_general
        return None
    
    def chunk_text_with_model(self, text: str, chunk_size: int = 460, overlap: int = 30) -> list[str]:
        model = self._get_chunking_model()
        if model is None:
            return [text]

        tok = model._first_module().tokenizer
        model_limit = getattr(model, "max_seq_length", None) or getattr(tok, "model_max_length", 512)

        special_margin = 2
        hard_limit = max(8, int(model_limit) - special_margin)

        chunk_size = min(int(chunk_size), hard_limit)
        overlap = int(overlap)
        if overlap >= chunk_size:
            overlap = max(0, chunk_size // 4)

        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]

        if len(ids) <= chunk_size:
            return [text]

        step = max(1, chunk_size - overlap)
        chunks = []
        for i in range(0, len(ids), step):
            chunk_ids = ids[i:i + chunk_size]
            if not chunk_ids:
                break
            chunks.append(tok.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            if i + chunk_size >= len(ids):
                break

        return chunks

    def retrieve_resources_exact(
        self,
        query: str,
        top_k_resources: int = 5,
        chunk_size: int = 256,
        overlap: int = 40,
        max_chunks: int = 50,
        agg: str = "mean_top2",  # "max" o "mean_top2"
        filters=None,
    ):
        # IMPORTANT: alinear preprocesado con docs
        processed = preprocess_text_dense(query) if query and isinstance(query, str) else ""
        if not processed.strip():
            return []

        q_chunks = self.chunk_text_with_model(processed, chunk_size=chunk_size, overlap=overlap)
        q_chunks = [c for c in q_chunks if c.strip()][:chunk_size]

        # necesitamos TODOS los docs (chunks) porque el corpus es pequeño
        N = self.document_store.count_documents()

        scores_by_res = defaultdict(list)

        for ch in q_chunks:
            q_emb = self.embed_queries([ch])[0]
            docs = self.document_store.embedding_retrieval(
                q_emb.tolist(),
                top_k=N,
                filters=filters,
                scale_score=False
            )
            for d in docs:
                rid = (d.meta or {}).get("resource_id")
                if rid is None:
                    continue
                scores_by_res[rid].append(float(d.score))

        res_scores = []
        for rid, scs in scores_by_res.items():
            scs.sort(reverse=True)
            if not scs:
                continue
            if agg == "max":
                s = scs[0]
            else:  # mean_top2 (muy robusto)
                top2 = scs[:2]
                s = float(np.mean(top2))
            res_scores.append((rid, s))

        res_scores.sort(key=lambda x: x[1], reverse=True)

        # formato compatible (si no quieres score, lo quitas fuera)
        return [{"resource": rid, "score": s} for rid, s in res_scores[:top_k_resources]]
    
    # def retrieve_resources_exact(
    #     self,
    #     query: str,
    #     top_k_resources: int = 5,
    #     chunk_size: int = 256,
    #     overlap: int = 40,
    #     max_chunks: int = 50,
    #     agg: str = "mean_top2",
    #     filters=None,
    #     per_chunk_top_k_docs: int = 300,  # NUEVO: suficiente y evita ruido
    # ):
    #     processed = preprocess_text_dense(query) if query and isinstance(query, str) else ""
    #     if not processed.strip():
    #         return []

    #     q_chunks = self.chunk_text_with_model(processed, chunk_size=chunk_size, overlap=overlap)
    #     q_chunks = [c for c in q_chunks if c.strip()][:chunk_size]

    #     N = self.document_store.count_documents()
    #     top_k_docs = min(N, per_chunk_top_k_docs)

    #     scores_by_res = defaultdict(list)

    #     for ch in q_chunks:
    #         q_emb = self.embed_queries([ch])[0]
    #         docs = self.document_store.embedding_retrieval(
    #             q_emb.tolist(),
    #             top_k=top_k_docs,
    #             filters=filters,
    #             scale_score=False
    #         )

    #         # CLAVE: mejor score por recurso en este chunk de query
    #         best_by_rid = {}
    #         for d in docs:
    #             rid = (d.meta or {}).get("resource_id")
    #             if rid is None:
    #                 continue
    #             s = float(d.score)
    #             if (rid not in best_by_rid) or (s > best_by_rid[rid]):
    #                 best_by_rid[rid] = s

    #         # acumulas como mucho 1 score por recurso y query-chunk
    #         for rid, s in best_by_rid.items():
    #             scores_by_res[rid].append(s)

    #     res_scores = []
    #     for rid, scs in scores_by_res.items():
    #         scs.sort(reverse=True)
    #         if not scs:
    #             continue
    #         if agg == "max":
    #             s = scs[0]
    #         else:  # mean_top2
    #             s = float(np.mean(scs[:2]))
    #         res_scores.append((rid, s))

    #     res_scores.sort(key=lambda x: x[1], reverse=True)
    #     return [{"resource": rid, "score": s} for rid, s in res_scores[:top_k_resources]]
    
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        print(queries)
        print(f"Embedding {len(queries)} queries using CombinedRetriever...")

        out = np.zeros((len(queries), self.total_dim), dtype=np.float32)

        for i, query in enumerate(queries):
            if not query or not isinstance(query, str):
                continue

            text = preprocess_text_dense(query)
            parts = []

            if "general" in self.active_models:
                parts.append(self.model_general.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_general is not None else np.zeros(self.model_dims["general"], np.float32))

            if "general2" in self.active_models:
                parts.append(self.model_general2.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_general2 is not None else np.zeros(self.model_dims["general2"], np.float32))

            if "medico" in self.active_models:
                parts.append(self.model_medico.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_medico is not None else np.zeros(self.model_dims["medico"], np.float32))

            if "medico2" in self.active_models:
                parts.append(self.model_medico2.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_medico2 is not None else np.zeros(self.model_dims["medico2"], np.float32))

            if "biobert" in self.active_models:
                parts.append(self.model_biobert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_biobert is not None else np.zeros(self.model_dims["biobert"], np.float32))

            if "gemma" in self.active_models:
                parts.append(self.model_gemma.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.model_gemma is not None else np.zeros(self.model_dims["gemma"], np.float32))

            if "modern_medico" in self.active_models:
                parts.append(self.modern_medico.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            if self.modern_medico is not None else np.zeros(self.model_dims["modern_medico"], np.float32))

            out[i] = self._combine_parts(parts)

        return out


    def retrieve(self, query, top_k=5, filters=None):
        print(f"Retrieving top-{top_k} for query: {query[:100]}...")
        print("Query preprocessing...")
        print(f"Original query: {query}")
        # 1) Embedding de la query
        query_emb = self.embed_queries([query])[0]  # np.ndarray (dim,)

        # 2) Convertir a lista de floats para Haystack
        query_emb_list = query_emb.tolist()

        # 3) Llamar al método del document_store
        results = self.document_store.embedding_retrieval(
            query_emb_list,
            top_k=top_k,
            filters=filters,
            scale_score=False
        )

        print(f"Retrieved {len(results)} documents from store.")
        return results

    def _combine_parts(self, parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.zeros(self.total_dim, dtype=np.float32)

        v = parts[0].astype(np.float32) if len(parts) == 1 else np.concatenate(parts, axis=0).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def _rrf_fuse(self, results_per_chunk, k: int = 60):
        fused = defaultdict(float)
        doc_by_id = {}

        for docs in results_per_chunk:
            for rank, doc in enumerate(docs):
                # ID estable: doc.id si existe, sino meta
                doc_id = getattr(doc, "id", None)
                if not doc_id:
                    rid = (getattr(doc, "meta", {}) or {}).get("resource_id")
                    cid = (getattr(doc, "meta", {}) or {}).get("chunk_id")
                    doc_id = f"{rid}::{cid}" if rid is not None else str(id(doc))

                doc_by_id[doc_id] = doc
                fused[doc_id] += 1.0 / (k + rank + 1)

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        out = []
        for doc_id, rrf_score in ranked:
            d = doc_by_id[doc_id]
            d.score = float(rrf_score)
            out.append(d)
        return out

    def retrieve_chunked(
        self,
        query: str,
        top_k: int = 5,
        per_chunk_top_k: int = 20,
        chunk_size: int = 256,
        overlap: int = 40,
        max_chunks: int = 50,
        filters=None,
    ):
        processed = preprocess_text_dense(query) if query and isinstance(query, str) else ""
        if not processed.strip():
            return []

        chunks = self.chunk_text_with_model(processed, chunk_size=chunk_size, overlap=overlap)
        chunks = [c for c in chunks if c.strip()][:max_chunks]

        results_per_chunk = []
        for ch in chunks:
            q_emb = self.embed_queries([ch])[0]
            docs = self.document_store.embedding_retrieval(
                q_emb.tolist(),
                top_k=per_chunk_top_k,
                filters=filters,
                scale_score=False
            )
            results_per_chunk.append(docs)

        fused_docs = self._rrf_fuse(results_per_chunk, k=10)

        # DEDUP por resource_id ANTES de cortar a top_k
        out = []
        seen_resources = set()
        for doc in fused_docs:
            rid = (doc.meta or {}).get("resource_id")
            if rid is None:
                continue
            if rid in seen_resources:
                continue
            seen_resources.add(rid)
            out.append(doc)
            if len(out) >= top_k:
                break

        return out



# --- Core Logic Functions ---

def run_clustering_only_pipeline(attribute_path: str, max_k: int, embedding_models_instances: dict):

    print(f"Starting clustering pipeline (max_k={max_k})")
    # ... (checks, load attributes, call cluster_attributes, format results) ...
    attributes = load_json(attribute_path)
    print(attributes)
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
    clusters_dict["Winner Embedding Config"] = model_name
    clusters_dict["Embedding Config Info"] = models_info
    print("Clustering pipeline finished.")
    return clusters_dict

def compute_similarity_and_embeddings(cluster_text, resource_texts, embedding_models, model_name):
    print(f"Computing similarity with model: {model_name}")
    transformer_similarity = compute_transformer_embeddings(cluster_text, resource_texts, embedding_models, model_name)
    return transformer_similarity

def compute_only_similarity(cluster_embedding, resource_embeddings):
    similarity_scores = []
    for emb_res,embedding in resource_embeddings.items():
        similarity_scores.append(cosine_similarity(np.array(list(cluster_embedding.values())[0].reshape(1,-1)), np.array(list(embedding.values())[0].reshape(1,-1))))

    print("Similitudes entre cluster y recursos:")
    print(similarity_scores)
    return {i: score for i, score in enumerate(similarity_scores)}
def compute_transformer_embeddings(cluster_text, resource_texts, embedding_models, model_name):
    model_obj = embedding_models
    if isinstance(model_obj, SentenceTransformer):
        model = model_obj
        resource_embeddings = model.encode(resource_texts, show_progress_bar=True,normalize_embeddings=True)

        cluster_embedding = model.encode([cluster_text],normalize_embeddings=True)[0]
    else:
        model, tokenizer = model_obj
        print("transformers embeddings")
        resource_embeddings = get_transformers_embedding(model, tokenizer, resource_texts)

        cluster_emb = get_transformers_embedding(model, tokenizer, [cluster_text])
        cluster_embedding = cluster_emb[0]

    similarity_scores = cosine_similarity(np.array([cluster_embedding]), resource_embeddings)[0]
    print("Similitudes entre cluster y recursos:")
    print(similarity_scores)
    return {i: score for i, score in enumerate(similarity_scores)}

def get_transformers_embedding(model, tokenizer, texts):
    embeddings = []
    model.eval()
    import torch
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=768, normalize_embeddings=True)
            outputs = model(**encoded)
            # Asumimos un modelo tipo BERT: outputs.last_hidden_state existe
            last_hidden = outputs.last_hidden_state
            cls_vector = last_hidden[:,0,:].squeeze(0).numpy()
            embeddings.append(cls_vector)
    embeddings = np.array(embeddings)
    return embeddings

def load_embeddings_models(active_models):
    for model in active_models:
        embedding_instances = embedding_models_instances.get(model)    
    return embedding_instances
        # --- 3.2. Cargar instancias de modelos base una sola vez ---
        # model_gen  = embedding_models_instances.get("general")
        # model_gen2 = embedding_models_instances.get("general2")
        # model_med  = embedding_models_instances.get("medico")
        # model_med2  = embedding_models_instances.get("medico2")
        # model_gemma = embedding_models_instances.get("gemma")
        # model_biobert = embedding_models_instances.get("biobert")
        # modern_medico = embedding_models_instances.get("modern_medico")

cross = CrossEncoder("ncbi/MedCPT-Cross-Encoder")
def rerank(query: str, candidate_ids: list[str], resource_text_by_id: dict, cross, top_k: int = 30):
    # dedup conservando orden
    seen = set()
    cands = []
    print("Candidate IDs")
    print(candidate_ids)
    for rid in candidate_ids:
        if rid and rid not in seen:
            seen.add(rid)
            cands.append(rid)

    if not cands:
        return []

    pairs = []
    for rid in cands:
        txt = resource_text_by_id.get(rid)
        if not txt:
            # fallback: usa el nombre del recurso para no perder el candidato
            txt = rid
        pairs.append((query, txt))

    scores = cross.predict(pairs)
    ranked = [rid for rid, _ in sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)]
    print(f"Reranked top-{top_k} candidates.")
    print(ranked)
    
    return ranked[:top_k]

def build_attr_hits(attr_rankings: dict, top_l: int = 3):
    """
    Devuelve hits por atributo como sets de resource_ids.
    Soporta rankings en formato:
      - ["Patient", "Encounter", ...]
      - [("Patient", 0.53), ("Encounter", 0.47), ...]
      - [{"resource": "Patient", "score": 0.53}, ...]  (por si acaso)
    """
    hits = {}

    for attr, ranking in attr_rankings.items():
        if not ranking:
            hits[attr] = set()
            continue

        top = ranking[:top_l]
        ids = []

        for item in top:
            if isinstance(item, str):
                rid = item
            elif isinstance(item, tuple) or isinstance(item, list):
                # admite (rid, score) o (rid, score, ...) -> nos quedamos con el primero
                rid = item[0]
            elif isinstance(item, dict):
                rid = item.get("resource") or item.get("id")
            else:
                rid = str(item)

            if rid is not None:
                ids.append(rid)

        hits[attr] = set(ids)

    return hits


from collections import defaultdict

def select_topk_by_weighted_coverage(attr_hits: dict, fused_scores: dict, top_k: int = 5, gamma: float = 0.02):
    """
    Greedy set-cover con ganancia ponderada por rareza:
    - weight(attr) = 1 / |hits(attr)|  -> los atributos que "solo apuntan" a 1-2 recursos pesan mucho.
    - utility = score + gamma * weighted_gain
    """
    selected = []
    covered = set()
    attrs = list(attr_hits.keys())
    if not attrs:
        # fallback: top por score
        return [rid for rid, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]

    # pesos por atributo (rarity)
    w = {a: 1.0 / max(1, len(attr_hits[a])) for a in attrs}

    pool = set().union(*attr_hits.values())

    for _ in range(top_k):
        best_rid = None
        best_utility = -1e18

        for rid in pool:
            if rid in selected:
                continue

            weighted_gain = 0.0
            for a in attrs:
                if a in covered:
                    continue
                if rid in attr_hits[a]:
                    weighted_gain += w[a]

            score = fused_scores.get(rid, 0.0)
            utility = score + gamma * weighted_gain

            if utility > best_utility:
                best_utility = utility
                best_rid = rid

        if best_rid is None:
            break

        selected.append(best_rid)

        for a in attrs:
            if a not in covered and best_rid in attr_hits[a]:
                covered.add(a)

        if len(covered) == len(attrs):
            break

    # relleno por score
    if len(selected) < top_k:
        remaining = sorted(
            [(rid, sc) for rid, sc in fused_scores.items() if rid not in selected],
            key=lambda x: x[1],
            reverse=True
        )
        for rid, _ in remaining:
            selected.append(rid)
            if len(selected) >= top_k:
                break
    print("Selected", selected)
    return selected

def make_resource_cards(passages: dict, max_words: int = 180) -> dict:
    """
    Texto corto por recurso para el cross-encoder (evita truncado malo).
    """
    cards = {}
    for rid, txt in passages.items():
        txt = (txt or "").strip()
        snippet = " ".join(txt.split()[:max_words])
        cards[rid] = f"FHIR Resource: {rid}. {snippet}"
    return cards

def safe_rerank(query: str, candidate_ids: list[str], resource_cards: dict, cross, top_k: int = 25):
    # rerank solo top-N para reducir ruido + coste
    cands = [c for c in candidate_ids if isinstance(c, str)]
    cands = cands[:max(top_k, 10)]
    if not cands:
        return []

    pairs = [(query, resource_cards.get(rid, rid)) for rid in cands]
    scores = cross.predict(pairs)
    ranked = [rid for rid, _ in sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)]
    return ranked[:top_k]

def fuse_bi_and_cross(bi_ids: list[str], cross_ids: list[str], k: int = 60):
    # RRF a nivel recurso: devuelve [(rid, score)]
    return rrf_fuse_resource_rankings([to_rank_tuples(bi_ids), to_rank_tuples(cross_ids)], k=k)
def to_rank_tuples(ids):
    # Convierte ids ordenados -> lista de tuples para tu RRF actual (score monotónico)
    return [(rid, 1.0/(i+1)) for i, rid in enumerate(ids)]

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
    passages_preprocessed = {rid: preprocess_text_dense(txt) for rid, txt in passages.items()}
    if not passages: return {"error": "Failed to create passages."}
    #haystack_docs_processed = create_documents_from_passages_with_chunks(passages)

    # 3. Initialize Haystack Components
    print("Initializing Haystack...")
    try:
        document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine",
        )

        default_active_models = ["medico2"]

        winner_embedding_config = clusters_data.get("Winner Embedding Config")
        embedding_config_info = clusters_data.get("Embedding Config Info", {})
        print(f"Winner Embedding Config: {winner_embedding_config}, info: {embedding_config_info}")
        # Mapear nombre de modelo ganador -> lista de modelos activos para el CombinedRetriever
        CONFIG_TO_ACTIVE_MODELS = {
            "general_only":             ["general"],
            "general2_only":            ["general2"],
            "medico_only":              ["medico"],
            # "openai_only":            ["openai"],
            "biobert_only":             ["biobert"],
            "gemma_only":               ["gemma"],
            "medico2_only":            ["medico2"],
            "modern_medico_only":      ["modern_medico"],

            # "general+medico":           ["general", "medico"],
            # "general2+medico":          ["general2", "medico"],
            # "general+medico+gemma":     ["general", "medico", "gemma"],

            # "medico+biobert":           ["medico", "biobert"],
            # "medico+gemma":             ["medico", "gemma"],
            # "medico+biobert+gemma":     ["medico", "biobert", "gemma"],

            # "general+biobert":          ["general", "biobert"],
            # "general+biobert+gemma":    ["general", "biobert", "gemma"],
            # "general2+biobert":         ["general2", "biobert"],
            # "general2+biobert+gemma":   ["general2", "biobert", "gemma"],
            # "general+medico+biobert":   ["general", "medico", "biobert"],

            # "all_models_with_biobert":  ["general", 
            #                              "general2",
            #                              "medico", "biobert", "gemma"],
        }


        winner_model = True
        if isinstance(winner_embedding_config, str) and winner_model:
            active_models = CONFIG_TO_ACTIVE_MODELS.get(
                winner_embedding_config,
                default_active_models
            )
        else:
            # Si no viene la clave o no es str, usamos todos por defecto
            active_models = default_active_models

    
        print(f"Active models for RAG: {active_models}")


        # --- 3.3. Crear el CombinedRetriever una sola vez ---
        combined_retriever = CombinedRetriever(
            document_store=document_store,
            model_general=embedding_models_instances["general"],
            model_general2=embedding_models_instances["general2"],
            model_medico=embedding_models_instances["medico"],
            model_medico2=embedding_models_instances["medico2"],
            model_biobert=embedding_models_instances["biobert"],
            model_gemma=embedding_models_instances["gemma"],
            modern_medico=embedding_models_instances["modern_medico"],
            active_models=active_models,
        )
        resource_cards = make_resource_cards(passages, max_words=180)
        haystack_docs_processed = create_documents_from_passages_with_token_chunks(passages_preprocessed,combined_retriever,256,32)
        print(haystack_docs_processed)
        if not haystack_docs_processed: return {"error": "Failed to create document chunks."}

                   # --- 3.4. Embedder documentos y escribir en el DocumentStore ---
        doc_embeddings = combined_retriever.embed_documents(haystack_docs_processed)

        for doc, emb in zip(haystack_docs_processed, doc_embeddings):
            doc.embedding = emb

        combined_retriever.document_store.write_documents(haystack_docs_processed)
        print(f"Wrote {combined_retriever.document_store.count_documents()} documents to store.")

    except Exception as e:
        print(f"ERROR initializing Haystack: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed Haystack setup: {e}"}

    # 4. Perform Retrieval for Each Cluster
    print("Performing retrieval...")
    rag_results = {}
    SKIP_KEYS = {"Winner Embedding Config", "Embedding Config Info"}

    for cluster_key, cluster_content in clusters_data.items():
        if cluster_key in SKIP_KEYS:
            continue
        print(f"--- Processing {cluster_key} ---")
        if not isinstance(cluster_content, dict):
             print(f"Warn: Skip {cluster_key}, content not dict.")
             rag_results[cluster_key] = {"Attributes": "Invalid Content", "Top Resources": [], "Error": "Invalid cluster content"}
             continue

        # Construct query from the values (descriptions) in the cluster_content dict
        print("Constructing query from cluster attributes...")

        resource_text_by_id = passages
        query_parts = [f"Attribute: {name}. Description: {desc}." for name, desc in cluster_content.items()]
        cluster_query_text = " ".join(query_parts)
        print(cluster_query_text)
        if not cluster_query_text.strip():
             print(f"Warn: Skip {cluster_key}, empty query.")
             # Use the original dict for attributes even if query fails
             rag_results[cluster_key] = {"Attributes": cluster_content, "Top Resources": [], "Error": "Empty query"}
             continue

        try:  
            rankings = []
            attr_rankings = {}

            # ====== (A) Cluster-level bi + cross + fusion ======
            r_cluster = combined_retriever.retrieve_resources_exact(
                cluster_query_text,
                top_k_resources=45,
                agg="mean_top2",
                chunk_size=256,
                overlap=32,
            )

            #bi_cluster = [(x["resource"], x["score"]) for x in r_cluster]            
            bi_cluster = [x["resource"] for x in r_cluster]
            cross_cluster = safe_rerank(cluster_query_text, bi_cluster, resource_cards, cross, top_k=15)
            cluster_fused = fuse_bi_and_cross(bi_cluster, cross_cluster, k=15)

            rankings.append(cluster_fused)

            # ====== (B) Por atributo: bi + cross + fusion ======
            for attr_name, desc in cluster_content.items():
                q_embed = build_attr_query(attr_name, desc, v_type="")
                q_cross = f"Field: {attr_name}. Description: {desc}"

                r_attr = combined_retriever.retrieve_resources_exact(
                    q_embed,
                    top_k_resources=45,
                    agg="mean_top2",
                    chunk_size=256,
                    overlap=32,
                )

                print(r_attr)
                #bi_attr = [(x["resource"], x["score"]) for x in r_attr]
                bi_attr = [x["resource"] for x in r_attr]  
                cross_attr = safe_rerank(q_cross, bi_attr, resource_cards, cross, top_k=15)
                attr_fused = fuse_bi_and_cross(bi_attr, cross_attr, k=15)

                rankings.append(attr_fused)
                attr_rankings[attr_name] = attr_fused

            # ====== (C) Fusión global RRF (sobre listas ya fusionadas) ======
            fused = rrf_fuse_resource_rankings(rankings, k=15)
            fused_scores = {rid: sc for rid, sc in fused}

            # ====== (D) selección final ======
            attr_hits = build_attr_hits(attr_rankings, top_l=5)
            selected = select_topk_by_weighted_coverage(attr_hits, fused_scores, top_k=top_k, gamma=0.02)

            top_resources = [{"resource": rid} for rid in selected]
            print("Top Resources:", top_resources)

            rag_results[cluster_key] = {
                "Attributes": cluster_content, # Use the original dict here
                "Top Resources": top_resources, # Use the list formatted without score
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


def run_rag_pipeline_no_cluster(attributes_file_path: str, resource_path: str, json_schemas_path: str, top_k: int, threshold: float):
    """
    Loads clustered data, resources, schemas, performs RAG retrieval for each cluster,
    and returns the results matching the expected format.
    """
    print(f"Starting RAG pipeline: clusters='{attributes_file_path}', resources='{resource_path}', schemas='{json_schemas_path}', top_k={top_k}, threshold={threshold}")

    # ... (1. Load Input Data and Validation remains the same) ...
    print("Loading RAG input data...")
    attributes_data = load_json(attributes_file_path)
    resources = load_ndjson(resource_path)
    json_schemas = load_ndjson(json_schemas_path)
    if attributes_data is None: return {"error": f"Failed load attributes data: {attributes_file_path}."}
    if not isinstance(attributes_data, list): return {"error": "Attributes data not list."}
    if resources is None: return {"error": f"Failed load resource data: {resource_path}."}
    if not isinstance(resources, list): return {"error": "Resource data not list."}
    if json_schemas is None: return {"error": f"Failed load schema data: {json_schemas_path}."}
    if not isinstance(json_schemas, list): return {"error": "Schema data not list."}
    print(f"Loaded data.")




    # ... (2. Prepare Documents for Haystack remains the same) ...
    print("Creating passages and documents...")
    passages = create_passages(resources, json_schemas)
    passages_preprocessed = {rid: preprocess_text_dense(txt) for rid, txt in passages.items()}
    if not passages: return {"error": "Failed to create passages."}

    # 3. Initialize Haystack Components
    print("Initializing Haystack...")
    try:
        document_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine",
        )

        default_active_models = ["medico2"]
        

        # Mapear nombre de modelo ganador -> lista de modelos activos para el CombinedRetriever
        CONFIG_TO_ACTIVE_MODELS = {
            "general_only":             ["general"],
            "general2_only":            ["general2"],
            "medico_only":              ["medico"],
            # "openai_only":            ["openai"],
            "biobert_only":             ["biobert"],
            "gemma_only":               ["gemma"],
            "medico2_only":            ["medico2"],
            "modern_medico_only":      ["modern_medico"],
        }

        try:
            winner_embedding_config = attributes_data.get("Winner Embedding Config")
            active_models = winner_embedding_config
        except:
            winner_embedding_config = None
            active_models = default_active_models
            
        print(f"Winner Embedding Config: {active_models}")

        if isinstance(active_models, str):
            active_models = CONFIG_TO_ACTIVE_MODELS.get(
                active_models,
                default_active_models
            )
        else:
            active_models = default_active_models

        print(f"Active models for RAG: {active_models}")


        # --- 3.3. Crear el CombinedRetriever una sola vez ---
        combined_retriever = CombinedRetriever(
            document_store=document_store,
            model_general=embedding_models_instances["general"],
            model_general2=embedding_models_instances["general2"],
            model_medico=embedding_models_instances["medico"],
            model_medico2=embedding_models_instances["medico2"],
            model_biobert=embedding_models_instances["biobert"],
            model_gemma=embedding_models_instances["gemma"],
            modern_medico=embedding_models_instances["modern_medico"],
            active_models=active_models,
        )
        resource_cards = make_resource_cards(passages, max_words=180)
        haystack_docs_processed = create_documents_from_passages_with_token_chunks(passages_preprocessed,combined_retriever,256,32)
        print(haystack_docs_processed)
        if not haystack_docs_processed: return {"error": "Failed to create document chunks."}

                   # --- 3.4. Embedder documentos y escribir en el DocumentStore ---
        doc_embeddings = combined_retriever.embed_documents(haystack_docs_processed)

        for doc, emb in zip(haystack_docs_processed, doc_embeddings):
            doc.embedding = emb

        combined_retriever.document_store.write_documents(haystack_docs_processed)
        print(f"Wrote {combined_retriever.document_store.count_documents()} documents to store.")

    except Exception as e:
        print(f"ERROR initializing Haystack: {e}\n{traceback.format_exc()}")
        return {"error": f"Failed Haystack setup: {e}"}

    # 4. Perform Retrieval for Each Cluster
    print("Performing retrieval...")
    rag_results = {}
    SKIP_KEYS = {"Winner Embedding Config", "Embedding Config Info"}

    
    # for item in attributes_data:
    #     attr = item["Attribute name"]
    #     desc = item["Description"]
    #     values = item.get("Values", [])
   
    #     print(f"--- Processing {attr} ---")
    #     if not isinstance(attr, dict):
    #          print(f"Warn: Skip {attr}, content not dict.")
    #          rag_results[attr] = {"Attributes": "Invalid Content", "Top Resources": [], "Error": "Invalid cluster content"}
    #          continue
        
    #     # Construct query from the values (descriptions) in the cluster_content dict
    #     print("Constructing query from cluster attributes...")

    #     resource_text_by_id = passages
        
    #     query_text = f"Attribute: {attr}. Description: {desc}. Values: {values}."
    #     #print(cluster_query_text)
    #     #if not cluster_query_text.strip():
    #          #print(f"Warn: Skip {cluster_key}, empty query.")
    #          # Use the original dict for attributes even if query fails
    #          #rag_results[cluster_key] = {"Attributes": cluster_content, "Top Resources": [], "Error": "Empty query"}
    #          #continue

    try:  
        rankings = []
        attr_rankings = {}

            # # ====== (A) Cluster-level bi + cross + fusion ======
            # r_cluster = combined_retriever.retrieve_resources_exact(
            #     query_text,
            #     top_k_resources=45,
            #     agg="mean_top2",
            #     chunk_size=256,
            #     overlap=32,
            # )

            # #bi_cluster = [(x["resource"], x["score"]) for x in r_cluster]            
            # bi_cluster = [x["resource"] for x in r_cluster]
            # cross_cluster = safe_rerank(query_text, bi_cluster, resource_cards, cross, top_k=15)
            # cluster_fused = fuse_bi_and_cross(bi_cluster, cross_cluster, k=15)

            # rankings.append(cluster_fused)

            # ====== (B) Por atributo: bi + cross + fusion ======
        i = 0
        for attr_list in attributes_data:
            print(attr_list)
            q_embed = build_attr_query(attr_list.get("Attribute name"), attr_list.get("Description", ""), v_type="")
            q_cross = f"Field: {attr_list['Attribute name']}. Description: {attr_list['Description']}"

            r_attr = combined_retriever.retrieve_resources_exact(
                q_embed,
                top_k_resources=45,
                agg="mean_top2",
                chunk_size=256,
                overlap=32,
            )

            print(r_attr)
            #bi_attr = [(x["resource"], x["score"]) for x in r_attr]
            bi_attr = [x["resource"] for x in r_attr]  
            cross_attr = safe_rerank(q_cross, bi_attr, resource_cards, cross, top_k=15)
            attr_fused = fuse_bi_and_cross(bi_attr, cross_attr, k=15)

            rankings.append(attr_fused)
            attr_rankings[attr_list.get("Attribute name")] = attr_fused

            # ====== (C) Fusión global RRF (sobre listas ya fusionadas) ======
            fused = rrf_fuse_resource_rankings(rankings, k=15)
            fused_scores = {rid: sc for rid, sc in fused}

            # ====== (D) selección final ======
            attr_hits = build_attr_hits(attr_rankings, top_l=5)
            selected = select_topk_by_weighted_coverage(attr_hits, fused_scores, top_k=top_k, gamma=0.02)

            top_resources = [{"resource": rid} for rid in selected]
            print("Top Resources:", top_resources)

            rag_results[f"Cluster {i}"] = {
                "Attributes": attr_list.get("Attribute name"), # Use the original dict here
                "Description": attr_list.get("Description"),
                "Top Resources": top_resources, # Use the list formatted without score
            }
            i += 1
    except Exception as e:
            print(f"ERROR retrieving for {attr_list.get('Attribute name')}: {e}\n{traceback.format_exc()}")
            rag_results[f"Cluster {i}"] = {
                    "Attributes": attr_list.get("Attribute name"), # Still include attributes on error
                    "Top Resources": [],
                    "Error": f"Retrieval failed: {e}"
                }

    print("RAG pipeline finished.")
    return rag_results

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

def send_request_llama(payload, endpoint):
    """Sends request to Together Llama endpoint."""
    if not TOGETHER_API_KEY or not endpoint:
         raise ValueError("Together API Key or Endpoint not configured.")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=180) # Add timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json() # Use .json() method
    except requests.exceptions.RequestException as e:
         print(f"Error sending request to Together endpoint {endpoint}: {e}")
         return {"error": f"Request failed: {e}", "choices": [{"message": {"content": ""}}]}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from Together: {e}")
        print(f"Response content: {response.text}")
        return {"error": f"JSON decode failed: {e}", "choices": [{"message": {"content": response.text}}]} # Include raw text
# --- LLM Response Generation ---
# Updated to handle errors and structure differences
def generate_response_llm(query: str, context: list, json_schemas: list, model_name:str, iterations: int,  model_temperature: float = 1.0):
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
    if model_name is None or model_name.strip() == "":
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
                        """ + 
                        """   {
                            "mappings": [
                                {
                                "table_attribute_name": "...",
                                "fhir_attribute_name": ["Res.attr1", "Res.attr2", "Res.attr3"]
                                },
                                ...
                            ]
                        }
                        

                        Each key corresponds to the table attribute name, and its value is an array of exactly three strings. If fewer than three matches are found, use "No additional attribute found" to fill the empty slots.

                        ##################### EXAMPLE ####################  
                        For example, if the table attribute is "patient_birthDate":

                        {
                            "mappings": [
                                {
                                "table_attribute_name": "patient_birthDate",
                                "fhir_attribute_name": ["Patient.birthDate", "Encounter.birthDate", "Patient.anchorAge"]
                                },
                                ...
                            ]
                        }

                        ##################### ADDITIONAL INSTRUCTIONS ####################  
                        - Consider the attribute name, description, and sample values when determining the best FHIR attributes.
                        - Return only the final JSON object without additional commentary.
                        - Use the FHIR R5 specification as the reference (https://www.hl7.org/fhir/).
                        - Double-check your mappings before returning the final result.
                        - You must use the provided functions to format your response.
                        - Ensure the JSON is well-formed and adheres to the specified structure. 
                        """
                        
                    )
                },
                {
                    "role": "user",
                    "content": f"{query}" # Query contains the attribute details and explicit task/format instructions
                }
            ],
            "temperature": model_temperature, # Slightly higher temp might encourage finding alternatives
            "top_p": 0, # Allow full vocab exploration initially
            "response_format":{"type": "json_object"},
            }

        print("Sending initial request to Azure GPT...")
        response_data = send_request(payload, GPT4V_ENDPOINT_FHIR)
        print(response_data)
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
        time.sleep(30) # Reduced sleep time

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
                           """ + """{
                            "mappings": [
                                {
                                "table_attribute_name": "...",
                                "fhir_attribute_name": ["Res.attr1", "Res.attr2", "Res.attr3"]
                                },
                                ...
                            ]
                            }
                            """
                            

                        )
                    },
                ],
                "temperature": model_temperature,
                "top_p": 0,
                "response_format":{"type": "json_object"},
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

            time.sleep(30) # Reduced sleep time

        final_response_content = current_response_content


    elif model_name == "Llama":
        together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

        if not together_client:
            raise ValueError("Together AI client not initialized (API Key missing?).")

        # --- Prepare Together Tools ---
        tools = []
        for i, schema in enumerate(json_schemas):
             if isinstance(schema, dict) and 'properties' in schema:
                 schema_title = schema.get("title", f"Mapping_{i + 1}")
                 schema_title_formatted = schema_title.replace(" ", "_").replace("-", "_").replace(".","_") # Sanitize
                 tools.append({
                     "type": "function",
                     "function": {
                         "name": schema_title_formatted[:64],
                         "description": f"Generate FHIR mapping for attributes potentially related to {schema_title}",
                         "parameters": {"type": "object", "properties": schema['properties']}
                     }
                 })
             else:
                  print(f"Warning: Skipping schema index {i} for tool definition due to invalid format.")

        print(f"Prepared {len(tools)} tools for Llama call.")

        # --- Llama Call (Only one call, script's reflection for Llama wasn't well-defined) ---
        try:
            # Choose a capable Llama model available via Together API
            # Meta-Llama-3.1-405B-Instruct-Turbo might be overkill/expensive, consider 70B or 8B.
            # llama_model = "meta-llama/Llama-3-70b-chat-hf"
            llama_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo" # Or 405B if needed and available

            print(f"Sending request to Together Llama ({llama_model})...")
            response = together_client.chat.completions.create(
                model=llama_model,
                messages=[
                    {'role': 'system','content': ( # Keep the system prompt concise but complete
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
                        """ + 
                        """   {
                            "mappings": [
                                {
                                "table_attribute_name": "...",
                                "fhir_attribute_name": ["Res.attr1", "Res.attr2", "Res.attr3"]
                                },
                                ...
                            ]
                        }
                        

                        Each key corresponds to the table attribute name, and its value is an array of exactly three strings. If fewer than three matches are found, use "No additional attribute found" to fill the empty slots.

                        ##################### EXAMPLE ####################  
                        For example, if the table attribute is "patient_birthDate":

                        {
                            "mappings": [
                                {
                                "table_attribute_name": "patient_birthDate",
                                "fhir_attribute_name": ["Patient.birthDate", "Encounter.birthDate", "Patient.anchorAge"]
                                },
                                ...
                            ]
                        }

                        ##################### ADDITIONAL INSTRUCTIONS ####################  
                        - Consider the attribute name, description, and sample values when determining the best FHIR attributes.
                        - Return only the final JSON object without additional commentary.
                        - Use the FHIR R5 specification as the reference (https://www.hl7.org/fhir/).
                        - Double-check your mappings before returning the final result.
                        - You must use the provided functions to format your response.
                        - Ensure the JSON is well-formed and adheres to the specified structure. 
                        """                        )
                    },
                    {'role': 'user', 'content': query} # Query contains the attribute details and task
                ],
                temperature=model_temperature, # Adjusted temperature
                # Add tools only if they exist
                 **({"tools": tools, "tool_choice": "auto"} if tools else {})
            )
            print("Llama response received.")
            print(response)
            response_message = response.choices[0].message
            if response_message.tool_calls:
                # Preference tool call arguments if available
                try:
                     # Assuming only one tool call based on script logic
                     tool_arguments = response_message.tool_calls[0].function.arguments
                     # Attempt to parse/format as JSON string for consistency
                     current_response_content = json.dumps(json.loads(tool_arguments))
                except (IndexError, AttributeError, json.JSONDecodeError) as e:
                     print(f"Warning: Could not extract/parse Llama tool call arguments: {e}")
                     current_response_content = response_message.content or json.dumps({"error": "Failed to process tool call."}) # Fallback to content or error
            elif response_message.content:
                current_response_content = response_message.content
            else:
                 print("Error: Llama response has no content or tool calls.")
                 current_response_content = json.dumps({"error": "Empty response from Llama."})
            for i in range(iterations):
                print(f"Starting reflection iteration {i+1}/{iterations}...")
                response = together_client.chat.completions.create(
                model=llama_model,
                messages=[
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
                           """ + """{
                            "mappings": [
                                {
                                "table_attribute_name": "...",
                                "fhir_attribute_name": ["Res.attr1", "Res.attr2", "Res.attr3"]
                                },
                                ...
                            ]
                            }
                            """
                            

                        )
                    },
                ],
                temperature=model_temperature,
                top_p=0,
                response_format={"type": "json_object"},
            )
                response_message = response.choices[0].message
                new_response_content = ""
                if response_message.tool_calls:
                # Preference tool call arguments if available
                    try:
                        # Assuming only one tool call based on script logic
                        tool_arguments = response_message.tool_calls[0].function.arguments
                        # Attempt to parse/format as JSON string for consistency
                        new_response_content = json.dumps(json.loads(tool_arguments))
                    except (IndexError, AttributeError, json.JSONDecodeError) as e:
                        print(f"Warning: Could not extract/parse Llama tool call arguments: {e}")
                        new_response_content = response_message.content or json.dumps({"error": "Failed to process tool call."}) # Fallback to content or error
                elif response_message.content:
                    new_response_content = response_message.content


            # Basic check if response changed significantly (optional)
                if len(new_response_content) > 0 and new_response_content != current_response_content:
                    print(f"Reflection {i+1} yielded changes.")
                    current_response_content = new_response_content
                else:
                    print(f"Reflection {i+1} did not yield significant changes or failed extraction.")
                 # Optionally break early if no changes or errors occurred

                time.sleep(30) # Reduced sleep time

            final_response_content = current_response_content
        except Exception as e:
            print(f"Error during Together Llama API call: {e}")
            print(traceback.format_exc())
            final_response_content = json.dumps({"error": f"Llama API call failed: {e}"})

        # No reflection loop implemented for Llama here, matching simplified script logic for it.

    else:
        raise ValueError(f"Unsupported modelName: {model_name}. Choose 'GPT' or 'Llama'.")

    return final_response_content


# --- Core Logic Function for LLM Identification ---
def run_attribute_identification_pipeline(
    llm_provider: str,
    cluster_file_path: str,
    structured_attributes_path: str,
    resource_path: str,
    schema_path: str,
    temperature: float):
    """
    Runs the LLM-based attribute identification pipeline for all clusters.
    """
    print(f"Starting LLM Attribute Identification Pipeline: Provider={llm_provider}, ClusterFile={cluster_file_path}")

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
            "You are mapping table attributes to FHIR R5 attributes."
            """You MUST:
            1. Read ALL the attributes in the user query above (each starts with "Attribute Name:").
            2. Return ONE single JSON list.
            3. The list MUST contain EXACTLY ONE element for EACH "Attribute Name" defined above.
            4. For each element:
            - "table_attribute_name": must be EXACTLY the same as the Attribute Name string.
            - "fhir_attribute_name": an array with EXACTLY 3 strings (top 3 most specific FHIR R5 attributes).

            After constructing the JSON list, compare all "Attribute Name" values from the prompt
            with "table_attribute_name" values in your JSON. If ANY attribute is missing, REGENERATE
            the ENTIRE list so that none is missing.

            Output ONLY the JSON list. No explanation
            """
        )
        # print(f"Generated Query:\n{query[:500]}...") # Optional: Log query snippet
        print("Query constructed for LLM.")
        print(query)

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
            query = query,
            context = context_list,
            json_schemas = valid_filtered_schemas,  # Pass valid schemas for function/tool defs
            model_name = llm_provider,
            model_temperature= temperature,
            iterations=1
            )
            print(llm_response_str)
        except Exception as e:
            print(f"Error calling generate_response_llm for cluster {cluster_name}: {e}")
            print(traceback.format_exc())
            output_data.append({"Cluster": cluster_name, "Error": f"LLM call failed: {e}"})
            continue  # Skip to next cluster

        # --- Remover bloques Markdown y limpiar la respuesta ---
        # Se asume que la respuesta viene en un bloque marcado con ```json y termina con ```
        try:
            parsed = json.loads(llm_response_str)
        except json.JSONDecodeError:
    # Fallback por si algún gateway añade ruido (raro con json_schema)
            cleaned = llm_response_str.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
            parsed = json.loads(cleaned)

        output_data.append({
            "Cluster": cluster_name,
            # Se almacena la lista parseada o el diccionario de error
            "LLM_Mappings": parsed
        })

    print("\nLLM Attribute Identification Pipeline finished.")
    return output_data


def run_attribute_no_cluster_identification_pipeline(
    llm_provider: str,
    cluster_file_path: str,
    structured_attributes_path: str,
    resource_path: str,
    schema_path: str,
    temperature: float):
    """
    Runs the LLM-based attribute identification pipeline for all clusters.
    """
    print(f"Starting LLM Attribute Identification Pipeline: Provider={llm_provider}, ClusterFile={cluster_file_path}")

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
        attribute_name= cluster_info.get('Attributes') # Get the dict {name: desc}
        attribute_desc = cluster_info.get("Description")
        top_resources_list = cluster_info.get('Top Resources', [])

        print(f"Attributes: {attribute_name}")
        if not attribute_name or not isinstance(attribute_name, str):
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
        print(f"Attributes to map: {attribute_name}")

        # Find detailed info for attributes in this cluster from the structured data file
        attr_details_for_query = []
        if attribute_name:
                attr_details_for_query.append({"Attribute name" : attribute_name, "Description": attribute_desc, "Values":[]})
        else:
                 print(f"Warning: Details for attribute '{attribute_name}' not found in structured data. Using name only.")
                 # Add a basic entry if details are missing but name exists
                 attr_details_for_query.append({"Attribute name": attribute_name, "Description": "N/A", "Values": []})

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
            "You are mapping table attributes to FHIR R5 attributes."
            """You MUST:
            1. Read ALL the attributes in the user query above (each starts with "Attribute Name:").
            2. Return ONE single JSON list.
            3. The list MUST contain EXACTLY ONE element for EACH "Attribute Name" defined above.
            4. For each element:
            - "table_attribute_name": must be EXACTLY the same as the Attribute Name string.
            - "fhir_attribute_name": an array with EXACTLY 3 strings (top 3 most specific FHIR R5 attributes).

            After constructing the JSON list, compare all "Attribute Name" values from the prompt
            with "table_attribute_name" values in your JSON. If ANY attribute is missing, REGENERATE
            the ENTIRE list so that none is missing.

            Output ONLY the JSON list. No explanation
            """
        )
        # print(f"Generated Query:\n{query[:500]}...") # Optional: Log query snippet
        print("Query constructed for LLM.")
        print(query)

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
            query = query,
            context = context_list,
            json_schemas = valid_filtered_schemas,  # Pass valid schemas for function/tool defs
            model_name = llm_provider,
            model_temperature= temperature,
            iterations=1
            )
            print(llm_response_str)
        except Exception as e:
            print(f"Error calling generate_response_llm for cluster {cluster_name}: {e}")
            print(traceback.format_exc())
            output_data.append({"Cluster": cluster_name, "Error": f"LLM call failed: {e}"})
            continue  # Skip to next cluster

        # --- Remover bloques Markdown y limpiar la respuesta ---
        # Se asume que la respuesta viene en un bloque marcado con ```json y termina con ```
        try:
            parsed = json.loads(llm_response_str)
        except json.JSONDecodeError:
    # Fallback por si algún gateway añade ruido (raro con json_schema)
            cleaned = llm_response_str.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL)
            parsed = json.loads(cleaned)

        output_data.append({
            "Cluster": cluster_name,
            # Se almacena la lista parseada o el diccionario de error
            "LLM_Mappings": parsed
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
    max_k: int = Query(default=40, ge=3, le=50, description="Max number of clusters to explore (inclusive).")
):
    # (Keep existing /cluster-attributes endpoint code as provided previously)
    """
    Triggers the attribute clustering pipeline. Reads attributes, clusters them up to max_k,
    selects the best, and returns the results. Writes results to a predefined file.
    """
    print(OPENAI_API_KEY, nlp, embedding_models_instances) # Debugging line
    if not all([OPENAI_API_KEY, nlp, embedding_models_instances]): # Basic checks
        raise HTTPException(status_code=503, detail="Server prerequisites not met (OpenAI Key, SpaCy, Models).")
    try:
        
        attribute_path = "data/enriched_attribute_descriptions_SK.json" # Input for clustering
        # Define the output path where clustering results will be saved (and RAG will read from)
        cluster_output_path = "cluster_output/clusters_sk_demo.json"

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
        cluster_file_path = "cluster_output/clusters_sk_demo.json"
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
        rag_output_path = "output_rag/clusters_rag_sk_demo.json" # New name
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
    llm_provider: str = Query(default="GPT", description="LLM provider to use ('GPT' for Azure, 'Llama' for Together)."),
    num_exec: int = Query(default=1, description="Number of executions of the experiment."),
    temperature: float = Query(default=1.0, ge=0.0, le=2.0, description="Temperature setting for LLM generation."),
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
    print(AZURE_OPENAI_API_KEY, GPT4V_ENDPOINT_FHIR) # Debugging line
    if (not AZURE_OPENAI_API_KEY or not GPT4V_ENDPOINT_FHIR):
         raise HTTPException(status_code=503, detail="Azure OpenAI API Key or Endpoint not configured for GPT provider.")
    #if llm_provider == "Llama" and not together_client: # Check if client initialized
    #    raise HTTPException(status_code=503, detail="Together AI API Key not configured or client failed to initialize for Llama provider.")
    if not nlp:
         raise HTTPException(status_code=503, detail="Server configuration error: SpaCy model failed to load.")
    # Add checks for other necessary components if needed

    # --- Define File Paths ---
    base_data_path = "data/"
    base_output_rag_path = "output_rag/"
    base_output_path = "/output"
    llm_output_base = "llm_output/" # Separate output dir for LLM results
    cluster_file_name = "clusters_rag_sk_demo.json"
    cluster_file_path = os.path.join(base_output_rag_path, cluster_file_name)
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
    
    list_results = []
    for i in range(num_exec):
        print(f"LLM Identification Execution {i}/{num_exec} started.")
        try:
            results = run_attribute_identification_pipeline(
                llm_provider=llm_provider,
                cluster_file_path=cluster_file_path,
                structured_attributes_path=structured_attributes_path,
                resource_path=resource_path,
                schema_path=schema_path,
                temperature=temperature
            )

            if isinstance(results, dict) and "error" in results:
                # Pipeline itself reported an error (e.g., loading failed)
                raise HTTPException(status_code=500, detail=f"Pipeline error: {results['error']}")

            # --- Optional: Save results to file --
            output_filename = f"Slovak/{llm_provider}_llm_results_iter_{i}_temp_{temperature}_{cluster_file_name}"
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
            list_results.append(results) # Append the results to the list

        except HTTPException as http_exc:
            raise http_exc # Re-raise specific HTTP exceptions
        except ValueError as ve: # Catch config errors etc.
            print(f"Configuration or Value Error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            print(f"An unexpected error occurred in the LLM endpoint: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error during LLM identification: {str(e)}")

    return list_results if num_exec > 1 else list_results[0] # Return single result directly if only one execution

@app.get("/identify-attributes-llm-no-clusters", tags=["LLM Identification"])
async def identify_attributes_llm_endpoint_no_clusters(
    llm_provider: str = Query(default="GPT", description="LLM provider to use ('GPT' for Azure, 'Llama' for Together)."),
    num_exec: int = Query(default=1, description="Number of executions of the experiment."),
    temperature: float = Query(default=1.0, ge=0.0, le=2.0, description="Temperature setting for LLM generation."),
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
    print(AZURE_OPENAI_API_KEY, GPT4V_ENDPOINT_FHIR) # Debugging line
    if (not AZURE_OPENAI_API_KEY or not GPT4V_ENDPOINT_FHIR):
         raise HTTPException(status_code=503, detail="Azure OpenAI API Key or Endpoint not configured for GPT provider.")
    #if llm_provider == "Llama" and not together_client: # Check if client initialized
    #    raise HTTPException(status_code=503, detail="Together AI API Key not configured or client failed to initialize for Llama provider.")
    if not nlp:
         raise HTTPException(status_code=503, detail="Server configuration error: SpaCy model failed to load.")
    # Add checks for other necessary components if needed

    # --- Define File Paths ---
    base_data_path = "data/"
    base_output_rag_path = "output_rag/"
    base_output_path = "/output"
    llm_output_base = "llm_output/" # Separate output dir for LLM results
    attribute_file_name = "no_clusters_rag_mimic_prueba_config_desc.json"
    attribute_file_path = os.path.join(base_output_rag_path, attribute_file_name)
    # Path to the structured attributes file used by the script
    structured_attributes_path = "data/filtered_data_attributes.json" # Based on script variable 'path' + 'filename'
    resource_path = os.path.join(base_data_path, "datasetRecursosBase.ndjson") # Based on script
    schema_path = os.path.join(base_data_path, "enriched_dataset_schemasBase.ndjson") # Based on script


    # --- File Existence Checks ---
    required_files = {
        "Attribute File": attribute_file_path,
        "Structured Attributes": structured_attributes_path,
        "Resources": resource_path,
        "Schemas": schema_path
    }
    for name, path in required_files.items():
         if not os.path.exists(path):
              raise HTTPException(status_code=404, detail=f"{name} file not found at expected path: {path}")

    # --- Run Pipeline ---
    
    list_results = []
    for i in range(num_exec):
        print(f"LLM Identification Execution {i}/{num_exec} started.")
        try:
            results = run_attribute_no_cluster_identification_pipeline(
                llm_provider=llm_provider,
                cluster_file_path=attribute_file_path,
                structured_attributes_path=structured_attributes_path,
                resource_path=resource_path,
                schema_path=schema_path,
                temperature=temperature
            )

            if isinstance(results, dict) and "error" in results:
                # Pipeline itself reported an error (e.g., loading failed)
                raise HTTPException(status_code=500, detail=f"Pipeline error: {results['error']}")

            # --- Optional: Save results to file --
            output_filename = f"{llm_provider}/{llm_provider}_llm_results_iter_{i+6}_temp_{temperature}_{attribute_file_name}"
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
            list_results.append(results) # Append the results to the list

        except HTTPException as http_exc:
            raise http_exc # Re-raise specific HTTP exceptions
        except ValueError as ve: # Catch config errors etc.
            print(f"Configuration or Value Error: {ve}")
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            print(f"An unexpected error occurred in the LLM endpoint: {e}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Internal server error during LLM identification: {str(e)}")

    return list_results if num_exec > 1 else list_results[0] # Return single result directly if only one execution

@app.get("/retrieve-resources-no-cluster", tags=["Retrieval (RAG)"])
async def retrieve_resources_no_cluster_endpoint(
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top documents to retrieve per cluster."),
    threshold: float = Query(default=0.70, ge=0.0, le=1.0, description="Minimum similarity score threshold for retrieved documents.")
):
    # (Keep the endpoint logic, ensuring it calls the corrected run_rag_pipeline_no_cluster)
    """
    Triggers the RAG pipeline using pre-computed clusters. Reads cluster data, resources, schemas,
    indexes them using Haystack with a combined embedding model (using text-embedding-3-large),
    and retrieves the top_k relevant resources for each cluster, filtered by score threshold.
    Output matches the specified format (attribute dictionary, resources without scores).
    """
    if not all([OPENAI_API_KEY, nlp, embedding_models_instances]):
        raise HTTPException(status_code=503, detail="Server prerequisites not met.")
    try:
        attr_file_path = "data/filtered_data_attributes.json"
        resource_path = "data/datasetRecursosBase.ndjson"
        json_schemas_path = "data/enriched_dataset_schemasBase.ndjson"
        print(f"RAG API triggered: K={top_k}, Threshold={threshold}")

        # --- File existence checks ---
        if not os.path.exists(attr_file_path): raise HTTPException(status_code=404, detail=f"Attribute file missing: {attr_file_path}. Run /cluster-attributes first.")
        if not os.path.exists(resource_path): raise HTTPException(status_code=500, detail=f"Resource file missing: {resource_path}")
        if not os.path.exists(json_schemas_path): raise HTTPException(status_code=500, detail=f"Schema file missing: {json_schemas_path}")

        results = run_rag_pipeline_no_cluster(attr_file_path, resource_path, json_schemas_path, top_k, threshold)

        if isinstance(results, dict) and "error" in results: raise HTTPException(status_code=500, detail=results["error"])

        # Optional: Save RAG results to file
        rag_output_path = "output_rag/no_clusters_rag_mimic_prueba_config_desc.json" # New name
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









# --- Run the API (for local development) ---





if __name__ == "__main__":
    print("Starting Uvicorn server for local development...")
    uvicorn.run(
        "main_api_clustering:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_config=LOGGING_CONFIG,  
    )