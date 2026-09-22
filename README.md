# INTEROPERABILITYFHIRAPI

API REST construida con **FastAPI** (Python) que facilita la interoperabilidad con FHIR mediante técnicas de clustering semántico, recuperación de información aumentada (RAG) e identificación de atributos mediante LLMs.

---

## Tabla de contenidos

1. [Quickstart](#quickstart)
2. [Requisitos previos](#requisitos-previos)
3. [Instalación](#instalación)
4. [Configuración de variables de entorno](#configuración-de-variables-de-entorno)
5. [Iniciar el servidor](#iniciar-el-servidor)
6. [Endpoints](#endpoints)
   - [GET /cluster-attributes](#get-cluster-attributes)
   - [GET /retrieve-resources](#get-retrieve-resources)
   - [GET /retrieve-resources-no-cluster](#get-retrieve-resources-no-cluster)
   - [GET /identify-attributes-llm](#get-identify-attributes-llm)
   - [GET /identify-attributes-llm-no-clusters](#get-identify-attributes-llm-no-clusters)
7. [Orden de ejecución recomendado](#orden-de-ejecución-recomendado)
8. [Makefile (opcional)](#makefile-opcional)
9. [One-liner de configuración (opcional)](#one-liner-de-configuración-opcional)
10. [Contribuir](#contribuir)

---

## Quickstart

```bash
# 1. Clonar el repositorio
git clone https://github.com/alvumu/InteroperabilityFHIRAPI.git
cd InteroperabilityFHIRAPI

# 2. Crear y activar entorno virtual
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4. Descargar recursos de NLP
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt

# 5. Configurar variables de entorno
cp .env.example .env
# Edita .env con tus claves de API (ver sección siguiente)

# 6. Iniciar el servidor
uvicorn api.main_api_clustering:app --reload --port 8000

# 7. Abrir la documentación interactiva
# http://localhost:8000/docs
```

---

## Requisitos previos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Python | 3.8+ | Incluye el módulo `venv` |
| Git | cualquiera | Para clonar el repositorio |
| Make | cualquiera | Opcional, solo si usas el Makefile |

Las dependencias Python se detallan en [`requirements.txt`](requirements.txt). Las principales son:

- **FastAPI** + **Uvicorn** — framework y servidor ASGI
- **scikit-learn** — algoritmos de clustering (KMeans, DBSCAN, Birch, etc.)
- **sentence-transformers** — generación de embeddings semánticos
- **openai** — cliente para la API de OpenAI / Azure OpenAI
- **haystack-ai** — pipeline RAG con `InMemoryDocumentStore`
- **spacy** + **nltk** — preprocesamiento de texto

---

## Instalación

```bash
# Clonar
git clone https://github.com/alvumu/InteroperabilityFHIRAPI.git
cd InteroperabilityFHIRAPI

# Entorno virtual
python3 -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Dependencias Python
pip install --upgrade pip
pip install -r requirements.txt

# Recursos de NLP
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords punkt
```

---

## Configuración de variables de entorno

El servidor lee su configuración exclusivamente mediante variables de entorno. Se incluye el archivo [`.env.example`](.env.example) como plantilla:

```bash
cp .env.example .env
# Edita .env con un editor de texto
```

| Variable | Obligatoria | Descripción |
|---|---|---|
| `OPENAI_API_KEY` | Sí (para clustering y RAG) | Clave de API de OpenAI, usada para generar embeddings con `text-embedding-3-large`. |
| `AZURE_OPENAI_API` | Sí (para endpoints LLM con GPT) | Clave de API de Azure OpenAI. |
| `GPT4V_ENDPOINT_FHIR` | Sí (para endpoints LLM con GPT) | URL completa del endpoint de Azure OpenAI (incluyendo el nombre del deployment y la versión de API). |
| `TOGETHER_API_KEY` | Sí (si se usa `llm_provider=Llama`) | Clave de API de Together AI para el modelo Llama. |
| `LOG_FILE` | No | Ruta del archivo de log. Por defecto: `log.out`. |

> **Nota de seguridad:** El archivo `.env` está incluido en `.gitignore` y nunca debe commitearse al repositorio.

---

## Iniciar el servidor

```bash
# Modo desarrollo (con recarga automática al cambiar el código)
uvicorn api.main_api_clustering:app --reload --port 8000

# Modo producción (sin recarga, múltiples workers)
uvicorn api.main_api_clustering:app --host 0.0.0.0 --port 8000 --workers 2
```

Una vez arrancado, la documentación interactiva (Swagger UI) estará disponible en:

```
http://localhost:8000/docs
```

Y la versión ReDoc en:

```
http://localhost:8000/redoc
```

---

## Endpoints

### GET /cluster-attributes

**Tag:** `Clustering`

**Descripción:**  
Ejecuta el pipeline completo de clustering sobre los atributos FHIR enriquecidos. Lee el archivo `data/enriched_attribute_descriptions_SK.json`, genera embeddings semánticos para cada atributo, y aplica varios algoritmos de clustering (KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS, SpectralClustering) evaluando diferentes valores de *k* hasta `max_k`. Selecciona automáticamente el mejor clustering según métricas de calidad (silhouette, Calinski-Harabász, Davies-Bouldin) y guarda los resultados en `cluster_output/clusters_sk_demo.json`. **Debe ejecutarse antes de `/retrieve-resources`**.

**Método HTTP:** `GET`  
**Ruta:** `/cluster-attributes`

**Parámetros de query:**

| Parámetro | Tipo | Por defecto | Rango | Descripción |
|---|---|---|---|---|
| `max_k` | `int` | `40` | 3–50 | Número máximo de clusters a explorar (inclusive). |

**Headers requeridos:**  
Ninguno en la request HTTP. El servidor usa internamente `OPENAI_API_KEY` para los embeddings.

**Respuesta exitosa (200):**  
JSON con los clusters generados. Cada clave es el nombre del cluster y el valor es la lista de atributos agrupados.

```json
{
  "cluster_0": ["Atributo A", "Atributo B"],
  "cluster_1": ["Atributo C"]
}
```

**Códigos de estado:**

| Código | Descripción |
|---|---|
| `200` | Clustering completado correctamente. |
| `500` | Error interno (archivo de atributos no encontrado, fallo al guardar, etc.). |
| `503` | Prerrequisitos no cumplidos (clave OpenAI no configurada, modelo SpaCy no cargado). |

**Ejemplo con curl:**

```bash
# Explorar con el máximo por defecto (40 clusters)
curl -X GET "http://localhost:8000/cluster-attributes"

# Limitar la búsqueda a un máximo de 15 clusters
curl -X GET "http://localhost:8000/cluster-attributes?max_k=15"
```

---

### GET /retrieve-resources

**Tag:** `Retrieval (RAG)`

**Descripción:**  
Ejecuta el pipeline RAG (Retrieval-Augmented Generation) usando los clusters pre-computados por `/cluster-attributes`. Carga los datos de clusters desde `cluster_output/clusters_sk_demo.json`, los recursos FHIR desde `data/datasetRecursosBase.ndjson` y sus esquemas enriquecidos desde `data/enriched_dataset_schemasBase.ndjson`. Indexa todos los documentos en un `InMemoryDocumentStore` de Haystack usando embeddings de `text-embedding-3-large` y recupera los `top_k` recursos más relevantes por cluster, filtrando por el umbral de similitud `threshold`. Los resultados se guardan en `output_rag/clusters_rag_sk_demo.json`. **Requiere haber ejecutado `/cluster-attributes` previamente.**

**Método HTTP:** `GET`  
**Ruta:** `/retrieve-resources`

**Parámetros de query:**

| Parámetro | Tipo | Por defecto | Rango | Descripción |
|---|---|---|---|---|
| `top_k` | `int` | `5` | 1–50 | Número máximo de documentos a recuperar por cluster. |
| `threshold` | `float` | `0.70` | 0.0–1.0 | Umbral mínimo de similitud coseno para incluir un documento en los resultados. |

**Respuesta exitosa (200):**  
JSON con los recursos recuperados agrupados por cluster.

```json
{
  "cluster_0": {
    "attributes": {"Atributo A": "descripción", "Atributo B": "descripción"},
    "resources": ["Patient", "Observation"]
  }
}
```

**Códigos de estado:**

| Código | Descripción |
|---|---|
| `200` | Recuperación completada correctamente. |
| `404` | Archivo de clusters no encontrado (ejecuta `/cluster-attributes` antes). |
| `500` | Error interno (archivo de recursos o esquemas no encontrado, fallo en el pipeline). |
| `503` | Prerrequisitos no cumplidos. |

**Ejemplo con curl:**

```bash
# Con valores por defecto
curl -X GET "http://localhost:8000/retrieve-resources"

# Recuperar los 10 recursos más relevantes con un umbral más estricto
curl -X GET "http://localhost:8000/retrieve-resources?top_k=10&threshold=0.80"
```

---

### GET /retrieve-resources-no-cluster

**Tag:** `Retrieval (RAG)`

**Descripción:**  
Versión del pipeline RAG sin agrupamiento previo por clusters. En lugar de leer los clusters desde un archivo de salida del pipeline de clustering, trabaja directamente con el archivo de atributos filtrados `data/filtered_data_attributes.json`. Indexa los recursos y esquemas FHIR en Haystack y recupera los `top_k` más relevantes para cada atributo según el umbral de similitud. Los resultados se guardan en `output_rag/no_clusters_rag_mimic_prueba_config_desc.json`.

**Método HTTP:** `GET`  
**Ruta:** `/retrieve-resources-no-cluster`

**Parámetros de query:**

| Parámetro | Tipo | Por defecto | Rango | Descripción |
|---|---|---|---|---|
| `top_k` | `int` | `5` | 1–50 | Número máximo de documentos a recuperar por atributo. |
| `threshold` | `float` | `0.70` | 0.0–1.0 | Umbral mínimo de similitud coseno. |

**Respuesta exitosa (200):**  
JSON con recursos recuperados por atributo (mismo formato que `/retrieve-resources`).

**Códigos de estado:**

| Código | Descripción |
|---|---|
| `200` | Recuperación completada correctamente. |
| `404` | Archivo de atributos no encontrado (`data/filtered_data_attributes.json`). |
| `500` | Error interno. |
| `503` | Prerrequisitos no cumplidos. |

**Ejemplo con curl:**

```bash
curl -X GET "http://localhost:8000/retrieve-resources-no-cluster"

curl -X GET "http://localhost:8000/retrieve-resources-no-cluster?top_k=8&threshold=0.75"
```

---

### GET /identify-attributes-llm

**Tag:** `LLM Identification`

**Descripción:**  
Ejecuta el pipeline de identificación de atributos FHIR mediante LLMs usando los resultados del pipeline RAG con clusters. Lee el archivo `output_rag/clusters_rag_sk_demo.json` y, para cada cluster, construye un prompt que incluye los atributos del cluster y el contexto de los recursos/esquemas recuperados. Llama al proveedor LLM elegido (Azure GPT o Together Llama) para que genere los mapeos FHIR correspondientes. El resultado de cada ejecución se guarda en `llm_output/Slovak/`. **Requiere haber ejecutado `/retrieve-resources` previamente.**

**Método HTTP:** `GET`  
**Ruta:** `/identify-attributes-llm`

**Parámetros de query:**

| Parámetro | Tipo | Por defecto | Valores posibles | Descripción |
|---|---|---|---|---|
| `llm_provider` | `str` | `"GPT"` | `"GPT"`, `"Llama"` | Proveedor LLM a usar. `"GPT"` usa Azure OpenAI; `"Llama"` usa Together AI. |
| `num_exec` | `int` | `1` | ≥ 1 | Número de veces que se repite el experimento completo. |
| `temperature` | `float` | `1.0` | 0.0–2.0 | Parámetro de temperatura para la generación del LLM. Valores más altos producen respuestas más variadas. |

**Headers requeridos (configurados vía variables de entorno):**  
- `AZURE_OPENAI_API` + `GPT4V_ENDPOINT_FHIR` si `llm_provider=GPT`  
- `TOGETHER_API_KEY` si `llm_provider=Llama`

**Respuesta exitosa (200):**  
- Si `num_exec=1`: lista de mapeos FHIR generados para cada cluster.
- Si `num_exec>1`: lista de listas (una por ejecución).

```json
[
  {
    "Cluster": "cluster_0",
    "LLM_Mappings": [
      {"fhir_resource": "Patient", "fhir_attribute": "Patient.name", "justification": "..."}
    ]
  }
]
```

**Códigos de estado:**

| Código | Descripción |
|---|---|
| `200` | Identificación completada correctamente. |
| `400` | Error de configuración o valor inválido. |
| `404` | Alguno de los archivos requeridos no existe. |
| `500` | Error interno durante el pipeline LLM. |
| `503` | Claves de API de Azure OpenAI no configuradas, o modelo SpaCy no cargado. |

**Ejemplo con curl:**

```bash
# Usando Azure GPT con temperatura por defecto
curl -X GET "http://localhost:8000/identify-attributes-llm"

# Usando Llama con temperatura 0.5 y 3 ejecuciones
curl -X GET "http://localhost:8000/identify-attributes-llm?llm_provider=Llama&temperature=0.5&num_exec=3"
```

---

### GET /identify-attributes-llm-no-clusters

**Tag:** `LLM Identification`

**Descripción:**  
Versión del pipeline LLM de identificación de atributos que trabaja sin agrupamiento previo en clusters. En lugar de leer los resultados de `/retrieve-resources`, utiliza el archivo `output_rag/no_clusters_rag_mimic_prueba_config_desc.json` generado por `/retrieve-resources-no-cluster`. Para cada atributo construye un prompt con contexto de recursos/esquemas relevantes y llama al LLM elegido para obtener los mapeos FHIR. Los resultados se guardan en `llm_output/<llm_provider>/`. **Requiere haber ejecutado `/retrieve-resources-no-cluster` previamente.**

**Método HTTP:** `GET`  
**Ruta:** `/identify-attributes-llm-no-clusters`

**Parámetros de query:**

| Parámetro | Tipo | Por defecto | Valores posibles | Descripción |
|---|---|---|---|---|
| `llm_provider` | `str` | `"GPT"` | `"GPT"`, `"Llama"` | Proveedor LLM a usar. |
| `num_exec` | `int` | `1` | ≥ 1 | Número de ejecuciones del experimento. |
| `temperature` | `float` | `1.0` | 0.0–2.0 | Temperatura para la generación del LLM. |

**Respuesta exitosa (200):**  
Mismo formato que `/identify-attributes-llm`.

**Códigos de estado:**

| Código | Descripción |
|---|---|
| `200` | Identificación completada correctamente. |
| `400` | Error de configuración o valor inválido. |
| `404` | Alguno de los archivos requeridos no existe. |
| `500` | Error interno durante el pipeline LLM. |
| `503` | Claves de API no configuradas o modelo SpaCy no cargado. |

**Ejemplo con curl:**

```bash
# Usando Azure GPT con valores por defecto
curl -X GET "http://localhost:8000/identify-attributes-llm-no-clusters"

# Usando Llama con temperatura 0.7
curl -X GET "http://localhost:8000/identify-attributes-llm-no-clusters?llm_provider=Llama&temperature=0.7"
```

---

## Orden de ejecución recomendado

Los pipelines se encadenan y deben ejecutarse en el siguiente orden:

### Flujo con clusters (recomendado)

```
1. GET /cluster-attributes
        ↓
2. GET /retrieve-resources
        ↓
3. GET /identify-attributes-llm
```

### Flujo sin clusters

```
1. GET /retrieve-resources-no-cluster
        ↓
2. GET /identify-attributes-llm-no-clusters
```

---

## Makefile (opcional)

```bash
# Inicializar el entorno completo
make init

# Limpiar el entorno virtual y archivos generados
make clean
```

---

## One-liner de configuración (opcional)

```bash
bash <(curl -sL https://raw.githubusercontent.com/alvumu/InteroperabilityFHIRAPI/main/setup.sh)
```

---

## Contribuir

1. Haz **fork** del repositorio.
2. Crea una rama para tu cambio:
   ```bash
   git checkout -b feature/tu-característica
   ```
3. Haz commit, push y abre un Pull Request.



