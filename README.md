# INTEROPERABILITYFHIRAPI

API REST con FastAPI para facilitar la interoperabilidad con FHIR y aplicar técnicas de clustering a datos sanitarios.
## REQUISITOS PREVIOS

- **Python 3.8+**: incluye el módulo `venv` para crear entornos virtuales aislados.  
- **Git** para clonar el repositorio.  
- (Opcional) **Make** si vas a usar el Makefile.  
- Modelos SpaCy y datos NLTK que se descargarán tras la instalación.

## INSTALACIÓN

1. Clona el repositorio:
   ```bash
   git clone https://github.com/alvumu/InteroperabilityFHIRAPI.git
   cd InteroperabilityFHIRAPI
   ```
3. Crea y activa el entorno virtual:
   ```bash
   python3 -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
5. Actualiza pip e instala dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Descarga recursos de NLP:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader stopwords punkt
   ```
## EJECUCIÓN

Para levantar el servidor en modo desarrollo:
   ```bash
   uvicorn api.main_api_clustering:app --reload
   ```
Luego abre en el navegador:
```bash
   http://localhost:8000/docs
```
## USO BÁSICO

- Envía payloads FHIR en JSON a los endpoints expuestos.  
- Usa `/cluster` para agrupar recursos según semántica NLP.  
- Revisa los docstrings en `api/main_api_clustering.py` para más opciones.

## MAKEFILE (OPCIONAL)

- Inicializar todo:
  ```bash
    make init
   ```
- Limpiar el entorno:
   ```bash
     make clean
   ```
## ONE-LINER CON CURL (OPCIONAL)
```bash
bash <(curl -sL https://raw.githubusercontent.com/alvumu/InteroperabilityFHIRAPI/main/setup.sh)
```
## CONTRIBUIR

1. Haz **fork** del repositorio.  
2. Crea una rama para tu cambio:
   git checkout -b feature/tu-característica
3. Haz commit, push y abre un Pull Request.



