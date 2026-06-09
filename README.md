# Sistema RAG NoSQL — Gestión de Operaciones Aeroportuarias

**Proyecto Final — Bases de Datos No Relacionales**  
Isabela Guerrero · Luisa Ibarra  
Universidad de Caldas — Departamento de Inteligencia Artificial e Ingeniería

---

## Descripción

Sistema de Recuperación y Generación Aumentada (RAG) construido sobre MongoDB Atlas para consultar de forma inteligente reportes operativos aeroportuarios. El sistema permite hacer preguntas en lenguaje natural, recuperar contexto relevante mediante búsqueda vectorial y generar respuestas contextualizadas con un LLM (Groq + LLaMA 3.1).

### Capacidades principales

- Búsqueda semántica sobre 100 reportes aeroportuarios (incidentes, seguridad, mantenimiento, climáticos, operativos)
- Búsqueda híbrida: filtros de metadatos + similitud vectorial
- Búsqueda multimodal texto ↔ imagen con CLIP
- Comparación de 3 estrategias de chunking (fixed-size, sentence-aware, semántico)
- Pipeline RAG completo con historial de consultas en MongoDB
- API REST documentada con FastAPI

---

## Arquitectura

```
data/
  reportes.json       ← 100 reportes operativos (ES + EN)
  imagenes.json       ← 50 metadatos de imágenes sintéticas

src/
  config.py           ← Conexión MongoDB Atlas, variables de entorno
  Init_db.py          ← Inicialización de colecciones, validadores e índices

  chunking/
    strategies.py     ← Fixed-size, Sentence-aware, Semantic chunking

  ingesta/
    cargar_datos.py   ← Ingesta de reportes e imágenes con embeddings
    embeddings.py     ← MiniLM (384d texto) + CLIP (512d imagen)

  rag/
    retriever.py      ← Vector search, hybrid search, compare_strategies
    pipeline.py       ← Build context → prompt → Groq LLM → historial

  api/
    main.py           ← FastAPI: /health, /search, /rag
    models.py         ← Re-exporta funciones de embedding
```

---

## Instalación

### 1. Requisitos previos

- Python 3.10 o superior
- Cuenta en [MongoDB Atlas](https://www.mongodb.com/atlas) con cluster M0 (gratuito)
- API key de [Groq](https://console.groq.com) (gratuita)

### 2. Clonar e instalar dependencias

```bash
git clone <url-del-repositorio>
cd Proyecto_aereolinea
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Editar el archivo `.env` en la raíz del proyecto:

```env
MONGO_URI=mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/
DB_NAME=airport_db
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_INDEX_TEXT=vector_index_texto
VECTOR_INDEX_IMAGE=vector_index_imagen
```

> **Importante:** nunca subir el archivo `.env` a un repositorio público.

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual (Windows)
venv\Scripts\activate

# Activar entorno virtual (Mac/Linux)
# source venv/bin/activate

# Instalar todas las dependencias necesarias
pip install fastapi uvicorn pymongo python-dotenv sentence-transformers pydantic groq httpx pillow nltk scikit-learn

# Instalar modelo NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Verificar instalación
pip list

### 4. Inicializar la base de datos

```bash
cd src
python Init_db.py
```

Este script crea las colecciones, aplica validadores de esquema y genera todos los índices compuestos y de texto.

### 5. Crear el índice Atlas Vector Search (manual en Atlas UI)

En MongoDB Atlas → tu cluster → Search → Create Index:

**Índice de texto (MiniLM 384d):**
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 384,
    "similarity": "cosine"
  }]
}
```
Nombre del índice: `vector_index_texto` · Colección: `embeddings_texto`

**Índice de imagen (CLIP 512d):**
```json
{
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 512,
    "similarity": "cosine"
  }]
}
```
Nombre del índice: `vector_index_imagen` · Colección: `embeddings_imagen`

### 6. Ingestar los datos

```bash
# Desde la carpeta src/
python -m ingesta.cargar_datos --reportes ../data/reportes.json
python -m ingesta.cargar_datos --imagenes ../data/imagenes.json
```

Parámetros opcionales:
- `--strategies fixed sentence semantic` — qué estrategias aplicar (default: todas)
- `--clear` — borrar chunks previos antes de reinsertar

### 7. Levantar la API

```bash
cd src
uvicorn api.main:app --reload --port 8000
```

Documentación interactiva: http://localhost:8000/docs

---

## Uso de la API

### GET /health
Verifica la conectividad con MongoDB.

```bash
curl http://localhost:8000/health
```

### POST /search — Búsqueda híbrida

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "incidentes en pista con vehículos", "limit": 3}'
```

Parámetros opcionales: `strategy`, `tipo`, `idioma`, `prioridad`, `compare`.

```bash
# Comparar las 3 estrategias de chunking
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "condiciones climáticas adversas", "compare": true}'
```

### POST /rag — Pipeline RAG completo

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué protocolos se activaron en incidentes de seguridad?",
    "tipo": "seguridad",
    "limit": 5,
    "usuario": "agente_torre_01"
  }'
```

---

## Estrategias de Chunking

| Estrategia | Parámetros | Caso de uso ideal |
|---|---|---|
| `fixed` | chunk_size=256 chars, overlap=32 | Logs de vuelos, datos estructurados homogéneos |
| `sentence` | max_sentences=5, overlap=1 oración | Reportes narrativos de incidentes |
| `semantic` | threshold=0.80 similitud coseno | Documentos técnicos con cambios de tema |

Cada chunk se almacena con el campo `estrategia_chunking` para permitir comparaciones dentro de la misma base de datos.

---

## Colecciones MongoDB

| Colección | Documentos | Descripción |
|---|---|---|
| `reportes` | 100 | Reportes operativos aeroportuarios |
| `embeddings_texto` | ~900 | Chunks vectorizados (3 estrategias × ~100 reportes × ~3 chunks) |
| `embeddings_imagen` | 50 | Metadatos + embeddings CLIP de imágenes |
| `historial_consultas` | variable | Consultas RAG por usuario |
| `vuelos` | — | Vuelos con metadatos operativos |
| `pasajeros` | — | Datos de pasajeros con reservas embebidas |
| `aerolineas` | — | Catálogo de aerolíneas |
| `equipajes` | — | Trazabilidad de equipaje |
| `empleados` | — | Personal aeroportuario |
| `torre_control` | — | Observaciones por turno |

---

## Tecnologías

- **Base de datos:** MongoDB Atlas M0 (Vector Search nativo)
- **Embeddings texto:** `all-MiniLM-L6-v2` — 384 dimensiones
- **Embeddings imagen:** `openai/clip-vit-base-patch32` — 512 dimensiones
- **LLM:** Groq API con `llama-3.1-8b-instant`
- **API:** FastAPI + Uvicorn
- **Procesamiento de texto:** NLTK, scikit-learn
