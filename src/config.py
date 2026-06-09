"""
config.py — Conexión centralizada a MongoDB Atlas.
Todos los módulos importan `db` y las colecciones nombradas desde aquí.
Las credenciales se leen exclusivamente desde variables de entorno (.env).
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── Parámetros de entorno ─────────────────────────────────────────────────────
MONGO_URI: str       = os.environ["MONGO_URI"]
DB_NAME: str         = os.getenv("DB_NAME", "airport_db")
GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str      = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

VECTOR_INDEX_TEXT: str  = os.getenv("VECTOR_INDEX_TEXT",  "vector_index_texto")
VECTOR_INDEX_IMAGE: str = os.getenv("VECTOR_INDEX_IMAGE", "vector_index_imagen")

# Alias para compatibilidad con retriever.py
VECTOR_INDEX: str = VECTOR_INDEX_TEXT

# ── Cliente y base de datos ───────────────────────────────────────────────────
_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10_000)
db = _client[DB_NAME]

# ── Colecciones nombradas ─────────────────────────────────────────────────────
vuelos              = db["vuelos"]
aerolineas          = db["aerolineas"]
pasajeros           = db["pasajeros"]
equipajes           = db["equipajes"]
empleados           = db["empleados"]
reportes            = db["reportes"]
torre_control       = db["torre_control"]
embeddings_texto    = db["embeddings_texto"]
embeddings_imagen   = db["embeddings_imagen"]
historial_consultas = db["historial_consultas"]


def ping() -> bool:
    """Verifica conectividad con el clúster. Retorna True si OK."""
    try:
        _client.admin.command("ping")
        return True
    except Exception:
        return False
