"""
api/main.py — API REST del sistema RAG aeroportuario.
Endpoints: /health, /search, /rag
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles 
from pydantic import BaseModel, Field

import config
from rag import hybrid_search, compare_strategies, rag_query, image_search

app = FastAPI(
    title="Airport RAG API",
    description="Sistema RAG NoSQL — Gestión de Operaciones Aeroportuarias",
    version="1.0.0",
)

# ── CONFIGURACIÓN CORS ──────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── SERVIR ARCHIVOS ESTÁTICOS (IMÁGENES) ────────────────────────
# ← AÑADE ESTAS 3 LÍNEAS
import os
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ── Schemas ──────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:     str           = Field(..., min_length=3, description="Consulta en lenguaje natural")
    strategy:  Optional[str] = Field(None, description="fixed | sentence | semantic")
    tipo:      Optional[str] = Field(None, description="incidente | climatico | mantenimiento | operativo | seguridad")
    idioma:    Optional[str] = Field(None, description="es | en")
    prioridad: Optional[str] = Field(None, description="alta | media | baja")
    limit:     int           = Field(5, ge=1, le=20)
    compare:   bool          = Field(False, description="Si True, retorna resultados por estrategia de chunking")


class ImageSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Descripción textual de la imagen buscada")
    limit: int = Field(5, ge=1, le=20)


class RagRequest(BaseModel):
    question:  str           = Field(..., min_length=3)
    strategy:  Optional[str] = None
    tipo:      Optional[str] = None
    idioma:    Optional[str] = None
    prioridad: Optional[str] = None
    limit:     int           = Field(5, ge=1, le=20)
    usuario:   str           = Field("anonimo")


# ── Endpoints ───────────────────────────────────────────────────

@app.get("/health", tags=["sistema"])
def health() -> dict:
    """Verifica conectividad con MongoDB Atlas."""
    ok = config.ping()
    if not ok:
        raise HTTPException(status_code=503, detail="MongoDB no disponible")
    return {"status": "ok", "db": config.DB_NAME}


@app.post("/search", tags=["retrieval"])
def search(req: SearchRequest) -> dict:
    """
    Búsqueda vectorial/híbrida sobre el corpus de reportes aeroportuarios.
    Si `compare=true`, retorna resultados separados por estrategia de chunking.
    """
    if req.compare:
        results = compare_strategies(req.query, limit=req.limit)
        return {"query": req.query, "comparacion": results}

    results = hybrid_search(
        query=req.query,
        strategy=req.strategy,
        tipo=req.tipo,
        idioma=req.idioma,
        prioridad=req.prioridad,
        limit=req.limit,
    )
    return {"query": req.query, "total": len(results), "resultados": results}


@app.post("/search/images", tags=["retrieval"])
def search_images(req: ImageSearchRequest) -> dict:
    """
    Búsqueda multimodal texto → imagen usando embeddings CLIP.
    Permite encontrar imágenes aeroportuarias por descripción textual.
    """
    results = image_search(query=req.query, limit=req.limit)
    return {"query": req.query, "total": len(results), "resultados": results}


@app.post("/rag", tags=["generacion"])
def rag(req: RagRequest) -> dict:
    """
    Pipeline RAG completo: recupera contexto de MongoDB y genera respuesta con LLM (Groq).
    Guarda la consulta en el historial del usuario.
    """
    result = rag_query(
        question=req.question,
        strategy=req.strategy,
        tipo=req.tipo,
        idioma=req.idioma,
        prioridad=req.prioridad,
        limit=req.limit,
        usuario=req.usuario,
    )
    return result