from __future__ import annotations

from typing import Any

import config
from ingesta.embeddings import embed_text


# ── Constructor de pipeline $vectorSearch ─────────────────────────────────────

def _vector_pipeline(
    query_vector: list[float],
    limit: int,
    num_candidates: int,
    filter_query: dict | None,
    index_name: str,
    collection_fields: list[str],
) -> list[dict]:
    """Arma el pipeline de agregación con $vectorSearch y $project."""
    vs_stage: dict[str, Any] = {
        "$vectorSearch": {
            "index":         index_name,
            "path":          "embedding",
            "queryVector":   query_vector,
            "numCandidates": num_candidates,
            "limit":         limit,
        }
    }
    if filter_query:
        vs_stage["$vectorSearch"]["filter"] = filter_query

    project = {f: 1 for f in collection_fields}
    project["_id"]   = 0
    project["score"] = {"$meta": "vectorSearchScore"}

    return [vs_stage, {"$project": project}]


# ── Búsqueda semántica sobre texto ────────────────────────────────────────────

def vector_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 60,
) -> list[dict]:
   
    pipeline = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit,
        num_candidates   = num_candidates,
        filter_query     = None,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_index", "estrategia_chunking",
                             "chunk_texto", "metadatos"],
    )
    return list(config.embeddings_texto.aggregate(pipeline))


# ── Búsqueda híbrida (semántica + filtros) ────────────────────────────────────

def hybrid_search(
    query: str,
    strategy: str | None  = None,
    tipo: str | None       = None,
    idioma: str | None     = None,
    prioridad: str | None  = None,
    limit: int             = 5,
    num_candidates: int    = 60,
) -> list[dict]:
   
    filter_query: dict = {}
    if strategy:
        filter_query["estrategia_chunking"] = strategy
    if tipo:
        filter_query["metadatos.tipo"] = tipo
    if idioma:
        filter_query["metadatos.idioma"] = idioma
    if prioridad:
        filter_query["metadatos.prioridad"] = prioridad

    pipeline = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit,
        num_candidates   = num_candidates,
        filter_query     = filter_query or None,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_index", "estrategia_chunking",
                             "chunk_texto", "metadatos"],
    )
    return list(config.embeddings_texto.aggregate(pipeline))


# ── Comparación de estrategias ────────────────────────────────────────────────

def compare_strategies(
    query: str,
    limit: int = 3,
    num_candidates: int = 50,
) -> dict[str, list[dict]]:
    """
    Ejecuta la misma consulta con cada una de las tres estrategias.
    Retorna un dict {"fixed": [...], "sentence": [...], "semantic": [...]}.
    Usado en el experimento comparativo del proyecto.
    """
    from chunking import STRATEGIES
    return {
        s: hybrid_search(query, strategy=s, limit=limit, num_candidates=num_candidates)
        for s in STRATEGIES
    }


# ── Búsqueda de imágenes (texto → imagen via MiniLM) ─────────────────────────

def image_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 50,
) -> list[dict]:
    """
    Búsqueda de imágenes por descripción textual usando MiniLM (384d).
    """
    # Filtrar por tipo_fuente = "imagen" (sintaxis correcta de MongoDB)
    filter_query = {"tipo_fuente": {"$eq": "imagen"}}
    
    pipeline = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit,
        num_candidates   = num_candidates,
        filter_query     = filter_query,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_texto", "metadatos", "tipo_fuente"],
    )
    return list(config.embeddings_texto.aggregate(pipeline))