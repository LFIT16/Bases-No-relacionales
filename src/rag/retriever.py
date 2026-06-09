from __future__ import annotations

from typing import Any, Optional

import config
from ingesta.embeddings import embed_text, embed_text_clip


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
    """Búsqueda vectorial básica sobre textos."""
    pipeline = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit,
        num_candidates   = num_candidates,
        filter_query     = None,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_index", "estrategia_chunking",
                             "chunk_texto", "metadatos", "tipo_fuente"],
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
    """Búsqueda vectorial con filtros dinámicos."""
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
                             "chunk_texto", "metadatos", "tipo_fuente"],
    )
    return list(config.embeddings_texto.aggregate(pipeline))


# ── Comparación de estrategias ────────────────────────────────────────────────

def compare_strategies(
    query: str,
    limit: int = 3,
    num_candidates: int = 50,
) -> dict[str, list[dict]]:
    """Ejecuta la misma consulta con cada estrategia de chunking."""
    from chunking import STRATEGIES
    return {
        s: hybrid_search(query, strategy=s, limit=limit, num_candidates=num_candidates)
        for s in STRATEGIES
    }


# ── Búsqueda de imágenes con MiniLM (texto → texto de imagen) ─────────────────

def image_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 50,
) -> list[dict]:
    """Búsqueda de imágenes por descripción textual usando MiniLM."""
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


# ── NUEVO: Búsqueda multimodal con CLIP (texto → imagen real) ─────────────────

def multimodal_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 50,
    tipo_imagen: str | None = None,
) -> list[dict]:
    """
    Búsqueda multimodal usando CLIP (texto → embedding CLIP → imágenes).
    Esto permite encontrar imágenes por su contenido visual, no solo por texto.
    """
    filter_query = {"tipo_fuente": {"$eq": "imagen"}}
    if tipo_imagen:
        filter_query["metadatos.tipo_imagen"] = tipo_imagen
    
    # Usar CLIP para embedding de texto (512 dimensiones)
    query_vector = embed_text_clip(query)
    
    # Necesitaríamos un índice vectorial separado para CLIP (512 dims)
    # Por ahora, esto es un placeholder - requeriría colección separada
    pipeline = _vector_pipeline(
        query_vector     = query_vector,
        limit            = limit,
        num_candidates   = num_candidates,
        filter_query     = filter_query,
        index_name       = "vector_index_clip",  # Índice separado para CLIP
        collection_fields= ["doc_id", "chunk_texto", "metadatos", "tipo_fuente"],
    )
    # Nota: Esto requiere una colección embeddings_imagenes separada
    return []


# ── NUEVO: Consulta híbrida (texto + imágenes) ────────────────────────────────

def hybrid_multimodal_search(
    query: str,
    limit_texto: int = 5,
    limit_imagenes: int = 3,
    tipo_texto: str | None = None,
    tipo_imagen: str | None = None,
) -> dict[str, list[dict]]:
    """
    Realiza búsqueda paralela en textos e imágenes y combina resultados.
    Similar a la consulta híbrida del proyecto relacional.
    """
    # Buscar en textos
    filter_texto = {}
    if tipo_texto:
        filter_texto["metadatos.tipo"] = tipo_texto
    
    pipeline_texto = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit_texto,
        num_candidates   = limit_texto * 10,
        filter_query     = filter_texto or None,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_index", "estrategia_chunking",
                             "chunk_texto", "metadatos", "tipo_fuente"],
    )
    resultados_texto = list(config.embeddings_texto.aggregate(pipeline_texto))
    
    # Buscar en imágenes
    filter_imagen = {"tipo_fuente": {"$eq": "imagen"}}
    if tipo_imagen:
        filter_imagen["metadatos.tipo_imagen"] = tipo_imagen
    
    pipeline_imagen = _vector_pipeline(
        query_vector     = embed_text(query),
        limit            = limit_imagenes,
        num_candidates   = limit_imagenes * 10,
        filter_query     = filter_imagen,
        index_name       = config.VECTOR_INDEX,
        collection_fields= ["doc_id", "chunk_texto", "metadatos", "tipo_fuente"],
    )
    resultados_imagen = list(config.embeddings_texto.aggregate(pipeline_imagen))
    
    # Combinar y ordenar por score
    todos = resultados_texto + resultados_imagen
    todos.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return {
        "textos": resultados_texto,
        "imagenes": resultados_imagen,
        "combinados": todos,
    }


# ── NUEVO: Estadísticas y diagnóstico ─────────────────────────────────────────

def get_stats() -> dict:
    """Retorna estadísticas de la colección de embeddings."""
    total_textos = config.embeddings_texto.count_documents({"tipo_fuente": "reporte"})
    total_imagenes = config.embeddings_texto.count_documents({"tipo_fuente": "imagen"})
    imagenes_con_url = config.embeddings_texto.count_documents({
        "tipo_fuente": "imagen",
        "metadatos.url": {"$exists": True, "$ne": None}
    })
    
    return {
        "total_documentos": config.embeddings_texto.count_documents({}),
        "textos": total_textos,
        "imagenes": total_imagenes,
        "imagenes_con_url": imagenes_con_url,
        "imagenes_sin_url": total_imagenes - imagenes_con_url,
    }