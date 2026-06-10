"""
rag/retriever.py — Búsqueda vectorial e híbrida sobre todas las entidades
del airport_db.

Entidades soportadas:
  • vuelos
  • pasajeros
  • empleados
  • equipajes
  • torre_control
  • reportes
  • aerolineas
  • imagenes

Funciones públicas:
  vector_search(query, ...)
  hybrid_search(query, ...)
  entity_search(query, entidad, ...)
  image_search(query, ...)
  compare_strategies(query, ...)
  multimodal_search(query, ...)
  get_stats()
"""

from __future__ import annotations

from typing import Any

import config
from ingesta.embeddings import embed_text
from PIL import Image
from ingesta.embeddings import embed_image_clip


# ─────────────────────────────────────────────────────────────────────────────
# Tipos soportados
# ─────────────────────────────────────────────────────────────────────────────

TIPOS_FUENTE = (
    "reporte",
    "imagen",
    "vuelo",
    "pasajero",
    "empleado",
    "equipaje",
    "torre_control",
    "aerolinea",
)


# ─────────────────────────────────────────────────────────────────────────────
# Builder del pipeline vectorial
# ─────────────────────────────────────────────────────────────────────────────

def _vector_pipeline(
    query_vector: list[float],
    limit: int,
    num_candidates: int,
    filter_query: dict | None,
    index_name: str,
) -> list[dict]:

    vector_stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": index_name,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": num_candidates,
            "limit": limit,
        }
    }

    if filter_query:
        vector_stage["$vectorSearch"]["filter"] = filter_query

    return [
        vector_stage,
        {
            "$project": {
                "_id": 0,
                "doc_id": 1,
                "chunk_index": 1,
                "estrategia_chunking": 1,
                "chunk_texto": 1,
                "tipo_fuente": 1,
                "metadatos": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Executor interno
# ─────────────────────────────────────────────────────────────────────────────

def _run_vector_search(
    query: str,
    filter_query: dict | None = None,
    limit: int = 5,
    num_candidates: int = 60,
) -> list[dict]:
    """
    Ejecuta búsqueda vectorial sobre embeddings_texto.
    """

    query_vector = embed_text(query)

    pipeline = _vector_pipeline(
        query_vector=query_vector,
        limit=limit,
        num_candidates=num_candidates,
        filter_query=filter_query,
        index_name=config.VECTOR_INDEX,
    )

    return list(config.embeddings_texto.aggregate(pipeline))


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda global
# ─────────────────────────────────────────────────────────────────────────────

def vector_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 60,
) -> list[dict]:
    """
    Búsqueda semántica global sobre TODAS las entidades.
    """
    return _run_vector_search(
        query=query,
        limit=limit,
        num_candidates=num_candidates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda híbrida
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    strategy: str | None = None,
    tipo: str | None = None,
    idioma: str | None = None,
    prioridad: str | None = None,
    tipo_fuente: str | None = None,
    limit: int = 5,
    num_candidates: int = 60,
) -> list[dict]:
    """
    Búsqueda vectorial con filtros dinámicos.
    """

    filters: dict[str, Any] = {}

    if strategy:
        filters["estrategia_chunking"] = strategy

    if tipo:
        filters["metadatos.tipo"] = tipo

    if idioma:
        filters["metadatos.idioma"] = idioma

    if prioridad:
        filters["metadatos.prioridad"] = prioridad

    if tipo_fuente:
        filters["tipo_fuente"] = tipo_fuente

    return _run_vector_search(
        query=query,
        filter_query=filters or None,
        limit=limit,
        num_candidates=num_candidates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda por entidad
# ─────────────────────────────────────────────────────────────────────────────

def entity_search(
    query: str,
    entidad: str,
    limit: int = 5,
    num_candidates: int = 60,
    extra_filters: dict | None = None,
) -> list[dict]:
    """
    Busca únicamente dentro de una entidad.
    """

    tipo = (
        entidad.rstrip("s")
        if entidad.endswith("s") and entidad != "torre_control"
        else entidad
    )

    filter_query: dict[str, Any] = {
        "tipo_fuente": tipo
    }

    if extra_filters:
        filter_query.update(extra_filters)

    return _run_vector_search(
        query=query,
        filter_query=filter_query,
        limit=limit,
        num_candidates=num_candidates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda de imágenes
# ─────────────────────────────────────────────────────────────────────────────

def image_search(
    query: str,
    limit: int = 5,
    num_candidates: int = 50,
) -> list[dict]:
    """
    Busca imágenes usando descripción textual.
    """

    return _run_vector_search(
        query=query,
        filter_query={"tipo_fuente": "imagen"},
        limit=limit,
        num_candidates=num_candidates,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Comparar estrategias
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies(
    query: str,
    limit: int = 3,
    num_candidates: int = 50,
    tipo_fuente: str | None = None,
) -> dict[str, list[dict]]:
    """
    Ejecuta la consulta usando todas las estrategias de chunking.
    """

    from chunking import STRATEGIES

    resultados = {}

    for strategy in STRATEGIES:

        filters: dict[str, Any] = {
            "estrategia_chunking": strategy
        }

        if tipo_fuente:
            filters["tipo_fuente"] = tipo_fuente

        resultados[strategy] = _run_vector_search(
            query=query,
            filter_query=filters,
            limit=limit,
            num_candidates=num_candidates,
        )

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# Multimodal
# ─────────────────────────────────────────────────────────────────────────────

def multimodal_search(
    query: str,
    limit_texto: int = 5,
    limit_imagenes: int = 3,
) -> dict[str, list[dict]]:
    """
    Combina resultados de texto e imágenes.
    """

    textos = _run_vector_search(
        query=query,
        filter_query={"tipo_fuente": {"$ne": "imagen"}},
        limit=limit_texto,
        num_candidates=limit_texto * 12,
    )

    imagenes = image_search(
        query=query,
        limit=limit_imagenes,
        num_candidates=limit_imagenes * 12,
    )

    combinados = sorted(
        textos + imagenes,
        key=lambda x: x.get("score", 0),
        reverse=True,
    )

    return {
        "textos": textos,
        "imagenes": imagenes,
        "combinados": combinados,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Estadísticas
# ─────────────────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    """
    Retorna estadísticas del corpus vectorial.
    """

    pipeline = [
        {
            "$group": {
                "_id": "$tipo_fuente",
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]

    rows = list(config.embeddings_texto.aggregate(pipeline))

    return {
        "total": config.embeddings_texto.count_documents({}),
        "por_tipo": {
            r["_id"]: r["count"]
            for r in rows
        }
    }


def image_to_image_search(
    image_path: str,
    limit: int = 5,
    num_candidates: int = 50,
) -> list[dict]:
    """
    Busca imágenes similares usando una imagen de entrada.
    """
    from ingesta.embeddings import embed_image_clip
    from PIL import Image
    
    # Cargar y embedizar la imagen
    image = Image.open(image_path)
    query_vector = embed_image_clip(image)
    
    # Pipeline de búsqueda vectorial con lookup
    pipeline = [
        {
            "$vectorSearch": {
                "index": config.VECTOR_INDEX_IMAGE,  # ← CORREGIDO
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        {
            "$lookup": {
                "from": "imagenes",
                "localField": "img_id",
                "foreignField": "img_id",
                "as": "imagen_info"
            }
        },
        {
            "$unwind": "$imagen_info"
        },
        {
            "$project": {
                "_id": 0,
                "img_id": 1,
                "filename": 1,
                "score": {"$meta": "vectorSearchScore"},
                "url": "$imagen_info.url",
                "descripcion": "$imagen_info.descripcion",
                "tipo_imagen": "$imagen_info.tipo_imagen",
                "aeropuerto": "$imagen_info.aeropuerto",
            }
        }
    ]
    
    return list(config.embeddings_imagen.aggregate(pipeline))