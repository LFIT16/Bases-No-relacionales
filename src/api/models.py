"""
api/models.py — Re-exporta las funciones de embedding desde ingesta.embeddings.
Mantiene este módulo como alias para no romper importaciones existentes.
"""
from ingesta.embeddings import (
    get_minilm,
    get_clip,
    embed_text,
    embed_texts_batch,
    embed_text_clip,
    embed_image_clip,
)

__all__ = [
    "get_minilm",
    "get_clip",
    "embed_text",
    "embed_texts_batch",
    "embed_text_clip",
    "embed_image_clip",
]
