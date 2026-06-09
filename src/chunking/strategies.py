
from __future__ import annotations

import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Descarga silenciosa del tokenizador de oraciones
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize  # noqa: E402


# ── Fixed-size chunking ───────────────────────────────────────────────────────

def fixed_size_chunk(
    text: str,
    chunk_size: int = 256,
    overlap: int = 32,
) -> list[str]:
    """
    Divide el texto en fragmentos de `chunk_size` caracteres con solapamiento.
    Simple y predecible; útil para textos homogéneos como logs de vuelos.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── Sentence-aware chunking ───────────────────────────────────────────────────

def sentence_aware_chunk(
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1,
    language: str = "spanish",
) -> list[str]:
    """
    Agrupa oraciones completas. No corta en medio de una frase.
    Overlap de `overlap_sentences` oración(es) entre chunks consecutivos.
    Ideal para reportes de incidentes y observaciones narrativas.
    """
    sentences: list[str] = sent_tokenize(text, language=language)
    chunks: list[str] = []
    i = 0
    while i < len(sentences):
        group = sentences[i : i + max_sentences]
        chunk = " ".join(group).strip()
        if chunk:
            chunks.append(chunk)
        step = max(1, max_sentences - overlap_sentences)
        i += step
    return chunks


# ── Semantic chunking ─────────────────────────────────────────────────────────

def semantic_chunk(
    text: str,
    model: SentenceTransformer,
    threshold: float = 0.80,
    language: str = "spanish",
) -> list[str]:
    """
    Agrupa oraciones por similitud semántica usando embeddings del modelo.
    Un nuevo chunk comienza cuando la similitud coseno entre oraciones
    consecutivas cae por debajo de `threshold`.

    Produce chunks temáticamente coherentes; recomendado para documentos
    técnicos con cambios de tema (p. ej., reportes mixtos de torre de control).

    Nota: requiere el modelo ya cargado para no recargar pesos en cada llamada.
    """
    sentences: list[str] = sent_tokenize(text, language=language)
    if len(sentences) <= 1:
        return sentences

    embeddings: np.ndarray = model.encode(sentences, show_progress_bar=False)
    # similitudes entre oraciones contiguas
    sims = cosine_similarity(embeddings[:-1], embeddings[1:]).diagonal()

    # índices donde se produce un quiebre temático
    break_indices = [i + 1 for i, s in enumerate(sims) if s < threshold]

    chunks: list[str] = []
    start = 0
    for b in break_indices:
        chunk = " ".join(sentences[start:b]).strip()
        if chunk:
            chunks.append(chunk)
        start = b
    tail = " ".join(sentences[start:]).strip()
    if tail:
        chunks.append(tail)
    return chunks


# ── Función unificada ─────────────────────────────────────────────────────────

STRATEGIES = ("fixed", "sentence", "semantic")


def chunk_text(
    text: str,
    strategy: str,
    model: SentenceTransformer | None = None,
    **kwargs,
) -> list[str]:
    """
    Punto de entrada único.

    Args:
        text:     Texto a fragmentar.
        strategy: "fixed" | "sentence" | "semantic"
        model:    Obligatorio sólo para strategy="semantic".
        **kwargs: Parámetros opcionales que se pasan a la función interna.

    Returns:
        Lista de chunks (strings).
    """
    if strategy == "fixed":
        return fixed_size_chunk(text, **kwargs)
    if strategy == "sentence":
        return sentence_aware_chunk(text, **kwargs)
    if strategy == "semantic":
        if model is None:
            raise ValueError("semantic chunking requiere un SentenceTransformer cargado.")
        return semantic_chunk(text, model=model, **kwargs)
    raise ValueError(f"Estrategia desconocida: '{strategy}'. Usa: {STRATEGIES}")