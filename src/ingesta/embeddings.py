"""
ingesta/embeddings.py — Carga de modelos y funciones de embedding.

Dos modelos disponibles:
  • MiniLM  (all-MiniLM-L6-v2)      → texto,         384 dimensiones
  • CLIP    (clip-vit-base-patch32)  → imagen/texto,  512 dimensiones

Los modelos se cargan una sola vez (patrón singleton con lru_cache)
para no desperdiciar RAM ni tiempo en cada llamada.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

import config


# ── Singletons ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_minilm() -> SentenceTransformer:
    """Carga MiniLM una sola vez y lo reutiliza en toda la sesión."""
    print(f"⚙️  Cargando modelo: {config.EMBEDDING_MODEL}")
    return SentenceTransformer(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_clip():
    """
    Carga CLIP una sola vez. Retorna (CLIPModel, CLIPProcessor).
    Importación diferida: CLIP solo se carga si se usa búsqueda multimodal.
    """
    from transformers import CLIPModel, CLIPProcessor
    print("⚙️  Cargando modelo CLIP (clip-vit-base-patch32)…")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, proc


# ── Funciones de embedding ────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    """
    Embedding de texto con MiniLM. Dimensión: 512.
    Usado para chunking de reportes y búsqueda semántica.
    """
    vec = get_minilm().encode(text, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32).tolist()


def embed_texts_batch(texts: List[str]) -> List[List[float]]:
    """
    Versión por lotes de embed_text.
    10-30× más eficiente que llamadas individuales en ingesta masiva.
    """
    vecs = get_minilm().encode(texts, batch_size=64, show_progress_bar=False)
    return [np.asarray(v, dtype=np.float32).tolist() for v in vecs]


def embed_text_clip(text: str) -> List[float]:
    """
    Embedding de texto con CLIP. Dimensión: 512.
    Permite búsquedas cruzadas texto ↔ imagen.
    """
    clip_model, clip_proc = get_clip()
    inputs = clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        out    = clip_model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        feat   = clip_model.text_projection(pooled)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype(np.float32).tolist()


def embed_image_clip(image: Image.Image) -> List[float]:
    """
    Embedding de imagen con CLIP. Dimensión: 512.
    Usado para biometría, imágenes de equipaje y documentos de pasajero.
    """
    clip_model, clip_proc = get_clip()
    image  = image.convert("RGB")
    inputs = clip_proc(images=image, return_tensors="pt")
    with torch.no_grad():
        out    = clip_model.vision_model(pixel_values=inputs["pixel_values"])
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        feat   = clip_model.visual_projection(pooled)
        feat   = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().numpy().astype(np.float32).tolist()