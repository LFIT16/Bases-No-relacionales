"""
rag/pipeline.py — Pipeline RAG completo con soporte para todas las entidades.

Flujo:
  1. Recuperar chunks relevantes (textos + imágenes) con hybrid_search
  2. Construir contexto enriquecido con metadatos de la entidad
  3. Armar prompt con instrucciones según el tipo de consulta
  4. Llamar a Groq (LLaMA 3.1) y retornar la respuesta
  5. Guardar en historial_consultas
"""
from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

import httpx

import config
from rag.retriever import hybrid_search, image_search, entity_search


# ─────────────────────────────────────────────────────────────────────────────
# Construcción de contexto
# ─────────────────────────────────────────────────────────────────────────────

# Descripciones legibles por entidad para el prompt
_ENTITY_LABELS = {
    "reporte":       "Reporte aeroportuario",
    "imagen":        "Imagen aeroportuaria",
    "vuelo":         "Vuelo",
    "pasajero":      "Pasajero",
    "empleado":      "Empleado",
    "equipaje":      "Equipaje",
    "torre_control": "Torre de control",
}

# Campos de metadatos relevantes por entidad (para mostrar en el contexto)
_META_FIELDS = {
    "reporte":       ["tipo", "fecha", "prioridad", "idioma"],
    "imagen":        ["tipo_imagen", "aeropuerto", "fecha", "url"],
    "vuelo":         ["numero_vuelo", "origen", "destino", "estado", "fecha", "aerolinea", "terminal"],
    "pasajero":      ["nombre", "documento", "nacionalidad", "categoria_frecuente"],
    "empleado":      ["nombre", "rol", "turno", "terminal", "aerolinea"],
    "equipaje":      ["pasajero_id", "vuelo_id", "estado_trazabilidad", "tipo", "peso_kg"],
    "torre_control": ["fecha", "turno", "controlador", "condicion_meteorologica", "visibilidad_km"],
}


def build_context(chunks: list[dict], max_chunks: int = 6) -> str:
    """Construye el bloque de contexto para el LLM a partir de los chunks."""
    parts = []
    for i, c in enumerate(chunks[:max_chunks], 1):
        tipo   = c.get("tipo_fuente", "documento")
        meta   = c.get("metadatos", {})
        score  = c.get("score", 0)
        label  = _ENTITY_LABELS.get(tipo, tipo.capitalize())

        # Cabecera con metadatos relevantes
        campos = _META_FIELDS.get(tipo, [])
        meta_str = " | ".join(
            f"{k}: {meta[k]}"
            for k in campos
            if meta.get(k) is not None
        )
        header = f"[{label} {i} | Score: {score:.3f}"
        if meta_str:
            header += f" | {meta_str}"
        header += "]"

        content = f"{header}\n{c['chunk_texto']}"

        # Para imágenes, añadir URL si existe
        if tipo == "imagen" and meta.get("url"):
            content += f"\nURL: {meta['url']}"

        parts.append(content)

    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str, entidad: str | None = None) -> str:
    """Construye el prompt para el LLM con contexto enriquecido."""
    if entidad:
        label = _ENTITY_LABELS.get(entidad, entidad)
        dominio = f"sobre {label.lower()}s aeroportuarios"
    else:
        dominio = "sobre operaciones aeroportuarias"

    return f"""Eres un asistente experto en gestión aeroportuaria.
Responde la pregunta {dominio} usando EXCLUSIVAMENTE el contexto proporcionado.

REGLAS:
1. Usa solo la información del contexto para responder.
2. Si el contexto no es suficiente, indícalo claramente.
3. Si hay imágenes relevantes, menciona su URL o descripción.
4. Responde en español, de forma clara, estructurada y concisa.
5. Si encuentras múltiples entidades relacionadas, menciona la relación.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""


# ─────────────────────────────────────────────────────────────────────────────
# Llamada al LLM (Groq)
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    if not config.GROQ_API_KEY:
        return "[GROQ_API_KEY no configurada — agrega GROQ_API_KEY al archivo .env]"

    payload = {
        "model":    config.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens":  800,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Historial
# ─────────────────────────────────────────────────────────────────────────────

def _save_to_history(
    usuario: str,
    question: str,
    chunks: list[dict],
    answer: str,
    entidad: str | None,
) -> None:
    entry: dict[str, Any] = {
        "fecha":     datetime.now(UTC).isoformat(),
        "pregunta":  question,
        "entidad":   entidad or "general",
        "chunks_usados": [
            {
                "chunk_index":         c.get("chunk_index"),
                "estrategia_chunking": c.get("estrategia_chunking"),
                "tipo_fuente":         c.get("tipo_fuente"),
                "score":               c.get("score"),
            }
            for c in chunks
        ],
        "respuesta": answer,
    }
    config.historial_consultas.update_one(
        {"usuario": usuario},
        {
            "$push":       {"historial": entry},
            "$setOnInsert": {"usuario": usuario, "creado": datetime.now(UTC)},
        },
        upsert=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline RAG principal
# ─────────────────────────────────────────────────────────────────────────────

def rag_query(
    question: str,
    strategy: str | None   = None,
    tipo: str | None       = None,
    idioma: str | None     = None,
    prioridad: str | None  = None,
    entidad: str | None    = None,
    limit: int             = 5,
    usuario: str           = "anonimo",
    incluir_imagenes: bool = True,
) -> dict:
    """
    Pipeline RAG completo.

    Args:
        question   : Pregunta en lenguaje natural.
        strategy   : Filtro de estrategia de chunking (opcional).
        tipo       : Filtro por tipo de reporte (opcional).
        idioma     : Filtro por idioma (opcional).
        prioridad  : Filtro por prioridad (opcional).
        entidad    : Limitar la búsqueda a una entidad específica
                     (vuelo | pasajero | empleado | equipaje | torre_control | reporte).
        limit      : Número de chunks de texto a recuperar.
        usuario    : ID de usuario para el historial.
        incluir_imagenes: Si True, también busca imágenes relevantes.

    Returns:
        {pregunta, answer, chunks, context, entidades_encontradas}
    """
    # ── 1. Recuperar chunks ──────────────────────────────────────────────────
    if entidad:
        # Normalizar: "vuelos" → "vuelo"
        tipo_fuente = entidad.rstrip("s") if entidad.endswith("s") and entidad != "torre_control" else entidad
        chunks = entity_search(
            query          = question,
            entidad        = tipo_fuente,
            limit          = limit,
            extra_filters  = _build_extra_filters(strategy, tipo, idioma, prioridad),
        )
    else:
        chunks = hybrid_search(
            query      = question,
            strategy   = strategy,
            tipo       = tipo,
            idioma     = idioma,
            prioridad  = prioridad,
            limit      = limit,
        )

    # ── 2. Imágenes ──────────────────────────────────────────────────────────
    imagenes = []
    if incluir_imagenes and not entidad:
        imagenes = image_search(query=question, limit=3)

    # ── 3. Combinar por score ────────────────────────────────────────────────
    todos = sorted(chunks + imagenes, key=lambda x: x.get("score", 0), reverse=True)

    # ── 4. Construir contexto y prompt ───────────────────────────────────────
    context = build_context(todos)
    prompt  = build_prompt(question, context, entidad)

    # ── 5. Llamar al LLM ─────────────────────────────────────────────────────
    answer = _call_groq(prompt)

    # ── 6. Guardar historial ─────────────────────────────────────────────────
    _save_to_history(usuario, question, todos, answer, entidad)

    # ── 7. Resumen de entidades encontradas ──────────────────────────────────
    tipos_encontrados = list({c.get("tipo_fuente", "?") for c in todos})

    return {
        "pregunta":            question,
        "answer":              answer,
        "chunks":              todos,
        "context":             context[:600] + "…" if len(context) > 600 else context,
        "entidades_encontradas": tipos_encontrados,
    }


def _build_extra_filters(
    strategy: str | None,
    tipo: str | None,
    idioma: str | None,
    prioridad: str | None,
) -> dict | None:
    f: dict = {}
    if strategy:
        f["estrategia_chunking"] = strategy
    if tipo:
        f["metadatos.tipo"] = tipo
    if idioma:
        f["metadatos.idioma"] = idioma
    if prioridad:
        f["metadatos.prioridad"] = prioridad
    return f or None

# Agregar al final de rag/pipeline.py

def call_llm(question: str, contexts: list[str]) -> str:
    """
    Función de compatibilidad para ragas_eval.py.
    Genera respuesta usando el pipeline RAG.
    
    Args:
        question: Pregunta del usuario
        contexts: Lista de chunks (se ignora porque rag_query ya los recupera)
    
    Returns:
        Respuesta generada por el LLM
    """
    response = rag_query(
        question=question,
        limit=5,
        usuario="ragas_eval"
    )
    return response.get("answer", "No se pudo generar respuesta")