
from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

import httpx

import config
from rag.retriever import hybrid_search


# ── Construcción de contexto ──────────────────────────────────────────────────

def build_context(chunks: list[dict], max_chunks: int = 5) -> str:
    """Concatena los chunks recuperados en un bloque de contexto para el LLM."""
    selected = chunks[:max_chunks]
    parts = []
    for i, c in enumerate(selected, 1):
        meta = c.get("metadatos", {})
        header = f"[Fuente {i} | tipo={meta.get('tipo','?')} | fecha={meta.get('fecha','?')}]"
        parts.append(f"{header}\n{c['chunk_texto']}")
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return (
        "Eres un asistente experto en operaciones aeroportuarias. "
        "Responde la pregunta usando únicamente el contexto proporcionado. "
        "Si el contexto no tiene suficiente información, indícalo explícitamente.\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {question}\n\n"
        "RESPUESTA:"
    )


# ── Llamada al LLM (Groq) ─────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """Llama a la API de Groq con el prompt dado."""
    if not config.GROQ_API_KEY:
        return "[GROQ_API_KEY no configurada — agrega GROQ_API_KEY al archivo .env]"

    payload = {
        "model": config.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
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
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


# ── Persistencia de historial ─────────────────────────────────────────────────

def _save_to_history(
    usuario: str,
    question: str,
    chunks: list[dict],
    answer: str,
) -> None:
    """Guarda la consulta y su respuesta en historial_consultas."""
    entry: dict[str, Any] = {
        "fecha":    datetime.now(UTC).isoformat(),
        "pregunta": question,
        "chunks_usados": [
            {
                "chunk_index":         c.get("chunk_index"),
                "estrategia_chunking": c.get("estrategia_chunking"),
                "score":               c.get("score"),
            }
            for c in chunks
        ],
        "respuesta": answer,
    }
    config.historial_consultas.update_one(
        {"usuario": usuario},
        {
            "$push":        {"historial": entry},
            "$setOnInsert": {"usuario": usuario, "creado": datetime.now(UTC)},
        },
        upsert=True,
    )


# ── Pipeline RAG principal ────────────────────────────────────────────────────

def rag_query(
    question: str,
    strategy: str | None = None,
    tipo: str | None = None,
    idioma: str | None = None,
    prioridad: str | None = None,
    limit: int = 5,
    usuario: str = "anonimo",
) -> dict:
    """
    Pipeline RAG completo.

    Args:
        question:  Pregunta en lenguaje natural.
        strategy:  Filtro de estrategia de chunking (opcional): fixed | sentence | semantic
        tipo:      Filtro por tipo de reporte (opcional).
        idioma:    Filtro por idioma (opcional).
        prioridad: Filtro por prioridad (opcional).
        limit:     Número de chunks a recuperar.
        usuario:   Identificador de usuario para el historial.

    Returns:
        {
          "pregunta": str,
          "answer": str,
          "chunks": list[dict],
          "context": str,
        }
    """
    chunks = hybrid_search(
        query=question,
        strategy=strategy,
        tipo=tipo,
        idioma=idioma,
        prioridad=prioridad,
        limit=limit,
    )

    context = build_context(chunks)
    prompt  = build_prompt(question, context)
    answer  = _call_groq(prompt)

    _save_to_history(usuario, question, chunks, answer)

    return {
        "pregunta": question,
        "answer":   answer,
        "chunks":   chunks,
        "context":  context,
    }
