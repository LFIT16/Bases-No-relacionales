from __future__ import annotations

from datetime import datetime, UTC
from typing import Any

import httpx

import config
from rag.retriever import hybrid_search, image_search


# ── Construcción de contexto mejorada (incluye imágenes) ──────────────────────

def build_context(chunks: list[dict], max_chunks: int = 5) -> str:
    """Concatena los chunks (texto e imágenes) en un bloque de contexto para el LLM."""
    selected = chunks[:max_chunks]
    parts = []
    
    for i, c in enumerate(selected, 1):
        meta = c.get("metadatos", {})
        score = c.get("score", 0)
        tipo_fuente = c.get("tipo_fuente", "desconocido")
        
        if tipo_fuente == "imagen":
            # Formato para imágenes
            url = meta.get("url", "")
            tipo_imagen = meta.get("tipo_imagen", "imagen")
            aeropuerto = meta.get("aeropuerto", "")
            fecha = meta.get("fecha", "")
            
            header = f"[Imagen {i} | Tipo: {tipo_imagen} | Score: {score:.3f}"
            if aeropuerto:
                header += f" | Aeropuerto: {aeropuerto}"
            if fecha:
                header += f" | Fecha: {fecha}"
            header += "]"
            
            content = f"{header}\nDescripción: {c['chunk_texto']}"
            if url:
                content += f"\nURL: {url}"
        else:
            # Formato para textos (reportes)
            tipo = meta.get("tipo", "documento")
            fecha = meta.get("fecha", "")
            prioridad = meta.get("prioridad", "")
            
            header = f"[Documento {i} | Tipo: {tipo} | Score: {score:.3f}"
            if fecha:
                header += f" | Fecha: {fecha}"
            if prioridad:
                header += f" | Prioridad: {prioridad}"
            header += "]"
            
            content = f"{header}\n{c['chunk_texto']}"
        
        parts.append(content)
    
    return "\n\n---\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    return f"""Eres un asistente experto en operaciones aeroportuarias.
Responde la pregunta usando EXCLUSIVAMENTE el contexto proporcionado.

REGLAS IMPORTANTES:
1. Usa solo la información del contexto para responder
2. Si el contexto no tiene suficiente información, indícalo claramente
3. Si hay imágenes relevantes, menciónalas en tu respuesta
4. Responde en español de forma clara y concisa

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""


# ── Llamada al LLM (Groq) ─────────────────────────────────────────────────────

def _call_groq(prompt: str) -> str:
    """Llama a la API de Groq con el prompt dado."""
    if not config.GROQ_API_KEY:
        return "[GROQ_API_KEY no configurada — agrega GROQ_API_KEY al archivo .env]"

    payload = {
        "model": config.GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 800,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=payload,
        headers=headers,
        timeout=60,
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
        "fecha": datetime.now(UTC).isoformat(),
        "pregunta": question,
        "chunks_usados": [
            {
                "chunk_index": c.get("chunk_index"),
                "estrategia_chunking": c.get("estrategia_chunking"),
                "tipo_fuente": c.get("tipo_fuente"),
                "score": c.get("score"),
            }
            for c in chunks
        ],
        "respuesta": answer,
    }
    config.historial_consultas.update_one(
        {"usuario": usuario},
        {
            "$push": {"historial": entry},
            "$setOnInsert": {"usuario": usuario, "creado": datetime.now(UTC)},
        },
        upsert=True,
    )


# ── Pipeline RAG principal (MEJORADO) ─────────────────────────────────────────

def rag_query(
    question: str,
    strategy: str | None = None,
    tipo: str | None = None,
    idioma: str | None = None,
    prioridad: str | None = None,
    limit: int = 5,
    usuario: str = "anonimo",
    incluir_imagenes: bool = True,
) -> dict:
    """
    Pipeline RAG completo con soporte multimodal.

    Args:
        question:  Pregunta en lenguaje natural.
        strategy:  Filtro de estrategia de chunking (opcional)
        tipo:      Filtro por tipo de reporte (opcional)
        idioma:    Filtro por idioma (opcional)
        prioridad: Filtro por prioridad (opcional)
        limit:     Número de chunks a recuperar
        usuario:   Identificador de usuario para el historial
        incluir_imagenes: Si True, también busca imágenes relevantes

    Returns:
        {
          "pregunta": str,
          "answer": str,
          "chunks": list[dict],
          "context": str,
        }
    """
    # Recuperar textos
    chunks = hybrid_search(
        query=question,
        strategy=strategy,
        tipo=tipo,
        idioma=idioma,
        prioridad=prioridad,
        limit=limit,
    )
    
    # Recuperar imágenes (NUEVO)
    imagenes = []
    if incluir_imagenes:
        imagenes = image_search(query=question, limit=3)
    
    # Combinar y ordenar por score
    todos = chunks + imagenes
    todos.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    # Generar respuesta
    context = build_context(todos)
    prompt = build_prompt(question, context)
    answer = _call_groq(prompt)
    
    _save_to_history(usuario, question, todos, answer)
    
    return {
        "pregunta": question,
        "answer": answer,
        "chunks": todos,
        "context": context[:500] + "..." if len(context) > 500 else context,
    }