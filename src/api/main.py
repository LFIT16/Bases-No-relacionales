"""
api/main.py — API REST del sistema RAG aeroportuario.

Endpoints:
  GET  /health                → ping a MongoDB
  GET  /stats                 → conteo de embeddings por tipo_fuente
  POST /search                → búsqueda híbrida/vectorial general
  POST /search/entity         → búsqueda sobre una entidad específica
  POST /search/images         → búsqueda texto → imagen
  POST /search/multimodal     → textos + imágenes combinados
  POST /rag                   → pipeline RAG completo (Groq + LLaMA)
  POST /rag/entity            → RAG enfocado en una entidad
  POST /chunking/compare      → compara las 3 estrategias de chunking
  POST /ingesta/entidades     → dispara la ingesta de embeddings para entidades
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, BackgroundTasks, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rag.retriever import image_to_image_search

import config
from rag import (
    hybrid_search,
    entity_search,
    compare_strategies,
    image_search,
    multimodal_search,
    rag_query,
    get_stats,
)

app = FastAPI(
    title="Airport RAG API",
    description=(
        "Sistema RAG NoSQL — Gestión de Operaciones Aeroportuarias.\n\n"
        "Entidades con embeddings: reportes, imágenes, vuelos, pasajeros, "
        "empleados, equipajes, torre_control."
    ),
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Archivos estáticos (imágenes) ─────────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas de request
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:       str            = Field(..., min_length=3, description="Consulta en lenguaje natural")
    strategy:    Optional[str]  = Field(None, description="fixed | sentence | semantic")
    tipo:        Optional[str]  = Field(None, description="Tipo de reporte (incidente, climatico, etc.)")
    idioma:      Optional[str]  = Field(None, description="es | en")
    prioridad:   Optional[str]  = Field(None, description="alta | media | baja")
    tipo_fuente: Optional[str]  = Field(None, description="reporte | vuelo | pasajero | empleado | equipaje | torre_control | imagen")
    limit:       int            = Field(5, ge=1, le=20)
    compare:     bool           = Field(False, description="Si True, compara las 3 estrategias de chunking")


class EntitySearchRequest(BaseModel):
    query:   str           = Field(..., min_length=3)
    entidad: str           = Field(..., description="vuelos | pasajeros | empleados | equipajes | torre_control | reportes")
    limit:   int           = Field(5, ge=1, le=20)
    extra_filters: Optional[dict] = Field(None, description="Filtros adicionales sobre metadatos")


class ImageSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Descripción textual de la imagen buscada")
    limit: int = Field(5, ge=1, le=20)


class MultimodalRequest(BaseModel):
    query:          str           = Field(..., min_length=3)
    limit_texto:    int           = Field(5, ge=1, le=20)
    limit_imagenes: int           = Field(3, ge=1, le=10)
    tipo_fuente:    Optional[str] = Field(None)


class RagRequest(BaseModel):
    question:  str           = Field(..., min_length=3)
    strategy:  Optional[str] = None
    tipo:      Optional[str] = None
    idioma:    Optional[str] = None
    prioridad: Optional[str] = None
    limit:     int           = Field(5, ge=1, le=20)
    usuario:   str           = Field("anonimo")
    incluir_imagenes: bool   = Field(True)


class RagEntityRequest(BaseModel):
    question: str           = Field(..., min_length=3)
    entidad:  str           = Field(..., description="vuelos | pasajeros | empleados | equipajes | torre_control | reportes")
    strategy: Optional[str] = None
    limit:    int           = Field(5, ge=1, le=20)
    usuario:  str           = Field("anonimo")


class ChunkingCompareRequest(BaseModel):
    query:       str           = Field(..., min_length=3)
    limit:       int           = Field(3, ge=1, le=10)
    tipo_fuente: Optional[str] = Field(None, description="Filtrar por entidad (opcional)")


class IngestaRequest(BaseModel):
    entidades:  Optional[list[str]] = Field(
        None,
        description="Lista de entidades a ingestar. None = todas. "
                    "Opciones: vuelos, pasajeros, empleados, equipajes, torre_control"
    )
    strategies: list[str] = Field(
        ["fixed", "sentence", "semantic"],
        description="Estrategias de chunking a aplicar"
    )
    clear: bool = Field(False, description="Si True, borra embeddings previos")

class MultimodalRequest(BaseModel):
    query: str
    limit_texto: int = 5
    limit_imagenes: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints de sistema
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["sistema"])
def health() -> dict:
    """Verifica conectividad con MongoDB Atlas."""
    if not config.ping():
        raise HTTPException(status_code=503, detail="MongoDB no disponible")
    return {"status": "ok", "db": config.DB_NAME}


@app.get("/stats", tags=["sistema"])
def stats() -> dict:
    """Retorna el conteo de embeddings por tipo_fuente en embeddings_texto."""
    return get_stats()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints de búsqueda
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/search", tags=["retrieval"])
def search(req: SearchRequest) -> dict:
    """
    Búsqueda vectorial/híbrida sobre el corpus completo.
    Si `compare=true`, retorna resultados separados por estrategia de chunking.
    """
    if req.compare:
        results = compare_strategies(
            req.query,
            limit=req.limit,
            tipo_fuente=req.tipo_fuente,
        )
        return {"query": req.query, "comparacion": results}

    results = hybrid_search(
        query       = req.query,
        strategy    = req.strategy,
        tipo        = req.tipo,
        idioma      = req.idioma,
        prioridad   = req.prioridad,
        tipo_fuente = req.tipo_fuente,
        limit       = req.limit,
    )
    return {"query": req.query, "total": len(results), "resultados": results}


@app.post("/search/entity", tags=["retrieval"])
def search_entity(req: EntitySearchRequest) -> dict:
    """
    Búsqueda vectorial restringida a una entidad específica.
    Ejemplo: buscar vuelos cancelados, pasajeros con categoría oro, etc.
    """
    results = entity_search(
        query         = req.query,
        entidad       = req.entidad,
        limit         = req.limit,
        extra_filters = req.extra_filters,
    )
    return {
        "query":    req.query,
        "entidad":  req.entidad,
        "total":    len(results),
        "resultados": results,
    }


@app.post("/search/images", tags=["retrieval"])
def search_images(req: ImageSearchRequest) -> dict:
    """
    Búsqueda multimodal texto → imagen.
    Encuentra imágenes aeroportuarias (pasajeros, equipaje dañado) por descripción.
    """
    results = image_search(query=req.query, limit=req.limit)
    return {"query": req.query, "total": len(results), "resultados": results}


@app.post("/search/multimodal", tags=["retrieval"])
def search_multimodal(req: MultimodalRequest) -> dict:
    """
    Búsqueda paralela en textos e imágenes, combinados por score de relevancia.
    """
    results = multimodal_search(
        query          = req.query,
        limit_texto    = req.limit_texto,
        limit_imagenes = req.limit_imagenes,
        tipo_fuente    = req.tipo_fuente,
    )
    return {"query": req.query, **results}


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints RAG
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/rag", tags=["generacion"])
def rag(req: RagRequest) -> dict:
    """
    Pipeline RAG completo sobre todo el corpus.
    Recupera contexto de MongoDB (textos + imágenes) y genera respuesta con Groq/LLaMA.
    Guarda la consulta en historial_consultas.
    """
    result = rag_query(
        question         = req.question,
        strategy         = req.strategy,
        tipo             = req.tipo,
        idioma           = req.idioma,
        prioridad        = req.prioridad,
        limit            = req.limit,
        usuario          = req.usuario,
        incluir_imagenes = req.incluir_imagenes,
    )
    return result


@app.post("/rag/entity", tags=["generacion"])
def rag_entity(req: RagEntityRequest) -> dict:
    """
    Pipeline RAG enfocado en una entidad específica.
    Útil para preguntas como:
      - '¿Qué vuelos están cancelados hoy?' → entidad: vuelos
      - '¿Quién es el pasajero con documento X?' → entidad: pasajeros
      - '¿Qué empleados tienen turno nocturno?' → entidad: empleados
    """
    result = rag_query(
        question = req.question,
        strategy = req.strategy,
        entidad  = req.entidad,
        limit    = req.limit,
        usuario  = req.usuario,
        incluir_imagenes = False,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experimento de chunking
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/chunking/compare", tags=["experimento"])
def chunking_compare(req: ChunkingCompareRequest) -> dict:
    """
    Ejecuta la misma consulta con las 3 estrategias de chunking y retorna
    los resultados comparativos.

    Útil para el experimento obligatorio del proyecto:
    analizar qué estrategia produce mejores chunks para el dominio aeroportuario.
    """
    comparacion = compare_strategies(
        query       = req.query,
        limit       = req.limit,
        tipo_fuente = req.tipo_fuente,
    )

    # Construir tabla resumen para facilitar el análisis
    resumen = {}
    for estrategia, chunks in comparacion.items():
        if chunks:
            scores      = [c.get("score", 0) for c in chunks]
            longitudes  = [len(c.get("chunk_texto", "")) for c in chunks]
            resumen[estrategia] = {
                "num_resultados":     len(chunks),
                "score_promedio":     round(sum(scores) / len(scores), 4),
                "score_maximo":       round(max(scores), 4),
                "longitud_promedio":  round(sum(longitudes) / len(longitudes), 1),
                "chunks":             chunks,
            }
        else:
            resumen[estrategia] = {
                "num_resultados": 0,
                "score_promedio": 0,
                "score_maximo":   0,
                "longitud_promedio": 0,
                "chunks": [],
            }

    return {
        "query":      req.query,
        "tipo_fuente": req.tipo_fuente or "todos",
        "resumen":    resumen,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ingesta on-demand
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/ingesta/entidades", tags=["ingesta"])
def ingestar_entidades(req: IngestaRequest, background_tasks: BackgroundTasks) -> dict:
    """
    Dispara la ingesta de embeddings para las entidades indicadas.
    Se ejecuta en background para no bloquear la respuesta HTTP.

    Entidades disponibles: vuelos, pasajeros, empleados, equipajes, torre_control.
    Si `entidades` es null, procesa todas.
    """
    from ingesta.cargar_entidades import cargar_todas_las_entidades

    def _run():
        cargar_todas_las_entidades(
            entidades  = req.entidades,
            strategies = tuple(req.strategies),
            clear      = req.clear,
        )

    background_tasks.add_task(_run)

    entidades_msg = req.entidades or ["vuelos", "pasajeros", "empleados", "equipajes", "torre_control"]
    return {
        "status":     "ingesta iniciada en background",
        "entidades":  entidades_msg,
        "strategies": req.strategies,
        "clear":      req.clear,
        "mensaje":    "Consulta /stats en unos segundos para ver el progreso.",
    }

@app.post("/search/multimodal")
def search_multimodal(req: MultimodalRequest):

    resultados = multimodal_search(
        query=req.query,
        limit_texto=req.limit_texto,
        limit_imagenes=req.limit_imagenes,
    )

    return {
        "query": req.query,
        "textos": resultados["textos"],
        "imagenes": resultados["imagenes"],
        "combinados": resultados["combinados"],
    }


# api/main.py
@app.post("/search/image-to-image")
async def search_image_to_image(file: UploadFile = File(...)):
    import os
    from pathlib import Path
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        path = tmp.name

    resultados = image_to_image_search(image_path=path, limit=6)
    
    # Procesar resultados para asegurar URLs correctas
    for r in resultados:
        filename = r.get('filename', '')
        
        # Construir URL correcta para las imágenes
        if filename:
            # La URL debe ser accesible desde el frontend
            r['url'] = f"/static/images/{filename}"
        else:
            r['url'] = None
        
        # Asegurar descripción
        if not r.get('descripcion'):
            r['descripcion'] = f"Imagen: {filename}" if filename else "Imagen sin descripción"
        
        # Valores por defecto
        r.setdefault('tipo_imagen', 'general')
        r.setdefault('aeropuerto', '')
        r.setdefault('score', 0.0)
    
    # Limpiar archivo temporal
    try:
        os.unlink(path)
    except:
        pass
    
    return {
        "total": len(resultados),
        "resultados": resultados
    }