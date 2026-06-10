"""
ingesta/cargar_entidades.py — Genera y almacena embeddings para TODAS las
entidades del airport_db que ya existen en MongoDB Atlas.

Entidades cubiertas:
  • vuelos         → texto generado desde campos del documento
  • pasajeros      → nombre + reservas embebidas
  • empleados      → rol + turno + terminal
  • equipajes      → estado de trazabilidad + descripción
  • torre_control  → contenido de observaciones del turno

Cada entidad produce documentos en `embeddings_texto` con:
  - tipo_fuente   = nombre de la colección  (ej. "vuelo")
  - doc_id        = str(_id) del documento origen
  - estrategia_chunking = "fixed" | "sentence" | "semantic"
  - chunk_texto, embedding, modelo, metadatos, fecha_ingesta

Uso:
  python -m ingesta.cargar_entidades               # todas las entidades
  python -m ingesta.cargar_entidades --entidad vuelos
  python -m ingesta.cargar_entidades --clear        # borra embeddings previos
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, UTC
from typing import Callable

import config
from chunking import chunk_text, STRATEGIES
from ingesta.embeddings import embed_texts_batch, get_minilm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers internos
# ─────────────────────────────────────────────────────────────────────────────

def _build_chunk_docs(
    doc_id: str,
    tipo_fuente: str,
    metadatos: dict,
    chunks: list[str],
    strategy: str,
    embeddings: list[list[float]],
) -> list[dict]:
    return [
        {
            "doc_id":               doc_id,
            "chunk_index":          i,
            "estrategia_chunking":  strategy,
            "chunk_texto":          chunk,
            "embedding":            embeddings[i],
            "modelo":               config.EMBEDDING_MODEL,
            "tipo_fuente":          tipo_fuente,
            "metadatos":            metadatos,
            "fecha_ingesta":        datetime.now(UTC),
        }
        for i, chunk in enumerate(chunks)
    ]



def _ingestar_coleccion(
    tipo_fuente: str,
    collection,
    texto_fn: Callable[[dict], str],
    meta_fn: Callable[[dict], dict],
    strategies: tuple[str, ...],
    clear: bool,
) -> dict:
    """
    Lógica genérica de ingesta para cualquier colección con PREVENCIÓN de duplicados.
    """
    if clear:
        deleted = config.embeddings_texto.delete_many({"tipo_fuente": tipo_fuente})
        print(f"  🗑️  {deleted.deleted_count} embeddings previos de '{tipo_fuente}' borrados.")

    docs = list(collection.find({}))
    print(f"\n📦 [{tipo_fuente}] {len(docs)} documentos encontrados.")

    if not docs:
        print(f"  ⚠️  Colección vacía.")
        return {"tipo": tipo_fuente, "docs": 0, "chunks": 0}

    semantic_model = get_minilm()
    total_chunks = 0
    textos_insertados_en_sesion = set()  # ← Prevenir duplicados en memoria

    for doc in docs:
        doc_id = str(doc["_id"])
        texto = texto_fn(doc).strip()
        if not texto:
            continue

        metadatos = meta_fn(doc)

        for strategy in strategies:
            kw = {"model": semantic_model} if strategy == "semantic" else {}
            chunks = chunk_text(texto, strategy=strategy, **kw)
            if not chunks:
                continue

            chunks_nuevos = []
            for chunk in chunks:
                # Verificar duplicado en esta sesión
                clave_sesion = (chunk, tipo_fuente, strategy)
                if clave_sesion in textos_insertados_en_sesion:
                    print(f"  ⚠️ Duplicado evitado: {chunk[:50]}...")
                    continue
                
                # Verificar duplicado en base de datos
                existente = config.embeddings_texto.find_one({
                    "chunk_texto": chunk,
                    "tipo_fuente": tipo_fuente,
                    "estrategia_chunking": strategy
                })
                
                if existente:
                    print(f"  ⚠️ Duplicado existente en DB: {chunk[:50]}...")
                    continue
                
                chunks_nuevos.append(chunk)
                textos_insertados_en_sesion.add(clave_sesion)

            if not chunks_nuevos:
                continue

            # Borrar solo chunks antiguos de este documento específico (no todos)
            config.embeddings_texto.delete_many({
                "doc_id": doc_id,
                "tipo_fuente": tipo_fuente,
                "estrategia_chunking": strategy,
            })

            embeddings = embed_texts_batch(chunks_nuevos)
            chunk_docs = _build_chunk_docs(doc_id, tipo_fuente, metadatos, chunks_nuevos, strategy, embeddings)
            
            if chunk_docs:
                config.embeddings_texto.insert_many(chunk_docs)
                total_chunks += len(chunk_docs)

        print(f"  ✅ {doc_id[:20]}… → {tipo_fuente} ({len(chunks_nuevos)} nuevos)")

    print(f"  📊 Chunks insertados para '{tipo_fuente}': {total_chunks}")
    return {"tipo": tipo_fuente, "docs": len(docs), "chunks": total_chunks}

# ─────────────────────────────────────────────────────────────────────────────
# Funciones de texto y metadatos por entidad
# ─────────────────────────────────────────────────────────────────────────────

# ── VUELOS ───────────────────────────────────────────────────────────────────

def _texto_vuelo(doc: dict) -> str:
    partes = [
        f"Vuelo {doc.get('numero_vuelo', '')} de {doc.get('origen', '')} a {doc.get('destino', '')}.",
        f"Fecha de salida: {doc.get('fecha_salida', '')}.",
        f"Estado: {doc.get('estado', '')}.",
    ]
    if doc.get("aerolinea"):
        partes.append(f"Aerolínea: {doc['aerolinea']}.")
    if doc.get("puerta"):
        partes.append(f"Puerta de embarque: {doc['puerta']}.")
    if doc.get("terminal"):
        partes.append(f"Terminal: {doc['terminal']}.")
    if doc.get("capacidad"):
        partes.append(f"Capacidad: {doc['capacidad']} pasajeros.")
    if doc.get("observaciones"):
        partes.append(f"Observaciones: {doc['observaciones']}.")
    return " ".join(partes)

def _meta_vuelo(doc: dict) -> dict:
    return {
        "numero_vuelo": doc.get("numero_vuelo"),
        "origen":       doc.get("origen"),
        "destino":      doc.get("destino"),
        "estado":       doc.get("estado"),
        "fecha":        doc.get("fecha_salida"),
        "aerolinea":    doc.get("aerolinea"),
        "terminal":     doc.get("terminal"),
    }


# ── PASAJEROS ────────────────────────────────────────────────────────────────

def _texto_pasajero(doc: dict) -> str:
    partes = [f"Pasajero: {doc.get('nombre', '')}. Documento: {doc.get('documento', '')}."]
    if doc.get("nacionalidad"):
        partes.append(f"Nacionalidad: {doc['nacionalidad']}.")
    if doc.get("email"):
        partes.append(f"Contacto: {doc['email']}.")
    if doc.get("categoria_frecuente"):
        partes.append(f"Categoría viajero frecuente: {doc['categoria_frecuente']}.")

    reservas = doc.get("reservas", [])
    if reservas:
        for r in reservas[:5]:  # máximo 5 reservas por texto
            vuelo  = r.get("vuelo_id") or r.get("numero_vuelo", "")
            asiento = r.get("asiento", "")
            clase   = r.get("clase", "")
            estado_r = r.get("estado", "")
            partes.append(
                f"Reserva en vuelo {vuelo}, asiento {asiento}, clase {clase}, estado {estado_r}."
            )
    return " ".join(partes)

def _meta_pasajero(doc: dict) -> dict:
    return {
        "nombre":               doc.get("nombre"),
        "documento":            doc.get("documento"),
        "nacionalidad":         doc.get("nacionalidad"),
        "categoria_frecuente":  doc.get("categoria_frecuente"),
        "num_reservas":         len(doc.get("reservas", [])),
    }


# ── EMPLEADOS ────────────────────────────────────────────────────────────────

def _texto_empleado(doc: dict) -> str:
    partes = [
        f"Empleado: {doc.get('nombre', '')}.",
        f"Rol: {doc.get('rol', '')}.",
        f"Turno: {doc.get('turno', '')}.",
    ]
    if doc.get("terminal"):
        partes.append(f"Terminal asignada: {doc['terminal']}.")
    if doc.get("aerolinea"):
        partes.append(f"Aerolínea: {doc['aerolinea']}.")
    if doc.get("certificaciones"):
        certs = ", ".join(doc["certificaciones"])
        partes.append(f"Certificaciones: {certs}.")
    if doc.get("idiomas"):
        idiomas = ", ".join(doc["idiomas"])
        partes.append(f"Idiomas: {idiomas}.")
    if doc.get("observaciones"):
        partes.append(f"Observaciones: {doc['observaciones']}.")
    return " ".join(partes)

def _meta_empleado(doc: dict) -> dict:
    return {
        "nombre":   doc.get("nombre"),
        "rol":      doc.get("rol"),
        "turno":    doc.get("turno"),
        "terminal": doc.get("terminal"),
        "aerolinea": doc.get("aerolinea"),
    }


# ── EQUIPAJES ────────────────────────────────────────────────────────────────

def _texto_equipaje(doc: dict) -> str:
    partes = [f"Equipaje con identificador {doc.get('_id', '')}."]
    if doc.get("pasajero_id"):
        partes.append(f"Pertenece al pasajero {doc['pasajero_id']}.")
    if doc.get("vuelo_id"):
        partes.append(f"Asociado al vuelo {doc['vuelo_id']}.")
    if doc.get("peso_kg"):
        partes.append(f"Peso: {doc['peso_kg']} kg.")
    if doc.get("tipo"):
        partes.append(f"Tipo: {doc['tipo']}.")

    estado = doc.get("estado_trazabilidad", "")
    if estado:
        partes.append(f"Estado actual: {estado}.")

    historial = doc.get("historial_trazabilidad", [])
    for h in historial[-3:]:  # últimos 3 eventos
        evento    = h.get("evento", "")
        ubicacion = h.get("ubicacion", "")
        fecha_h   = h.get("fecha", "")
        partes.append(f"Evento: {evento} en {ubicacion} ({fecha_h}).")

    if doc.get("descripcion_dano"):
        partes.append(f"Descripción del daño: {doc['descripcion_dano']}.")
    if doc.get("incidente"):
        partes.append(f"Incidente registrado: {doc['incidente']}.")

    return " ".join(partes)

def _meta_equipaje(doc: dict) -> dict:
    return {
        "pasajero_id":        doc.get("pasajero_id"),
        "vuelo_id":           doc.get("vuelo_id"),
        "estado_trazabilidad": doc.get("estado_trazabilidad"),
        "tipo":               doc.get("tipo"),
        "peso_kg":            doc.get("peso_kg"),
    }


# ── TORRE DE CONTROL ─────────────────────────────────────────────────────────

def _texto_torre(doc: dict) -> str:
    partes = []
    if doc.get("fecha"):
        partes.append(f"Registro del {doc['fecha']}.")
    if doc.get("turno"):
        partes.append(f"Turno: {doc['turno']}.")
    if doc.get("controlador"):
        partes.append(f"Controlador: {doc['controlador']}.")
    if doc.get("condicion_meteorologica"):
        partes.append(f"Condición meteorológica: {doc['condicion_meteorologica']}.")
    if doc.get("visibilidad_km") is not None:
        partes.append(f"Visibilidad: {doc['visibilidad_km']} km.")
    if doc.get("viento_kmh") is not None:
        partes.append(f"Viento: {doc['viento_kmh']} km/h.")
    if doc.get("observaciones"):
        partes.append(f"Observaciones: {doc['observaciones']}.")

    vuelos_gestionados = doc.get("vuelos_gestionados", [])
    if vuelos_gestionados:
        partes.append(f"Vuelos gestionados en el turno: {', '.join(str(v) for v in vuelos_gestionados[:10])}.")

    incidentes = doc.get("incidentes_reportados", [])
    for inc in incidentes:
        tipo_inc = inc.get("tipo", "")
        desc_inc = inc.get("descripcion", "")
        partes.append(f"Incidente tipo '{tipo_inc}': {desc_inc}.")

    return " ".join(partes)

def _meta_torre(doc: dict) -> dict:
    return {
        "fecha":                  doc.get("fecha"),
        "turno":                  doc.get("turno"),
        "controlador":            doc.get("controlador"),
        "condicion_meteorologica": doc.get("condicion_meteorologica"),
        "visibilidad_km":         doc.get("visibilidad_km"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mapa de entidades
# ─────────────────────────────────────────────────────────────────────────────

ENTIDADES = {
    "vuelos":       (config.vuelos,       _texto_vuelo,     _meta_vuelo),
    "pasajeros":    (config.pasajeros,    _texto_pasajero,  _meta_pasajero),
    "empleados":    (config.empleados,    _texto_empleado,  _meta_empleado),
    "equipajes":    (config.equipajes,    _texto_equipaje,  _meta_equipaje),
    "torre_control":(config.torre_control,_texto_torre,     _meta_torre),
}


# ─────────────────────────────────────────────────────────────────────────────
# Función principal exportable
# ─────────────────────────────────────────────────────────────────────────────

def cargar_todas_las_entidades(
    entidades: list[str] | None = None,
    strategies: tuple[str, ...] = STRATEGIES,
    clear: bool = False,
) -> list[dict]:
    """
    Genera embeddings para todas (o las indicadas) entidades.

    Args:
        entidades  : lista de nombres de entidades; None = todas.
        strategies : estrategias de chunking a aplicar.
        clear      : si True, borra embeddings previos por tipo_fuente.

    Returns:
        Lista de dicts con estadísticas por entidad.
    """
    if not config.ping():
        raise ConnectionError("❌ No se pudo conectar a MongoDB. Revisa MONGO_URI en .env")

    seleccionadas = entidades or list(ENTIDADES.keys())
    resultados = []

    for nombre in seleccionadas:
        if nombre not in ENTIDADES:
            print(f"⚠️  Entidad desconocida: '{nombre}'. Disponibles: {list(ENTIDADES.keys())}")
            continue

        collection, texto_fn, meta_fn = ENTIDADES[nombre]
        print(f"\n{'─'*50}")
        print(f"🔄 Ingesta: {nombre.upper()}")

        stats = _ingestar_coleccion(
            tipo_fuente = nombre.rstrip("s"),   # "vuelos" → "vuelo", etc.
            collection  = collection,
            texto_fn    = texto_fn,
            meta_fn     = meta_fn,
            strategies  = strategies,
            clear       = clear,
        )
        resultados.append(stats)

    print(f"\n{'='*50}")
    print("✅ INGESTA COMPLETADA")
    print(f"{'='*50}")
    for r in resultados:
        print(f"  {r['tipo']:<20} {r['docs']:>4} docs  →  {r['chunks']:>5} chunks")
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera embeddings para todas las entidades del airport_db"
    )
    parser.add_argument(
        "--entidad", nargs="+",
        choices=list(ENTIDADES.keys()),
        help="Entidades a procesar. Default: todas.",
    )
    parser.add_argument(
        "--strategies", nargs="+",
        default=list(STRATEGIES),
        choices=list(STRATEGIES),
        help="Estrategias de chunking. Default: todas.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Borra embeddings previos de las entidades seleccionadas.",
    )
    args = parser.parse_args()

    try:
        cargar_todas_las_entidades(
            entidades  = args.entidad,
            strategies = tuple(args.strategies),
            clear      = args.clear,
        )
    except ConnectionError as e:
        print(e)
        sys.exit(1)
