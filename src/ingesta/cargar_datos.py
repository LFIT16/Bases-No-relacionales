from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, UTC
from pathlib import Path

import config
from chunking import chunk_text, STRATEGIES
from ingesta.embeddings import embed_texts_batch, embed_text, get_minilm


# ── Helpers de I/O ────────────────────────────────────────────────────────────

def _load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Ingesta de reportes ───────────────────────────────────────────────────────

def _upsert_reporte(doc: dict) -> str:
    """
    Inserta o actualiza un reporte por título.
    Retorna el _id del documento como string.
    $setOnInsert garantiza idempotencia: si ya existe, no lo modifica.
    """
    config.reportes.update_one(
        {"titulo": doc["titulo"]},
        {"$setOnInsert": {**doc, "fecha_ingesta": datetime.now(UTC)}},
        upsert=True,
    )
    stored = config.reportes.find_one({"titulo": doc["titulo"]}, {"_id": 1})
    return str(stored["_id"])


def _build_chunk_docs(
    doc_id: str,
    doc_meta: dict,
    chunks: list[str],
    strategy: str,
    embeddings: list[list[float]],
) -> list[dict]:
    """Construye lista de documentos listos para embeddings_texto."""
    return [
        {
            "doc_id":              doc_id,
            "chunk_index":         i,
            "estrategia_chunking": strategy,
            "chunk_texto":         chunk,
            "embedding":           embeddings[i],
            "modelo":              config.EMBEDDING_MODEL,
            "tipo_fuente":         "reporte",
            "metadatos": {
                "tipo":      doc_meta.get("tipo"),
                "fecha":     doc_meta.get("fecha"),
                "idioma":    doc_meta.get("idioma", "es"),
                "prioridad": doc_meta.get("prioridad"),
                "tags":      doc_meta.get("tags", []),
            },
            "fecha_ingesta": datetime.now(UTC),
        }
        for i, chunk in enumerate(chunks)
    ]


def cargar_reportes(
    file_path: str | Path,
    strategies: tuple[str, ...] = STRATEGIES,
    clear_chunks: bool = False,
) -> dict:
    
    docs = _load_json(file_path)
    print(f"\n📄 {len(docs)} reportes encontrados en {file_path}")

    if clear_chunks:
        deleted = config.embeddings_texto.delete_many({"tipo_fuente": "reporte"})
        print(f"🗑️  {deleted.deleted_count} chunks anteriores borrados.")

    # Singleton: no recarga pesos en cada iteración
    semantic_model = get_minilm()
    stats = {"reportes": 0, "chunks_total": 0}

    for doc in docs:
        doc_id = _upsert_reporte(doc)
        texto  = doc.get("contenido_texto", "").strip()
        if not texto:
            continue
        stats["reportes"] += 1

        for strategy in strategies:
            kw = {"model": semantic_model} if strategy == "semantic" else {}
            chunks = chunk_text(texto, strategy=strategy, **kw)
            if not chunks:
                continue

            embeddings  = embed_texts_batch(chunks)
            chunk_docs  = _build_chunk_docs(doc_id, doc, chunks, strategy, embeddings)
            config.embeddings_texto.insert_many(chunk_docs)
            stats["chunks_total"] += len(chunk_docs)
            print(f"  ✅ [{strategy:9s}] '{doc['titulo'][:48]}' → {len(chunk_docs)} chunks")

    print(f"\n📊 Reportes: {stats['reportes']}  |  Chunks totales: {stats['chunks_total']}")
    return stats


# ── Ingesta de imágenes ───────────────────────────────────────────────────────

def cargar_imagenes(file_path: str | Path, clear_prev: bool = False) -> dict:
    """
    Ingesta metadatos de imágenes usando embeddings MiniLM (384d) sobre la
    descripción textual. Se almacena en embeddings_texto con tipo_fuente='imagen'
    para usar el mismo índice vectorial que los reportes (limitación M0 de Atlas).
    """
    metas = _load_json(file_path)
    print(f"\n🖼️  {len(metas)} imágenes encontradas en {file_path}")

    if clear_prev:
        deleted = config.embeddings_texto.delete_many({"tipo_fuente": "imagen"})
        print(f"🗑️  {deleted.deleted_count} embeddings de imágenes anteriores borrados.")

    stats = {"imagenes": 0}

    for meta in metas:
        descripcion = meta.get("descripcion", "")
        embedding = embed_text(descripcion)

        config.embeddings_texto.update_one(
            {"doc_id": meta["img_id"], "tipo_fuente": "imagen"},
            {"$setOnInsert": {
                "doc_id":              meta["img_id"],
                "chunk_index":         0,
                "estrategia_chunking": "fixed",
                "chunk_texto":         descripcion,
                "embedding":           embedding,
                "modelo":              config.EMBEDDING_MODEL,
                "tipo_fuente":         "imagen",
                "metadatos": {
                    "filename":    meta.get("filename"),
                    "url":         meta.get("url"),  
                    "tipo_imagen": meta.get("tipo_imagen"),
                    "aeropuerto":  meta.get("aeropuerto"),
                    "fecha":       meta.get("fecha"),
                    "tags":        meta.get("tags", []),
                },
                "fecha_ingesta": datetime.now(UTC),
            }},
            upsert=True,
        )
        stats["imagenes"] += 1
        print(f"  ✅ {meta['img_id']} — {meta['filename']}")

    print(f"\n📊 Imágenes ingresadas: {stats['imagenes']}")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta del corpus aeroportuario")
    parser.add_argument("--reportes",  metavar="FILE", help="JSON de reportes")
    parser.add_argument("--imagenes",  metavar="FILE", help="JSON de metadatos de imágenes")
    parser.add_argument("--clear",     action="store_true", help="Borrar datos previos")
    parser.add_argument(
        "--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES),
        help="Estrategias de chunking a aplicar (default: todas)",
    )
    args = parser.parse_args()

    if not config.ping():
        print("❌  No se pudo conectar a MongoDB. Revisa MONGO_URI en .env")
        sys.exit(1)

    if args.reportes:
        cargar_reportes(args.reportes, strategies=tuple(args.strategies), clear_chunks=args.clear)
    if args.imagenes:
        cargar_imagenes(args.imagenes, clear_prev=args.clear)
    if not args.reportes and not args.imagenes:
        parser.print_help()