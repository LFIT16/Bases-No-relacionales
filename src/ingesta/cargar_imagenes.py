from __future__ import annotations
import json
from datetime import datetime, UTC
from pathlib import Path

from PIL import Image

import config
from ingesta.embeddings import embed_image_clip


def cargar_imagenes():

    # =========================
    # LEER JSON
    # =========================

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    json_path = BASE_DIR / "data" / "imagenes.json"

    with open(json_path, "r", encoding="utf-8") as f:
        imagenes = json.load(f)

    print(f"📄 {len(imagenes)} documentos encontrados en JSON")

    for item in imagenes:

        try:

            # =========================
            # RUTA DE LA IMAGEN
            # =========================

            ruta_imagen = Path("static/images") / item["filename"]

            if not ruta_imagen.exists():
                print(f"⚠️ Imagen no encontrada: {ruta_imagen}")
                continue

            # =========================
            # ABRIR IMAGEN
            # =========================

            image = Image.open(ruta_imagen)

            # =========================
            # GENERAR EMBEDDING
            # =========================

            embedding = embed_image_clip(image)

            # =========================
            # AGREGAR METADATA EXTRA
            # =========================

            item["fecha_ingesta"] = datetime.now(UTC)

            item["metadatos"] = {
                "ruta": str(ruta_imagen),
                "formato_real": image.format,
                "size": image.size,
            }

            # =========================
            # INSERTAR DOCUMENTO
            # =========================

            config.db["imagenes"].insert_one(item)

            # =========================
            # INSERTAR EMBEDDING
            # =========================

            embedding_doc = {
                "img_id": item["img_id"],
                "filename": item["filename"],
                "embedding": embedding,
                "fecha_ingesta": datetime.now(UTC),
            }

            config.embeddings_imagen.insert_one(embedding_doc)

            print(f"✅ {item['img_id']} - {item['filename']}")

        except Exception as e:
            print(f"❌ Error con {item['filename']}: {e}")


if __name__ == "__main__":
    cargar_imagenes()