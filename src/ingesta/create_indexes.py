import sys
sys.path.insert(0, 'src')
import config

def crear_todos_los_indices():
    """Crea todos los índices necesarios para prevenir duplicados"""
    
    print("🔧 CREANDO ÍNDICES ÚNICOS")
    print("="*50)
    
    # =========================================================
    # ÍNDICES PARA embeddings_texto
    # =========================================================
    print("\n📄 ÍNDICES PARA embeddings_texto:")
    print("-"*30)
    
    # Índice 1: Evitar el mismo texto para el mismo tipo_fuente y estrategia
    try:
        config.embeddings_texto.create_index(
            [("chunk_texto", 1), ("tipo_fuente", 1), ("estrategia_chunking", 1)],
            unique=True,
            name="unique_texto_tipo_estrategia"
        )
        print("✅ Índice: chunk_texto + tipo_fuente + estrategia")
    except Exception as e:
        print(f"⚠️ Error creando índice único texto: {e}")
    
    # Índice 2: Evitar duplicados por documento
    try:
        config.embeddings_texto.create_index(
            [("doc_id", 1), ("chunk_index", 1), ("estrategia_chunking", 1)],
            unique=True,
            name="unique_doc_chunk_strategy"
        )
        print("✅ Índice: doc_id + chunk_index + estrategia")
    except Exception as e:
        print(f"⚠️ Error creando índice único documento: {e}")
    
    # =========================================================
    # ÍNDICES PARA embeddings_imagen
    # =========================================================
    print("\n🖼️ ÍNDICES PARA embeddings_imagen:")
    print("-"*30)
    
    # Índice 3: Evitar duplicados por img_id
    try:
        # Eliminar índice anterior si existe
        try:
            config.embeddings_imagen.drop_index("unique_img_id")
        except:
            pass
        
        config.embeddings_imagen.create_index(
            [("img_id", 1)],
            unique=True,
            name="unique_img_id"
        )
        print("✅ Índice único: img_id (previene duplicados de imágenes)")
    except Exception as e:
        print(f"⚠️ Error creando índice único img_id: {e}")
    
    # =========================================================
    # ÍNDICES PARA colección imagenes
    # =========================================================
    print("\n📸 ÍNDICES PARA colección imagenes:")
    print("-"*30)
    
    # Índice 4: Evitar duplicados por img_id
    try:
        config.db["imagenes"].create_index(
            [("img_id", 1)],
            unique=True,
            name="unique_img_id"
        )
        print("✅ Índice único: imagenes.img_id")
    except Exception as e:
        print(f"⚠️ Error creando índice único para imagenes: {e}")
    
    # =========================================================
    # ÍNDICES ADICIONALES PARA BÚSQUEDA
    # =========================================================
    print("\n🔍 ÍNDICES ADICIONALES PARA BÚSQUEDA:")
    print("-"*30)
    
    # Índice 5: Para filtrar por tipo_fuente
    try:
        config.embeddings_texto.create_index(
            [("tipo_fuente", 1)],
            name="idx_tipo_fuente"
        )
        print("✅ Índice: tipo_fuente")
    except Exception as e:
        print(f"⚠️ Error creando índice tipo_fuente: {e}")
    
    # Índice 6: Para filtrar por estrategia de chunking
    try:
        config.embeddings_texto.create_index(
            [("estrategia_chunking", 1)],
            name="idx_estrategia_chunking"
        )
        print("✅ Índice: estrategia_chunking")
    except Exception as e:
        print(f"⚠️ Error creando índice estrategia_chunking: {e}")
    
    # =========================================================
    # MOSTRAR ÍNDICES EXISTENTES
    # =========================================================
    print("\n" + "="*50)
    print("📋 ÍNDICES EXISTENTES EN embeddings_texto:")
    print("="*50)
    for idx in config.embeddings_texto.index_information():
        print(f"  - {idx}")
    
    print("\n📋 ÍNDICES EXISTENTES EN embeddings_imagen:")
    print("="*50)
    for idx in config.embeddings_imagen.index_information():
        print(f"  - {idx}")
    
    print("\n📋 ÍNDICES EXISTENTES EN imagenes:")
    print("="*50)
    for idx in config.db["imagenes"].index_information():
        print(f"  - {idx}")
    
    print("\n✅ TODOS LOS ÍNDICES CREADOS CORRECTAMENTE")

def limpiar_duplicados_imagenes():
    """Limpia duplicados existentes en embeddings_imagen"""
    
    print("\n" + "="*50)
    print("🧹 LIMPIANDO DUPLICADOS EXISTENTES")
    print("="*50)
    
    # Encontrar duplicados por img_id
    pipeline = [
        {
            "$group": {
                "_id": "$img_id",
                "ids": {"$push": "$_id"},
                "count": {"$sum": 1}
            }
        },
        {
            "$match": {
                "count": {"$gt": 1}
            }
        }
    ]
    
    duplicados = list(config.embeddings_imagen.aggregate(pipeline))
    
    if not duplicados:
        print("✅ No hay duplicados en embeddings_imagen")
        return
    
    total_eliminados = 0
    for dup in duplicados:
        ids_a_eliminar = dup["ids"][1:]
        resultado = config.embeddings_imagen.delete_many({"_id": {"$in": ids_a_eliminar}})
        total_eliminados += resultado.deleted_count
        print(f"  - {dup['_id']}: eliminados {resultado.deleted_count} duplicados")
    
    print(f"\n✅ Total eliminados: {total_eliminados}")

if __name__ == "__main__":
    # Primero limpiar duplicados existentes
    limpiar_duplicados_imagenes()
    
    # Luego crear índices
    crear_todos_los_indices()