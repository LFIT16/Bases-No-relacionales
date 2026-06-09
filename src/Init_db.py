import os
import time
from pymongo import MongoClient, ASCENDING, TEXT
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = os.getenv("DB_NAME", "airport_db")

if not MONGO_URI:
    raise EnvironmentError("❌ MONGO_URI no encontrada. Crea un archivo .env con tu URI de MongoDB.")

print("🔌 Conectando a MongoDB Atlas...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

try:
    client.admin.command("ping")
    print(f"✅ Conectado a: {DB_NAME}\n")
except Exception as e:
    raise ConnectionError(f"❌ No se pudo conectar: {e}")


# ── COLECCIONES ───────────────────────────────────────────────────────────────
COLECCIONES = [
    "vuelos", "aerolineas", "pasajeros", "equipajes", "empleados",
    "reportes", "torre_control", "embeddings_texto", "embeddings_imagen",
    "historial_consultas",
]

print("📁 Creando colecciones...")
existentes = db.list_collection_names()
for col in COLECCIONES:
    if col not in existentes:
        db.create_collection(col)
        print(f"  ✅ Creada: {col}")
    else:
        print(f"  ℹ️  Ya existe (sin cambios): {col}")


# ── VALIDADORES ───────────────────────────────────────────────────────────────
print("\n📋 Aplicando validadores de esquema...")

db.command("collMod", "vuelos", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["numero_vuelo", "origen", "destino", "fecha_salida", "estado"],
        "properties": {
            "numero_vuelo": {"bsonType": "string"},
            "origen":       {"bsonType": "string", "minLength": 3, "maxLength": 3},
            "destino":      {"bsonType": "string", "minLength": 3, "maxLength": 3},
            "fecha_salida": {"bsonType": "string"},
            "estado":       {"enum": ["programado", "embarcando", "en_vuelo", "aterrizado", "cancelado"]}
        }
    }
}, validationLevel="moderate")
print("  ✅ Validador aplicado: vuelos")

db.command("collMod", "pasajeros", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["nombre", "documento"],
        "properties": {
            "nombre":    {"bsonType": "string"},
            "documento": {"bsonType": "string"},
            "reservas":  {"bsonType": "array"}
        }
    }
}, validationLevel="moderate")
print("  ✅ Validador aplicado: pasajeros")

db.command("collMod", "embeddings_texto", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["doc_id", "chunk_index", "estrategia_chunking", "chunk_texto", "embedding", "modelo"],
        "properties": {
            "doc_id":               {"bsonType": "string"},
            "chunk_index":          {"bsonType": "int"},
            "estrategia_chunking":  {"enum": ["fixed", "sentence", "semantic"]},
            "chunk_texto":          {"bsonType": "string"},
            "embedding":            {"bsonType": "array"},
            "modelo":               {"bsonType": "string"}
        }
    }
}, validationLevel="moderate")
print("  ✅ Validador aplicado: embeddings_texto")

db.command("collMod", "reportes", validator={
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["tipo", "titulo", "contenido_texto", "fecha"],
        "properties": {
            "tipo":            {"enum": ["incidente", "seguridad", "climatico", "mantenimiento", "operativo"]},
            "titulo":          {"bsonType": "string"},
            "contenido_texto": {"bsonType": "string"},
            "fecha":           {"bsonType": "string"}
        }
    }
}, validationLevel="moderate")
print("  ✅ Validador aplicado: reportes")


# ── ÍNDICES B-TREE Y TEXTO ────────────────────────────────────────────────────
def crear_indice(coleccion, keys, **kwargs):
    nombre = kwargs.get("name", "")
    try:
        db[coleccion].create_index(keys, **kwargs)
        print(f"  ✅ {coleccion}: índice creado ({nombre})")
    except OperationFailure as e:
        if e.code == 85:
            print(f"  ℹ️  {coleccion}: índice ya existe, se omite ({nombre})")
        else:
            raise

print("\n🔍 Creando índices tradicionales...")

crear_indice("vuelos", "numero_vuelo", unique=True, name="idx_vuelos_numero_unico")
crear_indice("vuelos", [("fecha_salida", ASCENDING), ("estado", ASCENDING), ("puerta", ASCENDING)], name="idx_vuelos_fecha_estado_puerta")
crear_indice("vuelos", [("fecha_salida", ASCENDING), ("idioma", ASCENDING)], name="idx_vuelos_fecha_idioma")
crear_indice("pasajeros", "documento", unique=True, name="idx_pasajeros_documento_unico")
crear_indice("aerolineas", "codigo_iata", unique=True, name="idx_aerolineas_iata_unico")
crear_indice("empleados", [("rol", ASCENDING), ("turno", ASCENDING), ("terminal", ASCENDING)], name="idx_empleados_rol_turno_terminal")
crear_indice("reportes", [("tipo", ASCENDING), ("fecha", ASCENDING), ("prioridad", ASCENDING)], name="idx_reportes_tipo_fecha_prioridad")
crear_indice("reportes", [("contenido_texto", TEXT)], name="idx_reportes_texto_full")
crear_indice("equipajes", [("pasajero_id", ASCENDING), ("estado_trazabilidad", ASCENDING)], name="idx_equipajes_pasajero_estado")
crear_indice("torre_control", [("fecha", ASCENDING), ("turno", ASCENDING)], name="idx_torre_fecha_turno")
crear_indice("embeddings_texto", [("doc_id", ASCENDING), ("chunk_index", ASCENDING)], name="idx_chunks_doc_index")
crear_indice("embeddings_texto", "estrategia_chunking", name="idx_chunks_estrategia")
crear_indice("historial_consultas", [("usuario", ASCENDING), ("fecha", ASCENDING)], name="idx_historial_usuario_fecha")


# ── ÍNDICES ATLAS VECTOR SEARCH ───────────────────────────────────────────────
print("\n🧠 Creando índices Atlas Vector Search...")

def crear_vector_index(coleccion, nombre, num_dims):
    """Crea un índice vectorial en Atlas. Lo omite si ya existe."""
    col = db[coleccion]

    # Verificar si ya existe
    try:
        existentes = list(col.list_search_indexes())
        nombres_existentes = [idx.get("name") for idx in existentes]
        if nombre in nombres_existentes:
            print(f"  ℹ️  {coleccion}: índice vectorial ya existe ({nombre})")
            return
    except Exception:
        pass  # Si falla el listado, intentamos crear igual

    modelo = SearchIndexModel(
        definition={
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": num_dims,
                    "similarity": "cosine",
                }
            ]
        },
        name=nombre,
        type="vectorSearch",
    )
    try:
        col.create_search_index(modelo)
        print(f"  ✅ {coleccion}: índice vectorial enviado ({nombre}, {num_dims}d)")
    except Exception as e:
        print(f"  ⚠️  {coleccion}: no se pudo crear el índice vectorial — {e}")
        print(f"      → Créalo manualmente en Atlas UI con numDimensions={num_dims}")

crear_vector_index("embeddings_texto",  "vector_index_texto",  384)
crear_vector_index("embeddings_imagen", "vector_index_imagen", 512)

# Esperar a que Atlas procese los índices (pueden tardar 1-3 min en activarse)
print("\n⏳ Esperando 10s para que Atlas registre los índices...")
time.sleep(10)

# Verificar estado de los índices vectoriales
print("\n📡 Estado de índices vectoriales:")
for coleccion, nombre in [("embeddings_texto", "vector_index_texto"), ("embeddings_imagen", "vector_index_imagen")]:
    try:
        indices = list(db[coleccion].list_search_indexes())
        for idx in indices:
            if idx.get("name") == nombre:
                estado = idx.get("status", "desconocido")
                emoji = "✅" if estado == "READY" else "🔄"
                print(f"  {emoji} {coleccion} / {nombre}: {estado}")
    except Exception as e:
        print(f"  ⚠️  No se pudo verificar {coleccion}: {e}")

print("\n💡 Si el estado es 'BUILDING', espera 1-2 minutos y el índice se activará solo.")
print("   El sistema RAG funcionará normalmente una vez que el estado sea 'READY'.")


# ── VERIFICACIÓN FINAL ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  VERIFICACIÓN FINAL — airport_db")
print("=" * 50)

for col in sorted(db.list_collection_names()):
    count = db[col].count_documents({})
    n_idx = sum(1 for _ in db[col].list_indexes())
    print(f"  {col:<25} {count:>3} docs   {n_idx} índice(s)")

print("\n✅ Inicialización completada exitosamente")