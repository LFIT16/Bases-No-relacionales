
from config import embeddings_texto

# Actualizar URLs con el puerto correcto
urls = {
    'IMG_001': 'http://localhost:8000/static/images/persona1.png',
    'IMG_002': 'http://localhost:8000/static/images/persona2.webp',
    'IMG_003': 'http://localhost:8000/static/images/persona3.png',
    'IMG_004': 'http://localhost:8000/static/images/persona4.jpg',
    'IMG_005': 'http://localhost:8000/static/images/persona5.jpg',
    'IMG_006': 'http://localhost:8000/static/images/persona6.jpeg',
    'IMG_007': 'http://localhost:8000/static/images/persona7.webp',
    'IMG_008': 'http://localhost:8000/static/images/persona8.avif',
    'IMG_009': 'http://localhost:8000/static/images/persona9.avif',
    'IMG_010': 'http://localhost:8000/static/images/persona10.jpg',
    'IMG_021': 'http://localhost:8000/static/images/equipajeDañado3.webp',
    'IMG_022': 'http://localhost:8000/static/images/equipajeDañado2.webp',
    'IMG_024': 'http://localhost:8000/static/images/equipajeDañado1.jpeg',
    'IMG_027': 'http://localhost:8000/static/images/equipajeDañado4.webp',
}

for doc_id, url in urls.items():
    result = embeddings_texto.update_one(
        {'doc_id': doc_id, 'tipo_fuente': 'imagen'},
        {'$set': {'metadatos.url': url}}
    )
    if result.modified_count > 0:
        print(f'✅ {doc_id}: {url}')
    else:
        print(f'⚠️ {doc_id}: No se encontró')

print('\n¡Actualización completada!')
print('Ahora recarga la página y busca nuevamente')
