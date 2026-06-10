const API = 'http://localhost:8000';

const STATIC_URL = 'http://localhost:8000/static';  
const TIPO_ICONS = {
  biometria: '👤', documento_identidad: '🪪', equipaje: '🧳',
  incidente: '⚠️', infraestructura: '🏗️'
};

async function checkHealth() {
  try {
    const r = await fetch(API + '/health');
    const d = await r.json();
    document.getElementById('dot').className = 'status-dot';
    document.getElementById('status-text').textContent = 'Conectado · ' + d.db;
  } catch {
    document.getElementById('dot').className = 'status-dot red';
    document.getElementById('status-text').textContent = 'Sin conexión';
  }
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

function badgeStrategy(s) {
  const cls = s === 'fixed' ? 'badge-fixed' : s === 'sentence' ? 'badge-sentence' : 'badge-semantic';
  return `<span class="badge ${cls}">${s}</span>`;
}

function renderChunks(chunks) {
  if (!chunks || chunks.length === 0)
    return '<p style="color:#888; font-size:0.9rem; padding:1rem">No se encontraron chunks. Verifica que el índice vectorial esté activo (READY) en Atlas.</p>';
  return chunks.map(c => `
    <div class="chunk">
      <div class="chunk-header">
        ${badgeStrategy(c.estrategia_chunking)}
        ${c.metadatos?.tipo ? `<span class="badge badge-tipo">${c.metadatos.tipo}</span>` : ''}
        ${c.metadatos?.fecha ? `<span class="badge" style="background:#f5f5f5;color:#555">${c.metadatos.fecha}</span>` : ''}
        ${c.score ? `<span class="badge badge-score">score: ${c.score.toFixed(3)}</span>` : ''}
      </div>
      <div class="chunk-text">${c.chunk_texto}</div>
    </div>`).join('');
}

async function sendRag() {
  const q = document.getElementById('rag-question').value.trim();
  if (!q) return;
  const body = { question: q, limit: parseInt(document.getElementById('rag-limit').value) };
  const tipo = document.getElementById('rag-tipo').value;
  const strat = document.getElementById('rag-strategy').value;
  const prio = document.getElementById('rag-prioridad').value;
  if (tipo) body.tipo = tipo;
  if (strat) body.strategy = strat;
  if (prio) body.prioridad = prio;
  body.usuario = 'frontend';

  document.getElementById('rag-loader').classList.add('active');
  document.getElementById('rag-result').innerHTML = '';

  try {
    const r = await fetch(API + '/rag', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const d = await r.json();
    document.getElementById('rag-result').innerHTML = `
      <div class="card">
        <h2>💡 Respuesta del LLM</h2>
        <div class="result-box"><div class="answer-text">${d.answer}</div></div>
      </div>
      <div class="card">
        <h2>📎 Contexto recuperado (${d.chunks?.length || 0} chunks)</h2>
        ${renderChunks(d.chunks)}
      </div>`;
  } catch (e) {
    document.getElementById('rag-result').innerHTML = `<div class="error">Error al conectar con la API: ${e.message}</div>`;
  }
  document.getElementById('rag-loader').classList.remove('active');
}

async function sendSearch() {
  const q = document.getElementById('search-query').value.trim();
  if (!q) return;
  const body = { query: q, limit: parseInt(document.getElementById('search-limit').value) };
  const tipo = document.getElementById('search-tipo').value;
  const idioma = document.getElementById('search-idioma').value;
  if (tipo) body.tipo = tipo;
  if (idioma) body.idioma = idioma;

  document.getElementById('search-loader').classList.add('active');
  document.getElementById('search-result').innerHTML = '';

  try {
    const r = await fetch(API + '/search', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const d = await r.json();
    document.getElementById('search-result').innerHTML = `
      <div class="card">
        <h2>📄 ${d.total} resultados para: "${d.query}"</h2>
        ${renderChunks(d.resultados)}
      </div>`;
  } catch (e) {
    document.getElementById('search-result').innerHTML = `<div class="error">Error: ${e.message}</div>`;
  }
  document.getElementById('search-loader').classList.remove('active');
}

async function sendCompare() {
  const q = document.getElementById('compare-query').value.trim();
  if (!q) return;

  document.getElementById('compare-loader').classList.add('active');
  document.getElementById('compare-result').innerHTML = '';

  try {
    const r = await fetch(API + '/search', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ query: q, compare: true, limit: 3 }) });
    const d = await r.json();
    const c = d.comparacion;
    const strats = ['fixed', 'sentence', 'semantic'];
    const labels = { fixed: 'Fixed-size', sentence: 'Sentence-aware', semantic: 'Semantic' };

    let cols = strats.map(s => {
      const items = (c[s] || []).map(chunk =>
        `<div class="compare-item">${chunk.chunk_texto?.substring(0,200)}${chunk.chunk_texto?.length > 200 ? '...' : ''}<br><span style="font-size:0.75rem;color:#888;margin-top:4px;display:block">score: ${chunk.score?.toFixed(3) || 'N/A'}</span></div>`
      ).join('');
      return `<div class="compare-col col-${s}">
        <div class="compare-col-header">${labels[s]}<br><span style="font-weight:400;font-size:0.75rem">${(c[s]||[]).length} chunks</span></div>
        ${items || '<div class="compare-item" style="color:#aaa">Sin resultados</div>'}
      </div>`;
    }).join('');

    document.getElementById('compare-result').innerHTML = `
      <div class="card">
        <h2>📊 Comparación para: "${q}"</h2>
        <div class="compare-grid">${cols}</div>
      </div>`;
  } catch (e) {
    document.getElementById('compare-result').innerHTML = `<div class="error">Error: ${e.message}</div>`;
  }
  document.getElementById('compare-loader').classList.remove('active');
}

async function sendImageSearch() {
  const q = document.getElementById('img-query').value.trim();
  if (!q) return;

  document.getElementById('img-loader').classList.add('active');
  document.getElementById('img-result').innerHTML = '';

  try {
    const r = await fetch(API + '/search/images', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ query: q, limit: 6 }) });
    const d = await r.json();
    
    const items = (d.resultados || []).map(item => {
      const tipo = item.metadatos?.tipo_imagen || '';
      const url = item.metadatos?.url || '';
      const tieneUrl = url && url.length > 0;
      
      if (tieneUrl) {
        return `<div class="img-card">
          <img src="${url}" style="width:100%; height:160px; object-fit:cover; border-radius:8px; margin-bottom:10px;" 
                onerror="this.style.display='none'">
          <div class="img-tipo">${tipo}</div>
          <div class="img-desc">${item.chunk_texto.substring(0, 100)}${item.chunk_texto.length > 100 ? '...' : ''}</div>
          <div class="img-score">Relevancia: ${item.score?.toFixed(3) || 'N/A'} · ${item.metadatos?.aeropuerto || ''} · ${item.metadatos?.fecha || ''}</div>
        </div>`;
      } else {
        return `<div class="img-card">
          <div class="img-icon">${TIPO_ICONS[tipo] || '📁'}</div>
          <div class="img-tipo">${tipo}</div>
          <div class="img-desc">${item.chunk_texto.substring(0, 100)}${item.chunk_texto.length > 100 ? '...' : ''}</div>
          <div class="img-score">Relevancia: ${item.score?.toFixed(3) || 'N/A'} · ${item.metadatos?.aeropuerto || ''} · ${item.metadatos?.fecha || ''}</div>
        </div>`;
      }
    }).join('');

    document.getElementById('img-result').innerHTML = `
      <div class="card">
        <h2>🖼 ${d.total} imágenes encontradas</h2>
        <div class="imgs-grid">${items || '<p style="color:#888">Sin resultados</p>'}</div>
      </div>`;
  } catch (e) {
    document.getElementById('img-result').innerHTML = `<div class="error">Error: ${e.message}</div>`;
  }
  document.getElementById('img-loader').classList.remove('active');
}

// Event listeners
document.getElementById('rag-question').addEventListener('keydown', e => { if (e.ctrlKey && e.key === 'Enter') sendRag(); });
document.getElementById('search-query').addEventListener('keydown', e => { if (e.key === 'Enter') sendSearch(); });
document.getElementById('compare-query').addEventListener('keydown', e => { if (e.key === 'Enter') sendCompare(); });
document.getElementById('img-query').addEventListener('keydown', e => { if (e.key === 'Enter') sendImageSearch(); });

// Inicializar
checkHealth();

async function sendMultimodal() {

  const q = document.getElementById('multi-query').value.trim();

  if (!q) return;

  document.getElementById('multi-loader').classList.add('active');
  document.getElementById('multi-result').innerHTML = '';

  try {

    const r = await fetch(API + '/search/multimodal', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: q,
        limit_texto: parseInt(document.getElementById('multi-limit-text').value),
        limit_imagenes: parseInt(document.getElementById('multi-limit-img').value)
      })
    });

    const d = await r.json();

    // TEXTOS
    const textosHTML = renderChunks(d.textos || []);

    // IMÁGENES
    const imagenesHTML = (d.imagenes || []).map(item => {

      const tipo = item.metadatos?.tipo_imagen || '';
      const url  = item.metadatos?.url || '';

      return `
        <div class="img-card">

          ${
            url
              ? `<img src="${url}"
                    style="width:100%;height:160px;object-fit:cover;border-radius:8px;margin-bottom:10px;">`
              : `<div class="img-icon">${TIPO_ICONS[tipo] || '📁'}</div>`
          }

          <div class="img-tipo">${tipo}</div>

          <div class="img-desc">
            ${item.chunk_texto.substring(0,100)}
          </div>

          <div class="img-score">
            score: ${item.score?.toFixed(3) || 'N/A'}
          </div>

        </div>
      `;

    }).join('');

    document.getElementById('multi-result').innerHTML = `

      <div class="card">
        <h2>📄 Resultados texto</h2>
        ${textosHTML}
      </div>

      <div class="card">
        <h2>🖼 Resultados imágenes</h2>
        <div class="imgs-grid">
          ${imagenesHTML || '<p>Sin imágenes</p>'}
        </div>
      </div>

    `;

  } catch (e) {

    document.getElementById('multi-result').innerHTML = `
      <div class="error">
        Error: ${e.message}
      </div>
    `;

  }

  document.getElementById('multi-loader').classList.remove('active');
}

async function sendImageToImage() {

  const fileInput = document.getElementById('img2img-file');

  if (!fileInput.files.length) {
    alert('Selecciona una imagen');
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  document.getElementById('img2img-loader').classList.add('active');
  document.getElementById('img2img-result').innerHTML = '';

  try {

    const r = await fetch(API + '/search/image-to-image', {
      method: 'POST',
      body: formData
    });

    const d = await r.json();

    if (!d.resultados || d.resultados.length === 0) {
      document.getElementById('img2img-result').innerHTML = `
        <div class="card">
          <h2>🖼 Resultados de búsqueda</h2>
          <p>No se encontraron imágenes similares</p>
        </div>
      `;
      return;
    }

    const placeholderBase64 = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='160' viewBox='0 0 200 160'%3E%3Crect width='200' height='160' fill='%23e0e0e0'/%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23999' font-size='24'%3E🖼%3C/text%3E%3C/svg%3E";

    const items = d.resultados.map(item => {
      const filename = item.filename || '';
      const descripcion = item.descripcion || 'Sin descripción';
      const score = item.score || 0;
      const tipo_imagen = item.tipo_imagen || 'General';
      
      const safeSubstring = (str, maxLen) => {
        if (!str || typeof str !== 'string') return '';
        return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
      };

      // Usar STATIC_URL en lugar de URL relativa
      const imgUrl = `${STATIC_URL}/images/${encodeURIComponent(filename)}`;
      
      return `
        <div class="img-card">
          <img src="${imgUrl}" 
               style="width:100%; height:160px; object-fit:cover; border-radius:8px; margin-bottom:10px; background:#f0f0f0;"
               onerror="this.src='${placeholderBase64}'; this.onerror=null;">
          
          <div class="img-tipo">${safeSubstring(tipo_imagen, 30)}</div>
          
          <div class="img-desc">
            ${safeSubstring(descripcion, 100)}
          </div>
          
          <div class="img-score">
            🔍 Similaridad: ${(score * 100).toFixed(1)}%
          </div>
          
          ${filename ? `<div class="img-filename">📄 ${safeSubstring(filename, 40)}</div>` : ''}
        </div>
      `;
    }).join('');

    document.getElementById('img2img-result').innerHTML = `
      <div class="card">
        <h2>🖼 Imágenes similares encontradas (${d.total || d.resultados.length})</h2>
        <div class="imgs-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;">
          ${items}
        </div>
      </div>
    `;

  } catch (e) {
    console.error('Error en image-to-image:', e);
    document.getElementById('img2img-result').innerHTML = `
      <div class="error" style="padding: 20px; background: #ffebee; border-radius: 8px; color: #c62828;">
        ❌ Error: ${e.message}
      </div>
    `;
  }

  document.getElementById('img2img-loader').classList.remove('active');
}