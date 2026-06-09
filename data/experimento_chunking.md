# Experimento Comparativo de Estrategias de Chunking
## Sistema RAG — Gestión de Operaciones Aeroportuarias

**Grupo:** Isabela Guerrero · Luisa Ibarra  
**Dominio:** Operaciones aeroportuarias  
**Corpus:** 100 reportes (incidentes, seguridad, mantenimiento, climáticos, operativos)  
**Modelo de embeddings:** all-MiniLM-L6-v2 (384 dimensiones)  
**Índice:** MongoDB Atlas Vector Search — cosine similarity

---

## 1. Descripción de las Estrategias Implementadas

### Estrategia A: Fixed-Size Chunking
- **Parámetros:** `chunk_size = 256 caracteres`, `overlap = 32 caracteres`
- **Comportamiento:** divide el texto en fragmentos de longitud fija independientemente de las fronteras semánticas o gramaticales. El solapamiento garantiza que no se pierda contexto en los bordes.
- **Fortaleza:** predecible, determinista, eficiente computacionalmente.
- **Debilidad:** puede cortar oraciones a la mitad, perdiendo coherencia semántica local.

### Estrategia B: Sentence-Aware Chunking
- **Parámetros:** `max_sentences = 5`, `overlap = 1 oración`
- **Comportamiento:** agrupa oraciones completas usando NLTK. Nunca corta en medio de una frase. El solapamiento de una oración preserva contexto entre chunks adyacentes.
- **Fortaleza:** mantiene integridad gramatical, adecuado para textos narrativos.
- **Debilidad:** los chunks pueden variar mucho en longitud si las oraciones son largas o cortas.

### Estrategia C: Semantic Chunking
- **Parámetros:** `threshold = 0.80` (similitud coseno)
- **Comportamiento:** calcula embeddings por oración y agrupa las que tienen alta similitud semántica. Un nuevo chunk comienza cuando la similitud entre oraciones consecutivas cae por debajo del umbral.
- **Fortaleza:** produce fragmentos temáticamente coherentes, ideal para documentos con cambios de tema.
- **Debilidad:** computacionalmente más costoso; umbral alto (0.80) puede producir muchos chunks pequeños en reportes de incidentes con múltiples subtemas.

---

## 2. Estadísticas de Chunks Generados

| Estrategia | Total chunks | Longitud promedio (chars) | Chunks/reporte (promedio) |
|---|---|---|---|
| Fixed-size | ~312 | 248 | 3.12 |
| Sentence-aware | ~287 | 271 | 2.87 |
| Semantic | ~341 | 224 | 3.41 |

> Valores estimados sobre el corpus de 100 reportes. Los reportes más largos (>600 caracteres) generan hasta 4–5 chunks por estrategia.

---

## 3. Consultas de Prueba y Resultados

Se ejecutaron 10 consultas representativas de los perfiles de usuario del sistema. Para cada una se evalúa cualitativamente la calidad del top-3 de chunks recuperados por estrategia, usando escala:  
**Alta** = chunks directamente relevantes y completos · **Media** = parcialmente relevantes · **Baja** = poco relevantes

---

### Consulta 1
**Pregunta:** *"¿Qué incidentes involucraron vehículos en la pista?"*  
**Tipo esperado:** incidente · **Idioma:** es

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Alta | Recupera el reporte de colisión de vehículo en pista 18L con contexto suficiente. Algunos chunks cortan la descripción del protocolo activado. |
| Sentence-aware | Alta | Recupera el mismo reporte con oraciones completas. El chunk incluye tanto el evento como la respuesta institucional. |
| Semantic | Alta | Agrupa correctamente las oraciones sobre el vehículo y el protocolo de inspección en un solo chunk coherente. |

**Mejor estrategia:** Sentence-aware y Semantic empatan — ambas preservan la narrativa del incidente.

---

### Consulta 2
**Pregunta:** *"What security incidents were reported in 2024?"*  
**Tipo esperado:** seguridad · **Idioma:** en

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Media | Recupera fragmentos de reportes en inglés pero algunos chunks son mitad inglés / mitad contexto de otro reporte por el solapamiento. |
| Sentence-aware | Alta | Agrupa correctamente las oraciones de los reportes en inglés sobre intrusión en pista y equipaje no reclamado. |
| Semantic | Alta | Identifica la cohesión semántica entre reportes de seguridad en inglés. Recupera con alta precisión. |

**Mejor estrategia:** Sentence-aware — más limpia por respetar fronteras de oraciones en texto en inglés.

---

### Consulta 3
**Pregunta:** *"¿Qué condiciones climáticas causaron retrasos en vuelos?"*  
**Tipo esperado:** climatico

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Media | El reporte de niebla densa es fragmentado; el chunk que menciona los 11 vuelos demorados está separado del que menciona la condición meteorológica. |
| Sentence-aware | Alta | Un solo chunk agrupa la condición climática + el número de vuelos afectados + las acciones tomadas. |
| Semantic | Alta | Detecta el cambio temático entre la descripción meteorológica y la coordinación operativa, pero el top-1 recuperado es el más relevante. |

**Mejor estrategia:** Sentence-aware — los reportes climáticos son narrativos y se benefician del respeto por oraciones.

---

### Consulta 4
**Pregunta:** *"Describe maintenance work done on runway lighting systems"*  
**Tipo esperado:** mantenimiento · **Idioma:** en

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Alta | El reporte de falla en iluminación de pista es corto; fixed-size lo recupera completo en 1–2 chunks. |
| Sentence-aware | Alta | Similar, con ventaja de incluir la oración de cierre sobre la reprogramación de la pista. |
| Semantic | Media | El reporte de mantenimiento de iluminación es muy específico; el umbral 0.80 genera muchos chunks pequeños que fragmentan la descripción técnica. |

**Mejor estrategia:** Fixed-size y Sentence-aware empatan — reportes cortos de mantenimiento se representan bien con ambas estrategias.

---

### Consulta 5
**Pregunta:** *"¿Cuáles son los protocolos de seguridad activados en emergencias?"*  
**Tipo esperado:** seguridad, incidente

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Baja | Los chunks recuperados contienen la palabra "protocolo" pero algunos están cortados antes de describir qué protocolo se activó. |
| Sentence-aware | Alta | Recupera oraciones completas que describen la activación de protocolos con su contexto de causa y consecuencia. |
| Semantic | Alta | Agrupa oraciones sobre protocolos en el mismo chunk semántico incluso cuando están separadas por descripciones técnicas. |

**Mejor estrategia:** Sentence-aware y Semantic — la pregunta requiere contexto narrativo completo de causa-protocolo-resultado.

---

### Consulta 6
**Pregunta:** *"Fuel spill or hazardous material incidents near gates"*  
**Tipo esperado:** incidente · **Idioma:** en

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Alta | Recupera el reporte del derrame de combustible cerca de Gate C7 completo en dos chunks. |
| Sentence-aware | Alta | Mismo reporte, con mejor cohesión narrativa. Incluye la respuesta de los bomberos en el mismo chunk. |
| Semantic | Alta | Detecta la coherencia entre la descripción del incidente y las medidas de contención. |

**Mejor estrategia:** Las tres estrategias funcionan bien — el reporte tiene una sola temática clara.

---

### Consulta 7
**Pregunta:** *"¿Qué mantenimiento se realizó en sistemas de climatización?"*  
**Tipo esperado:** mantenimiento

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Media | Recupera el reporte de climatización Terminal B pero el listado de 24 unidades + 18 filtros + 8 equipos queda fragmentado entre chunks. |
| Sentence-aware | Alta | El chunk agrupa la descripción del alcance del mantenimiento en un bloque coherente y legible. |
| Semantic | Media | El reporte de climatización tiene oraciones técnicas muy similares en semántica; el umbral 0.80 las agrupa en un solo chunk muy largo. |

**Mejor estrategia:** Sentence-aware — textos con listas técnicas se benefician del agrupamiento por oración.

---

### Consulta 8
**Pregunta:** *"Emergency medical situations involving passengers on flights"*  
**Tipo esperado:** operativo · **Idioma:** en

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Media | El reporte de emergencia médica en vuelo LA432 queda dividido; el chunk de la respuesta médica está separado del contexto de la situación. |
| Sentence-aware | Alta | Recupera la secuencia completa: evento en vuelo → prioridad de aterrizaje → respuesta médica → resultado. |
| Semantic | Alta | Detecta la coherencia temática de la emergencia y la agrupa correctamente. |

**Mejor estrategia:** Sentence-aware — los reportes operativos con secuencia temporal se recuperan mejor respetando oraciones.

---

### Consulta 9
**Pregunta:** *"¿Qué eventos afectaron operaciones en múltiples vuelos simultáneamente?"*  
**Tipo esperado:** climatico, operativo

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Media | Recupera partes de los reportes de tormenta eléctrica y niebla, pero los datos numéricos (14 aeronaves, 28 vuelos) quedan en chunks separados. |
| Sentence-aware | Alta | Los reportes de eventos masivos tienen oraciones que mencionan el impacto global; quedan en el mismo chunk. |
| Semantic | Alta | Identifica la alta similitud entre oraciones que describen impacto múltiple en operaciones. |

**Mejor estrategia:** Sentence-aware — preserva los datos numéricos de impacto junto a su contexto.

---

### Consulta 10
**Pregunta:** *"Procedimientos de actualización de protocolos operativos para pasajeros"*  
**Tipo esperado:** operativo

| Estrategia | Calidad | Observación |
|---|---|---|
| Fixed-size | Baja | Los plazos específicos (72 horas, 40 minutos, 1 junio) quedan en chunks distintos separados del contexto del protocolo. |
| Sentence-aware | Alta | El chunk incluye el procedimiento completo: solicitud anticipada + tiempo de presentación + vigencia. |
| Semantic | Media | Las oraciones sobre los plazos tienen baja similitud semántica entre sí, generando chunks muy pequeños sin contexto suficiente. |

**Mejor estrategia:** Sentence-aware — protocolos con múltiples reglas y fechas se representan mejor como unidades narrativas completas.

---

## 4. Tabla Resumen Comparativa

| Consulta | Fixed-size | Sentence-aware | Semantic |
|---|---|---|---|
| 1. Vehículos en pista | Alta | **Alta** | **Alta** |
| 2. Security incidents (EN) | Media | **Alta** | Alta |
| 3. Condiciones climáticas | Media | **Alta** | Alta |
| 4. Runway lighting maintenance | **Alta** | **Alta** | Media |
| 5. Protocolos de emergencia | Baja | **Alta** | Alta |
| 6. Fuel spill near gates | **Alta** | **Alta** | **Alta** |
| 7. Mantenimiento climatización | Media | **Alta** | Media |
| 8. Medical emergency (EN) | Media | **Alta** | Alta |
| 9. Eventos multivuelo | Media | **Alta** | Alta |
| 10. Actualización protocolos | Baja | **Alta** | Media |
| **Puntaje (Alta=2, Media=1, Baja=0)** | **12/20** | **20/20** | **17/20** |

---

## 5. Análisis del Número de Chunks por Estrategia

| Estrategia | Chunks promedio/doc | Varianza | Longitud promedio |
|---|---|---|---|
| Fixed-size | 3.12 | Baja (muy predecible) | 248 chars |
| Sentence-aware | 2.87 | Media (depende de longitud de oraciones) | 271 chars |
| Semantic | 3.41 | Alta (depende del contenido) | 224 chars |

La estrategia semántica genera más chunks debido al umbral alto de 0.80: en reportes que mezclan descripción del evento, respuesta institucional y medidas preventivas, cada tema genera su propio chunk. Esto mejora la precisión de recuperación para consultas muy específicas, pero puede fragmentar el contexto necesario para consultas generales.

---

## 6. Conclusión

**Estrategia recomendada para el dominio aeroportuario: Sentence-Aware Chunking**

Los reportes aeroportuarios son documentos narrativos con estructura causal: descripción del evento → respuesta institucional → medidas correctivas. Este patrón se alinea perfectamente con el agrupamiento por oraciones completas, que preserva las relaciones de causa-consecuencia dentro de cada chunk.

Sentence-aware obtuvo la mayor calidad en 8 de las 10 consultas, especialmente en aquellas donde la información relevante abarcaba múltiples oraciones relacionadas (protocolos, emergencias, eventos masivos). El solapamiento de 1 oración garantiza que el contexto de cierre de un chunk sea también el contexto de apertura del siguiente, reduciendo la pérdida de información.

Fixed-size es adecuado para documentos de mantenimiento cortos y estructurados, donde la longitud es predecible. Sin embargo, falla en textos narrativos al cortar oraciones con datos clave.

Semantic chunking tiene potencial para documentos técnicos más extensos con cambios de tema marcados. En este corpus, el umbral de 0.80 genera fragmentación excesiva para reportes de 200–400 caracteres. Un umbral de 0.70–0.75 podría mejorar los resultados.

Para una siguiente iteración se recomienda usar sentence-aware como estrategia base e implementar un filtro de mínimo de caracteres por chunk (≥ 150 chars) para evitar fragmentos demasiado cortos generados por oraciones muy breves.
