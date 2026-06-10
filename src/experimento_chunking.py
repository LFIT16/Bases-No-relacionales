"""
experimento_chunking.py — Experimento obligatorio de comparación de estrategias.

Ejecuta las 10 consultas de prueba sobre las 3 estrategias de chunking,
calcula métricas y genera una tabla comparativa en consola y en JSON.

Uso:
    python experimento_chunking.py
    python experimento_chunking.py --output resultados_chunking.json
    python experimento_chunking.py --entidad reportes
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC

import config
from rag.retriever import compare_strategies, get_stats
from chunking import STRATEGIES

# ─────────────────────────────────────────────────────────────────────────────
# 10 consultas de prueba obligatorias (dominio aeroportuario)
# ─────────────────────────────────────────────────────────────────────────────

CONSULTAS_PRUEBA = [
    # Texto narrativo
    "¿Qué incidentes de seguridad se han reportado en el aeropuerto?",
    "Describe los problemas climáticos que afectaron las operaciones de vuelo.",
    # Datos operativos
    "¿Qué vuelos están cancelados o con retrasos significativos?",
    "¿Qué empleados tienen turno nocturno en la terminal principal?",
    # Equipaje
    "¿Cuáles son los casos de equipaje dañado o perdido reportados?",
    # Pasajeros
    "¿Qué pasajeros tienen categoría de viajero frecuente oro o platino?",
    # Torre de control
    "¿Cuáles son las condiciones meteorológicas reportadas por torre de control?",
    # Mantenimiento
    "¿Qué reportes de mantenimiento indican problemas críticos en infraestructura?",
    # Multidominio
    "¿Qué incidentes operativos ocurrieron en las últimas semanas?",
    # Consulta técnica
    "¿Cuáles son los procedimientos de emergencia registrados en los reportes?",
]


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución del experimento
# ─────────────────────────────────────────────────────────────────────────────

def calcular_metricas(chunks: list[dict]) -> dict:
    """Calcula métricas de calidad para un conjunto de chunks."""
    if not chunks:
        return {
            "num_chunks":        0,
            "score_promedio":    0.0,
            "score_maximo":      0.0,
            "longitud_promedio": 0.0,
            "longitud_total":    0,
        }
    scores     = [c.get("score", 0) for c in chunks]
    longitudes = [len(c.get("chunk_texto", "")) for c in chunks]
    return {
        "num_chunks":        len(chunks),
        "score_promedio":    round(sum(scores) / len(scores), 4),
        "score_maximo":      round(max(scores), 4),
        "longitud_promedio": round(sum(longitudes) / len(longitudes), 1),
        "longitud_total":    sum(longitudes),
    }


def ejecutar_experimento(
    consultas: list[str] = CONSULTAS_PRUEBA,
    limit: int = 5,
    tipo_fuente: str | None = None,
) -> dict:
    """
    Ejecuta todas las consultas con las 3 estrategias y retorna resultados.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENTO DE COMPARACIÓN DE ESTRATEGIAS DE CHUNKING")
    print("=" * 60)
    print(f"  Consultas: {len(consultas)}  |  Estrategias: {STRATEGIES}")
    print(f"  Límite por consulta: {limit}")
    if tipo_fuente:
        print(f"  Filtro de entidad: {tipo_fuente}")
    print()

    resultados_por_consulta = []

    for i, consulta in enumerate(consultas, 1):
        print(f"[{i:2d}/{len(consultas)}] {consulta[:65]}…" if len(consulta) > 65 else f"[{i:2d}/{len(consultas)}] {consulta}")

        comparacion = compare_strategies(consulta, limit=limit, tipo_fuente=tipo_fuente)

        fila = {
            "consulta": consulta,
            "estrategias": {},
        }
        for estrategia in STRATEGIES:
            chunks = comparacion.get(estrategia, [])
            metricas = calcular_metricas(chunks)
            fila["estrategias"][estrategia] = metricas

            print(
                f"    {estrategia:<10} → "
                f"chunks: {metricas['num_chunks']} | "
                f"score_max: {metricas['score_maximo']:.4f} | "
                f"score_avg: {metricas['score_promedio']:.4f} | "
                f"long_avg: {metricas['longitud_promedio']:.0f} chars"
            )
        print()
        resultados_por_consulta.append(fila)

    # ── Resumen agregado ──────────────────────────────────────────────────────
    print("=" * 60)
    print("  RESUMEN AGREGADO POR ESTRATEGIA")
    print("=" * 60)

    resumen_global = {}
    for estrategia in STRATEGIES:
        scores_max = []
        scores_avg = []
        longs_avg  = []
        for fila in resultados_por_consulta:
            m = fila["estrategias"].get(estrategia, {})
            if m.get("num_chunks", 0) > 0:
                scores_max.append(m["score_maximo"])
                scores_avg.append(m["score_promedio"])
                longs_avg.append(m["longitud_promedio"])

        if scores_max:
            resumen_global[estrategia] = {
                "score_maximo_global":   round(sum(scores_max) / len(scores_max), 4),
                "score_promedio_global": round(sum(scores_avg) / len(scores_avg), 4),
                "longitud_promedio":     round(sum(longs_avg) / len(longs_avg), 1),
                "consultas_con_resultados": len(scores_max),
            }
        else:
            resumen_global[estrategia] = {
                "score_maximo_global":   0.0,
                "score_promedio_global": 0.0,
                "longitud_promedio":     0.0,
                "consultas_con_resultados": 0,
            }

        r = resumen_global[estrategia]
        print(f"\n  {estrategia.upper()}")
        print(f"    Score máximo promedio:    {r['score_maximo_global']:.4f}")
        print(f"    Score promedio global:    {r['score_promedio_global']:.4f}")
        print(f"    Longitud promedio chunks: {r['longitud_promedio']:.1f} chars")
        print(f"    Consultas con resultados: {r['consultas_con_resultados']}/{len(consultas)}")

    # ── Determinar mejor estrategia ───────────────────────────────────────────
    mejor = max(resumen_global, key=lambda s: resumen_global[s]["score_maximo_global"])
    print(f"\n  🏆 Estrategia con mayor relevancia: {mejor.upper()}")

    _imprimir_tabla_markdown(resumen_global)

    return {
        "fecha":                datetime.now(UTC).isoformat(),
        "consultas_usadas":     consultas,
        "limit_por_consulta":   limit,
        "tipo_fuente_filtrado": tipo_fuente,
        "resultados_detallados": resultados_por_consulta,
        "resumen_global":       resumen_global,
        "mejor_estrategia":     mejor,
        "conclusion": _generar_conclusion(resumen_global, mejor),
    }


def _imprimir_tabla_markdown(resumen: dict) -> None:
    """Imprime la tabla comparativa en formato Markdown para el informe."""
    print("\n\n" + "=" * 60)
    print("  TABLA COMPARATIVA — PARA EL INFORME (Markdown)")
    print("=" * 60)
    print()
    print("| Estrategia | Score Máx Prom | Score Avg Prom | Long. Prom (chars) | Consultas con resultados |")
    print("|------------|---------------|----------------|---------------------|--------------------------|")
    for estrategia, r in resumen.items():
        print(
            f"| {estrategia:<10} | "
            f"{r['score_maximo_global']:>13.4f} | "
            f"{r['score_promedio_global']:>14.4f} | "
            f"{r['longitud_promedio']:>19.1f} | "
            f"{r['consultas_con_resultados']:>24} |"
        )
    print()


def _generar_conclusion(resumen: dict, mejor: str) -> str:
    """Genera la conclusión argumentada para el informe."""
    r = resumen[mejor]
    otras = [s for s in STRATEGIES if s != mejor]

    conclusiones = {
        "fixed": (
            "La estrategia fixed-size resultó más efectiva para el dominio aeroportuario. "
            "Esto se explica porque los documentos del sistema (reportes operativos, registros "
            "de vuelos) tienen una densidad informativa homogénea, donde fragmentos de tamaño "
            "fijo capturan unidades semánticas completas sin desperdiciar contexto."
        ),
        "sentence": (
            "La estrategia sentence-aware resultó más efectiva para el dominio aeroportuario. "
            "Los documentos del sistema contienen reportes narrativos de incidentes, observaciones "
            "de torre de control y descripciones de equipaje, donde respetar los límites de "
            "oraciones preserva la coherencia semántica de cada afirmación."
        ),
        "semantic": (
            "La estrategia semantic chunking resultó más efectiva para el dominio aeroportuario. "
            "Los reportes aeroportuarios mezclan múltiples temas (clima, seguridad, operaciones) "
            "dentro del mismo documento. El chunking semántico detecta estos cambios temáticos "
            "y agrupa información coherente, mejorando la precisión de recuperación."
        ),
    }

    base = conclusiones.get(mejor, f"La estrategia {mejor} resultó más efectiva.")
    return (
        f"{base} "
        f"Con un score máximo promedio de {r['score_maximo_global']:.4f} y cobertura de "
        f"{r['consultas_con_resultados']} de {len(STRATEGIES) + len(otras)} consultas, "
        f"supera a {' y '.join(otras)} en relevancia para las consultas del dominio."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Conteo de chunks por estrategia (para el análisis de cantidad)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_chunks_en_bd() -> None:
    """Muestra cuántos chunks hay por estrategia y tipo_fuente en la BD."""
    print("\n" + "=" * 60)
    print("  ANÁLISIS DE CHUNKS EN BASE DE DATOS")
    print("=" * 60)

    pipeline = [
        {
            "$group": {
                "_id": {
                    "estrategia": "$estrategia_chunking",
                    "tipo":       "$tipo_fuente",
                },
                "total":           {"$sum": 1},
                "long_prom":       {"$avg": {"$strLenCP": "$chunk_texto"}},
            }
        },
        {"$sort": {"_id.estrategia": 1, "_id.tipo": 1}},
    ]

    rows = list(config.embeddings_texto.aggregate(pipeline))

    if not rows:
        print("  ⚠️  No hay chunks en embeddings_texto. Ejecuta la ingesta primero.")
        return

    print(f"\n  {'Estrategia':<12} {'Entidad':<18} {'# Chunks':>9} {'Long. Prom':>12}")
    print(f"  {'-'*12} {'-'*18} {'-'*9} {'-'*12}")
    for r in rows:
        est  = r["_id"]["estrategia"] or "N/A"
        tipo = r["_id"]["tipo"] or "N/A"
        tot  = r["total"]
        lp   = round(r["long_prom"], 1)
        print(f"  {est:<12} {tipo:<18} {tot:>9} {lp:>12.1f}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento de comparación de chunking")
    parser.add_argument("--output",   metavar="FILE", help="Guardar resultados en JSON")
    parser.add_argument("--limit",    type=int, default=5, help="Chunks por consulta (default: 5)")
    parser.add_argument("--entidad",  default=None,
                        help="Filtrar por entidad (reportes, vuelos, pasajeros, etc.)")
    args = parser.parse_args()

    if not config.ping():
        print("❌ No se pudo conectar a MongoDB. Revisa MONGO_URI.")
        exit(1)

    # Mostrar estado de la BD
    analisis_chunks_en_bd()

    # Ejecutar experimento
    resultados = ejecutar_experimento(
        consultas   = CONSULTAS_PRUEBA,
        limit       = args.limit,
        tipo_fuente = args.entidad,
    )

    # Guardar JSON si se pidió
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Resultados guardados en: {args.output}")

    print("\n✅ Experimento completado.")
    print(f"   Mejor estrategia: {resultados['mejor_estrategia'].upper()}")
    print(f"   Conclusión: {resultados['conclusion'][:120]}…")
