#!/usr/bin/env python3
"""
evaluar_ragas.py
================
Script standalone para ejecutar la evaluación RAGAS completa desde la terminal.

Uso:
    cd src/
    python evaluar_ragas.py                        # evalúa los 20 pares, guarda en MongoDB
    python evaluar_ragas.py --no-mongo             # sin guardar en MongoDB
    python evaluar_ragas.py --subset incidente     # solo pares de tipo incidente
    python evaluar_ragas.py --show-historial       # muestra últimas evaluaciones de MongoDB

El script genera al final una tabla resumen con las 3 métricas RAGAS
desglosadas por tipo de reporte.
"""
import argparse
import sys
import os

# Asegurar que src/ está en el path
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)


def print_table(resumen: dict) -> None:
    """Imprime tabla de resultados en consola."""
    sep = "─" * 70

    print(f"\n{'='*70}")
    print(f"  EVALUACIÓN RAGAS — Sistema RAG Aeroportuario")
    print(f"{'='*70}")
    print(f"  Total pares evaluados : {resumen['total_evaluados']}")
    print(f"  Faithfulness          : {resumen['faithfulness_promedio']:.4f}  {'🟢' if resumen['faithfulness_promedio'] >= 0.7 else '🟡' if resumen['faithfulness_promedio'] >= 0.5 else '🔴'}")
    print(f"  Answer Relevancy      : {resumen['answer_relevancy_promedio']:.4f}  {'🟢' if resumen['answer_relevancy_promedio'] >= 0.7 else '🟡' if resumen['answer_relevancy_promedio'] >= 0.5 else '🔴'}")
    print(f"  Context Recall        : {resumen['context_recall_promedio']:.4f}  {'🟢' if resumen['context_recall_promedio'] >= 0.7 else '🟡' if resumen['context_recall_promedio'] >= 0.5 else '🔴'}")
    print(f"  {sep}")
    print(f"  SCORE GLOBAL          : {resumen['score_global']:.4f}")
    print(f"{'='*70}")

    print(f"\n  Desglose por tipo de reporte:")
    print(f"  {'Tipo':<16} {'N':>3}  {'Faith.':>8}  {'Relev.':>8}  {'Recall':>8}  {'Avg':>8}")
    print(f"  {'─'*16} {'─'*3}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for tipo, m in resumen["resultados_por_tipo"].items():
        print(
            f"  {tipo:<16} {m['n']:>3}  "
            f"{m['faithfulness']:>8.4f}  "
            f"{m['answer_relevancy']:>8.4f}  "
            f"{m['context_recall']:>8.4f}  "
            f"{m['score_promedio']:>8.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del sistema RAG aeroportuario")
    parser.add_argument("--no-mongo",       action="store_true", help="No guardar resultados en MongoDB")
    parser.add_argument("--subset",         type=str,  default=None, help="Filtrar por tipo: incidente|climatico|seguridad|mantenimiento|operativo")
    parser.add_argument("--show-historial", action="store_true", help="Mostrar últimas evaluaciones de MongoDB y salir")
    parser.add_argument("--verbose",        action="store_true", default=True, help="Mostrar progreso por par")
    args = parser.parse_args()

    from evaluacion.ragas_eval  import run_evaluation, get_evaluation_history
    from evaluacion.dataset     import EVAL_DATASET

    if args.show_historial:
        print("\n  Últimas evaluaciones guardadas en MongoDB:\n")
        docs = get_evaluation_history(limit=10)
        if not docs or "error" in docs[0]:
            print("  No hay evaluaciones guardadas o error de conexión.")
        else:
            for d in docs:
                print(
                    f"  [{d.get('fecha','?')[:19]}] {d['id_consulta']:<12} "
                    f"F={d['faithfulness']:.3f} R={d['answer_relevancy']:.3f} "
                    f"C={d['context_recall']:.3f} | avg={d['score_promedio']:.3f}"
                )
        return

    dataset = EVAL_DATASET
    if args.subset:
        dataset = [d for d in dataset if d["tipo"] == args.subset]
        if not dataset:
            print(f"❌ No se encontraron pares con tipo='{args.subset}'.")
            print(f"   Tipos disponibles: incidente, climatico, seguridad, mantenimiento, operativo")
            sys.exit(1)
        print(f"ℹ️  Evaluando subconjunto '{args.subset}': {len(dataset)} pares.")

    print(f"\n🚀 Iniciando evaluación RAGAS ({len(dataset)} pares)...")

    resumen = run_evaluation(
        dataset       = dataset,
        save_to_mongo = not args.no_mongo,
        verbose       = args.verbose,
    )

    print_table(resumen)

    if not args.no_mongo:
        print("  ✅ Resultados guardados en colección 'evaluaciones' de MongoDB.")
    else:
        print("  ℹ️  Modo --no-mongo: resultados NO guardados en MongoDB.")


if __name__ == "__main__":
    main()
