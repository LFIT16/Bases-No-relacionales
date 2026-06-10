# evaluacion/__init__.py
from .ragas_eval import run_evaluation, get_evaluation_history
from .dataset import EVAL_DATASET

__all__ = ["run_evaluation", "get_evaluation_history", "EVAL_DATASET"]
