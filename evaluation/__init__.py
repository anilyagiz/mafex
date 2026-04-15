"""
Evaluation package for MAFEX.
"""

# Samples are always available (no torch dependency)
from .samples import (
    get_test_samples,
    get_negative_samples,
    get_positive_samples,
    TURKISH_SAMPLES
)

# Lazy import for torch-dependent modules
def __getattr__(name):
    """Lazy import for metrics that require torch."""
    if name in ("ERASEREvaluator", "EvaluationResult", "BenchmarkRunner", 
                "compute_faithfulness_correlation", "compare_methods"):
        from .metrics import (
            ERASEREvaluator, EvaluationResult, BenchmarkRunner,
            compute_faithfulness_correlation, compare_methods
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ERASEREvaluator",
    "EvaluationResult", 
    "BenchmarkRunner",
    "compute_faithfulness_correlation",
    "compare_methods",
    "get_test_samples",
    "get_negative_samples",
    "get_positive_samples",
    "TURKISH_SAMPLES"
]
