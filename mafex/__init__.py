"""
MAFEX - Morpheme-Aligned Faithful Explanations

A framework for faithful interpretability in Morphologically Rich Languages (MRLs).
Corrects the Tokenization-Morphology Misalignment (TMM) via morphological projection.

Paper: "Beyond the Token: Correcting the Tokenization Bias in XAI via 
        Morphologically-Aligned Projection"
Author: Muhammet Anıl Yağız
"""

__version__ = "0.1.1"
__author__ = "Muhammet Anıl Yağız"

# Core morphology - always available
from .morphology import (
    MorphemeAnalyzer,
    AlignmentMatrixBuilder,
    MorphologicalProjector
)

# Lazy imports for torch-dependent modules
def __getattr__(name):
    """Lazy import for modules requiring torch."""
    if name in ("IntegratedGradients", "SHAPAttributor", "RandomGroupingBaseline"):
        from .attribution import IntegratedGradients, SHAPAttributor, RandomGroupingBaseline
        return locals()[name]
    elif name in ("MAFEXPipeline", "CausalRegularizer"):
        from .projection import MAFEXPipeline, CausalRegularizer
        return locals()[name]
    elif name in ("ModelWrapper", "get_model"):
        from .models import ModelWrapper, get_model
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "MorphemeAnalyzer",
    "AlignmentMatrixBuilder", 
    "MorphologicalProjector",
    "IntegratedGradients",
    "SHAPAttributor",
    "RandomGroupingBaseline",
    "MAFEXPipeline",
    "CausalRegularizer",
    "ModelWrapper",
    "get_model",
]
