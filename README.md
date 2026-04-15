# MAFEX - Morpheme-Aligned Faithful Explanations

> **Beyond the Token: Correcting the Tokenization Bias in XAI via Morphologically-Aligned Projection**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

MAFEX is a framework for generating faithful, interpretable explanations for Large Language Models (LLMs) in **Morphologically Rich Languages (MRLs)** like Turkish. 

Current XAI methods operate on tokens, which fragment semantic units in agglutinative languages. MAFEX corrects this **Tokenization-Morphology Misalignment (TMM)** by projecting attributions onto linguistically meaningful morphemes.

### Key Equation

```
φ_morph = A · φ_tok           (Morphological Projection)
S* = λ·φ_morph + (1-λ)·φ_causal  (Causal Regularization)
```

Where:
- **A** ∈ {0,1}^{K×T} is the Alignment Matrix mapping T tokens to K morphemes
- **φ_tok** is token-level attribution (e.g., Integrated Gradients)
- **λ** controls the gradient/causal trade-off (default: 0.7)

## Installation

```bash
# Install from PyPI
pip install mafex

# Or install latest from GitHub
pip install git+https://github.com/anilyagiz/mafex.git
```

## Quick Start

### 1. Morphological Analysis

```python
from mafex.morphology import MorphemeAnalyzer

analyzer = MorphemeAnalyzer()

# Analyze Turkish word
analysis = analyzer.analyze_word("gelemedim")  # "I could not come"
print(analysis.morpheme_surfaces)  # ['gel', 'eme', 'di', 'm']
```

### 2. Run MAFEX Explanation

```python
from mafex.models import DemoModelWrapper
from mafex.projection import MAFEXPipeline

# Load model
wrapper = DemoModelWrapper()
wrapper.load()

# Create MAFEX pipeline
mafex = MAFEXPipeline(
    wrapper.model,
    wrapper.tokenizer,
    lambda_causal=0.7
)

# Generate explanation
result = mafex.explain("Gelemedim")

# Get top attributed morphemes
print(result.get_top_morphemes(3))
# [('-eme', 0.62), ('gel', 0.21), ('-di', 0.12)]
```

### 3. Command Line

```bash
# Single explanation
python run_mafex.py --model demo --text "Gelemedim"

# Evaluation
python run_mafex.py --model berturk --eval --samples 10
```

## Project Structure

```
mafex/
├── mafex/
│   ├── __init__.py         # Package exports
│   ├── morphology.py       # Morphological analysis & alignment
│   ├── attribution.py      # IG, SHAP, DeepLIFT baselines
│   ├── projection.py       # MAFEX pipeline & causal regularization
│   ├── models.py           # Model wrappers (BERTurk, Cosmos, etc.)
│   └── visualization.py    # Plotting utilities
├── evaluation/
│   ├── __init__.py
│   └── metrics.py          # ERASER metrics
├── benchmark/
│   ├── __init__.py
│   └── trust_tr.py         # Trust-TR benchmark (N=850)
├── notebooks/
│   └── demo.ipynb          # Interactive demonstration
├── demo.py                 # CLI demo script
├── run_mafex.py           # Main runner
├── config.yaml            # Configuration
└── requirements.txt       # Dependencies
```

## Evaluation Metrics

MAFEX is evaluated using ERASER metrics:

- **Comprehensiveness**: Does removing important features hurt performance?
- **Sufficiency**: Are important features alone enough to maintain performance?

Expected results:

| Model | Token-IG | Random | MAFEX | Δ |
|-------|----------|--------|-------|---|
| BERTurk | 0.42 | 0.50 | 0.68 | +62% |
| Cosmos | 0.39 | 0.47 | 0.65 | +67% |
| Kumru | 0.45 | 0.53 | 0.71 | +58% |
| Aya-23 | 0.41 | 0.49 | 0.69 | +68% |

## Citation

```bibtex
@inproceedings{yagiz-horasan-2026-beyond,
    title = "Beyond the Token: Correcting the Tokenization Bias in {XAI} via Morphologically-Aligned Projection",
    author = "Yagiz, Muhammet Anil  and
      Horasan, Fahrettin",
    editor = {Oflazer, Kemal  and
      K{\"o}ksal, Abdullatif  and
      Varol, Onur},
    booktitle = "Proceedings of the Second Workshop Natural Language Processing for {T}urkic Languages ({SIGTURK} 2026)",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.sigturk-1.19/",
    doi = "10.18653/v1/2026.sigturk-1.19",
    pages = "228--235",
    ISBN = "979-8-89176-370-8",
    abstract = "Current interpretability methods for Large Language Models (LLMs) operate on a fundamental yet flawed assumption: that subword tokens represent independent semantic units. We prove that this assumption creates a fidelity bottleneck in Morphologically Rich Languages (MRLs), where semantic meaning is densely encoded in sub-token morphemes. We term this phenomenon the Tokenization-Morphology Misalignment (TMM). To resolve TMM, we introduce MAFEX (Morpheme-Aligned Faithful Explanations), a theoretically grounded framework that redefines feature attribution as a linear projection from the computational (token) basis to the linguistic (morpheme) basis. We evaluate our method on a diverse suite of Turkish LLMs, including BERTurk, BERTurk-Sentiment, Cosmos-BERT, and Kumru-2B. On our embedded benchmark (N=20), MAFEX achieves an average F1@1 of 91.25{\\%} compared to 13.75{\\%} for standard token-level baselines (IG, SHAP, DeepLIFT), representing a +77.5{\\%} absolute improvement, establishing it as the new standard for faithful multilingual interpretability."
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Zemberek](https://github.com/ahmetaa/zemberek-nlp) for Turkish morphological analysis
- [Captum](https://captum.ai/) for attribution methods
- [dbmdz/BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) for the Turkish BERT model
