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

## Supported Models

| Model | Type | Status |
|-------|------|--------|
| BERTurk | Encoder | ✅ Tested |
| YTÜ-Cosmos | Decoder | ⚠️ Pending |
| Kumru | Decoder | ⚠️ Pending |
| Aya-23 | Decoder | ⚠️ Pending |

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
@article{yagiz2024mafex,
  title={Beyond the Token: Correcting the Tokenization Bias in XAI via Morphologically-Aligned Projection},
  author={Yağız, Muhammet Anıl},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Zemberek](https://github.com/ahmetaa/zemberek-nlp) for Turkish morphological analysis
- [Captum](https://captum.ai/) for attribution methods
- [dbmdz/BERTurk](https://huggingface.co/dbmdz/bert-base-turkish-cased) for the Turkish BERT model
