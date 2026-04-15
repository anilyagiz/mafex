"""
MAFEX Benchmark Module

Contains embedded Turkish test samples for evaluation.
No external dataset files needed - all samples are inline.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class TestSample:
    """A single test sample for benchmark."""
    text: str
    label: int
    label_name: str
    expected_morphemes: List[str]
    key_morpheme: str
    reason: str


# Embedded Turkish test samples for benchmarking
# Optimized for clear morpheme boundaries and strong semantic signal
TURKISH_SAMPLES: List[TestSample] = [
    # === STRONG NEGATION SAMPLES (clear -me/-ma patterns) ===
    TestSample(
        text="Gelemedim",
        label=0,
        label_name="negative",
        expected_morphemes=["gel", "emed", "i", "m"],
        key_morpheme="emed",
        reason="Inability marker (-emed) drives negative"
    ),
    TestSample(
        text="Yapamadık",
        label=0,
        label_name="negative",
        expected_morphemes=["yap", "amad", "ı", "k"],
        key_morpheme="amad",
        reason="Inability marker (-amad) indicates failure"
    ),
    TestSample(
        text="Anlamadım",
        label=0,
        label_name="negative",
        expected_morphemes=["anla", "mad", "ı", "m"],
        key_morpheme="mad",
        reason="Negation suffix (-mad) negates understanding"
    ),
    TestSample(
        text="İstemedim",
        label=0,
        label_name="negative",
        expected_morphemes=["iste", "med", "i", "m"],
        key_morpheme="med",
        reason="Negation on wanting verb"
    ),
    TestSample(
        text="Görmediler",
        label=0,
        label_name="negative",
        expected_morphemes=["gör", "med", "i", "ler"],
        key_morpheme="med",
        reason="Negation with plural"
    ),
    TestSample(
        text="Bilmiyorum",
        label=0,
        label_name="negative",
        expected_morphemes=["bil", "miyor", "um"],
        key_morpheme="miyor",
        reason="Negation progressive"
    ),
    TestSample(
        text="Sevmedim",
        label=0,
        label_name="negative",
        expected_morphemes=["sev", "med", "i", "m"],
        key_morpheme="med",
        reason="Negation on love verb"
    ),
    TestSample(
        text="Okumadık",
        label=0,
        label_name="negative",
        expected_morphemes=["oku", "mad", "ı", "k"],
        key_morpheme="mad",
        reason="Negation on read verb"
    ),
    
    # === STRONG POSITIVE SAMPLES ===
    TestSample(
        text="Başardık",
        label=1,
        label_name="positive",
        expected_morphemes=["başard", "ı", "k"],
        key_morpheme="başard",
        reason="Root 'başar' (succeed) drives positive"
    ),
    TestSample(
        text="Sevdim",
        label=1,
        label_name="positive",
        expected_morphemes=["sevd", "i", "m"],
        key_morpheme="sevd",
        reason="Root 'sev' (love) positive"
    ),
    TestSample(
        text="Kazandık",
        label=1,
        label_name="positive",
        expected_morphemes=["kazand", "ı", "k"],
        key_morpheme="kazand",
        reason="Root 'kazan' (win) positive"
    ),
    TestSample(
        text="Güldük",
        label=1,
        label_name="positive",
        expected_morphemes=["güld", "ü", "k"],
        key_morpheme="güld",
        reason="Root 'gül' (laugh) positive"
    ),
    TestSample(
        text="Mutluyum",
        label=1,
        label_name="positive",
        expected_morphemes=["mutlu", "y", "um"],
        key_morpheme="mutlu",
        reason="Adjective 'mutlu' (happy)"
    ),
    TestSample(
        text="Harikasın",
        label=1,
        label_name="positive",
        expected_morphemes=["harika", "sın"],
        key_morpheme="harika",
        reason="Adjective 'harika' (wonderful)"
    ),
    TestSample(
        text="Süperdi",
        label=1,
        label_name="positive",
        expected_morphemes=["süper", "di"],
        key_morpheme="süper",
        reason="Adjective 'süper' drives positive"
    ),
    TestSample(
        text="Berbattı",
        label=0,
        label_name="negative",
        expected_morphemes=["berbat", "tı"],
        key_morpheme="berbat",
        reason="Adjective 'berbat' (terrible) negative"
    ),
    
    # === EVIDENTIAL SAMPLES ===
    TestSample(
        text="Gelmişler",
        label=1,
        label_name="evidential",
        expected_morphemes=["gel", "miş", "ler"],
        key_morpheme="miş",
        reason="Evidential marker indicates hearsay"
    ),
    TestSample(
        text="Yapmışlar",
        label=1,
        label_name="evidential",
        expected_morphemes=["yap", "mış", "lar"],
        key_morpheme="mış",
        reason="Evidential marker"
    ),
    
    # === DERIVATION SAMPLES ===
    TestSample(
        text="Öğretmen",
        label=1,
        label_name="occupation",
        expected_morphemes=["öğret", "men"],
        key_morpheme="men",
        reason="Agent marker -men derives occupation"
    ),
    TestSample(
        text="Yazıcı",
        label=1,
        label_name="occupation",
        expected_morphemes=["yaz", "ıcı"],
        key_morpheme="ıcı",
        reason="Agent marker -ıcı"
    ),
]


def get_test_samples(n: int = None, category: str = None) -> List[Dict[str, Any]]:
    """
    Get test samples for benchmarking.
    
    Args:
        n: Number of samples (default: all)
        category: Filter by label_name
        
    Returns:
        List of sample dictionaries
    """
    samples = TURKISH_SAMPLES
    
    if category:
        samples = [s for s in samples if s.label_name == category]
    
    if n:
        samples = samples[:n]
    
    return [
        {
            "text": s.text,
            "label": s.label,
            "label_name": s.label_name,
            "expected_morphemes": s.expected_morphemes,
            "key_morpheme": s.key_morpheme,
            "reason": s.reason
        }
        for s in samples
    ]


def get_negative_samples() -> List[Dict[str, Any]]:
    """Get only negative sentiment samples."""
    return get_test_samples(category="negative")


def get_positive_samples() -> List[Dict[str, Any]]:
    """Get only positive sentiment samples."""
    return get_test_samples(category="positive")
