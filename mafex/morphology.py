"""
Morphological Analysis Module for MAFEX

Provides:
- MorphemeAnalyzer: Turkish morphological analysis via Zemberek or fallback
- AlignmentMatrixBuilder: Constructs the A matrix (K x T)
- MorphologicalProjector: Projection operator from token to morpheme space

The core innovation is the Alignment Matrix A that maps tokens to morphemes,
enabling projection of token-level attributions to morpheme-level.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
import warnings


@dataclass
class Morpheme:
    """Represents a single morpheme with metadata."""
    surface: str          # Surface form (e.g., "yap")
    lemma: str           # Lemma/root (e.g., "yap")
    pos: str             # Part of speech (e.g., "Verb")
    features: Dict       # Morphological features
    start_char: int      # Character start position in word
    end_char: int        # Character end position in word


@dataclass  
class MorphAnalysis:
    """Complete morphological analysis of a word."""
    word: str
    morphemes: List[Morpheme]
    is_valid: bool
    
    @property
    def morpheme_surfaces(self) -> List[str]:
        return [m.surface for m in self.morphemes]


class ZemberekAnalyzer:
    """
    Wrapper for Zemberek Turkish NLP library.
    Falls back to rule-based analyzer if Zemberek unavailable.
    """
    
    def __init__(self, use_fallback: bool = True):
        self.zemberek = None
        self.use_fallback = use_fallback
        self._try_init_zemberek()
        
    def _try_init_zemberek(self):
        """Attempt to initialize Zemberek via JPype."""
        try:
            import jpype
            import jpype.imports
            
            if not jpype.isJVMStarted():
                # Path to Zemberek JAR - user should configure
                jpype.startJVM(classpath=["./lib/zemberek-full.jar"])
            
            from zemberek.morphology import TurkishMorphology
            self.zemberek = TurkishMorphology.createWithDefaults()
            print("✓ Zemberek initialized successfully")
            
        except Exception as e:
            if self.use_fallback:
                warnings.warn(f"Zemberek unavailable ({e}), using rule-based fallback")
            else:
                raise RuntimeError(f"Zemberek initialization failed: {e}")
    
    def analyze(self, word: str) -> MorphAnalysis:
        """Analyze a single word."""
        if self.zemberek:
            return self._zemberek_analyze(word)
        return self._fallback_analyze(word)
    
    def _zemberek_analyze(self, word: str) -> MorphAnalysis:
        """Use Zemberek for analysis."""
        results = self.zemberek.analyze(word)
        
        if not results:
            return MorphAnalysis(word=word, morphemes=[], is_valid=False)
        
        # Take best analysis
        best = results[0]
        morphemes = []
        
        for morph in best.getMorphemeList():
            morphemes.append(Morpheme(
                surface=str(morph.getSurface()),
                lemma=str(best.getLemmas()[0]) if morphemes == [] else "",
                pos=str(morph.getMorpheme().getPos()),
                features={},
                start_char=0,  # Would need character tracking
                end_char=0
            ))
        
        return MorphAnalysis(word=word, morphemes=morphemes, is_valid=True)
    
    def _fallback_analyze(self, word: str) -> MorphAnalysis:
        """
        Rule-based fallback for Turkish morphology.
        Handles common suffixes for demonstration.
        """
        morphemes = []
        remaining = word.lower()
        char_pos = 0
        
        # Common Turkish suffixes (simplified)
        suffix_patterns = [
            # Negation
            (r'(ma|me|mı|mi|mu|mü)$', 'NEG', 'Negation'),
            (r'(ama|eme|ıma|ime|uma|üme)$', 'NEG.ABIL', 'Inability'),
            # Tense
            (r'(acak|ecek|ıcak|icek|ucak|ücek)$', 'FUT', 'Future'),
            (r'(ıyor|iyor|uyor|üyor)$', 'PROG', 'Progressive'),
            (r'(dı|di|du|dü|tı|ti|tu|tü)$', 'PAST', 'Past'),
            (r'(mış|miş|muş|müş)$', 'EVID', 'Evidential'),
            # Person
            (r'(ım|im|um|üm)$', '1SG', 'FirstPerson'),
            (r'(sın|sin|sun|sün)$', '2SG', 'SecondPerson'),
            (r'(ız|iz|uz|üz)$', '1PL', 'FirstPlural'),
            # Derivation
            (r'(lık|lik|luk|lük)$', 'NMLZ', 'Nominalization'),
            (r'(çı|ci|cu|cü|çü)$', 'PROF', 'Professional'),
            (r'(lı|li|lu|lü)$', 'WITH', 'Possessing'),
            (r'(sız|siz|suz|süz)$', 'WITHOUT', 'Lacking'),
            # Case
            (r'(da|de|ta|te)$', 'LOC', 'Locative'),
            (r'(dan|den|tan|ten)$', 'ABL', 'Ablative'),
            (r'(a|e|ya|ye)$', 'DAT', 'Dative'),
            (r'(ı|i|u|ü|yı|yi|yu|yü)$', 'ACC', 'Accusative'),
        ]
        
        # Extract suffixes iteratively
        extracted_suffixes = []
        for pattern, tag, desc in suffix_patterns:
            match = re.search(pattern, remaining)
            if match:
                suffix = match.group(1)
                extracted_suffixes.append((suffix, tag, desc, match.start(), match.end()))
                remaining = remaining[:match.start()]
        
        # Root is what remains
        if remaining:
            morphemes.append(Morpheme(
                surface=remaining,
                lemma=remaining,
                pos="Root",
                features={},
                start_char=0,
                end_char=len(remaining)
            ))
            char_pos = len(remaining)
        
        # Add suffixes in order
        for suffix, tag, desc, _, _ in reversed(extracted_suffixes):
            morphemes.append(Morpheme(
                surface=suffix,
                lemma=suffix,
                pos=tag,
                features={"description": desc},
                start_char=char_pos,
                end_char=char_pos + len(suffix)
            ))
            char_pos += len(suffix)
        
        return MorphAnalysis(
            word=word,
            morphemes=morphemes if morphemes else [Morpheme(word, word, "UNK", {}, 0, len(word))],
            is_valid=len(morphemes) > 0
        )


class MorphemeAnalyzer:
    """
    High-level morpheme analyzer for text.
    Handles tokenization and word-level analysis.
    """
    
    def __init__(self, analyzer: Optional[ZemberekAnalyzer] = None):
        self.analyzer = analyzer or ZemberekAnalyzer(use_fallback=True)
        self._cache = {}
    
    def analyze_text(self, text: str) -> List[MorphAnalysis]:
        """Analyze all words in text."""
        words = self._tokenize(text)
        return [self.analyze_word(w) for w in words]
    
    def analyze_word(self, word: str) -> MorphAnalysis:
        """Analyze single word with caching."""
        if word not in self._cache:
            self._cache[word] = self.analyzer.analyze(word)
        return self._cache[word]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        # Split on whitespace and punctuation while keeping words
        return re.findall(r'\b\w+\b', text)
    
    def get_morphemes(self, text: str) -> List[str]:
        """Get flat list of all morphemes in text."""
        morphemes = []
        for analysis in self.analyze_text(text):
            morphemes.extend(analysis.morpheme_surfaces)
        return morphemes


class AlignmentMatrixBuilder:
    """
    Builds the Morphological Alignment Matrix A.
    
    A ∈ {0,1}^{K×T} where:
    - K = number of morphemes
    - T = number of tokens
    - A_kj = 1 iff token j is constituent of morpheme k
    
    Satisfies partition property: Σ_k A_kj = 1 for all j
    """
    
    def __init__(self, morpheme_analyzer: MorphemeAnalyzer):
        self.analyzer = morpheme_analyzer
    
    def build(
        self, 
        text: str, 
        tokens: List[str],
        token_offsets: Optional[List[Tuple[int, int]]] = None
    ) -> np.ndarray:
        """
        Build alignment matrix A mapping tokens to morphemes.
        
        Args:
            text: Original input text
            tokens: List of tokens from LLM tokenizer
            token_offsets: Character offsets for each token (start, end)
            
        Returns:
            A: Alignment matrix of shape (K, T)
        """
        # Get morpheme analysis
        analyses = self.analyzer.analyze_text(text)
        morphemes = []
        morpheme_spans = []  # Character spans for each morpheme
        
        char_offset = 0
        for analysis in analyses:
            for morph in analysis.morphemes:
                morphemes.append(morph.surface)
                # Estimate character span
                start = char_offset
                end = char_offset + len(morph.surface)
                morpheme_spans.append((start, end))
            char_offset += len(analysis.word) + 1  # +1 for space
        
        K = len(morphemes)
        T = len(tokens)
        
        # Build alignment matrix
        A = np.zeros((K, T), dtype=np.float32)
        
        if token_offsets is None:
            # Fallback: use string matching heuristic
            A = self._align_by_string_match(morphemes, tokens)
        else:
            # Use character span overlap
            A = self._align_by_spans(morpheme_spans, token_offsets, K, T)
        
        # Ensure partition property: normalize columns
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1  # Avoid division by zero
        A = A / col_sums
        
        return A, morphemes
    
    def _align_by_string_match(
        self, 
        morphemes: List[str], 
        tokens: List[str]
    ) -> np.ndarray:
        """
        Align morphemes to tokens using string matching.
        Uses greedy matching based on character overlap.
        """
        K = len(morphemes)
        T = len(tokens)
        A = np.zeros((K, T), dtype=np.float32)
        
        # Clean tokens (remove ## prefixes, etc.)
        clean_tokens = [t.replace('##', '').replace('▁', '').lower() for t in tokens]
        
        # Build morpheme string and token string for alignment
        morph_str = ''.join(morphemes).lower()
        token_str = ''.join(clean_tokens)
        
        # Character position tracking
        morph_char_pos = 0
        token_char_pos = 0
        morph_idx = 0
        token_idx = 0
        
        # Greedy alignment
        while morph_idx < K and token_idx < T:
            morph = morphemes[morph_idx].lower()
            tok = clean_tokens[token_idx]
            
            if not tok:  # Skip empty tokens
                token_idx += 1
                continue
            
            # Check overlap
            if morph.startswith(tok) or tok.startswith(morph):
                A[morph_idx, token_idx] = 1.0
            
            # Advance pointers based on character consumption
            morph_len = len(morph)
            tok_len = len(tok)
            
            if tok_len <= morph_len:
                token_idx += 1
                if tok_len == morph_len:
                    morph_idx += 1
            else:
                morph_idx += 1
        
        # Handle remaining unaligned tokens -> assign to last morpheme
        while token_idx < T:
            if morph_idx > 0:
                A[morph_idx - 1, token_idx] = 1.0
            token_idx += 1
        
        return A
    
    def _align_by_spans(
        self,
        morpheme_spans: List[Tuple[int, int]],
        token_offsets: List[Tuple[int, int]],
        K: int,
        T: int
    ) -> np.ndarray:
        """Align using character span overlap."""
        A = np.zeros((K, T), dtype=np.float32)
        
        for k, (m_start, m_end) in enumerate(morpheme_spans):
            for t, (t_start, t_end) in enumerate(token_offsets):
                # Check if spans overlap
                overlap = max(0, min(m_end, t_end) - max(m_start, t_start))
                if overlap > 0:
                    # Proportional assignment based on overlap
                    A[k, t] = overlap / max(t_end - t_start, 1)
        
        return A


class MorphologicalProjector:
    """
    Implements the core projection operator from paper Eq. (2):
    φ_morph = P(φ_tok) = A · φ_tok
    
    Projects token-level attributions to morpheme-level,
    preserving the Completeness Axiom (Theorem 1).
    """
    
    def __init__(self, alignment_builder: AlignmentMatrixBuilder):
        self.alignment_builder = alignment_builder
    
    def project(
        self,
        token_attributions: np.ndarray,
        text: str,
        tokens: List[str],
        token_offsets: Optional[List[Tuple[int, int]]] = None
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Project token attributions to morpheme space.
        
        Args:
            token_attributions: φ_tok ∈ R^T
            text: Original text
            tokens: Token list
            token_offsets: Optional character offsets
            
        Returns:
            morpheme_attributions: φ_morph ∈ R^K
            morphemes: List of morpheme strings
            A: Alignment matrix used
        """
        # Build alignment matrix
        A, morphemes = self.alignment_builder.build(text, tokens, token_offsets)
        
        # Project: φ_morph = A · φ_tok
        morpheme_attributions = A @ token_attributions
        
        # Verify completeness
        tok_sum = np.sum(token_attributions)
        morph_sum = np.sum(morpheme_attributions)
        
        if not np.isclose(tok_sum, morph_sum, rtol=1e-5):
            warnings.warn(
                f"Completeness violation: token_sum={tok_sum:.4f}, "
                f"morph_sum={morph_sum:.4f}"
            )
        
        return morpheme_attributions, morphemes, A


# Convenience function
def create_morphology_pipeline() -> MorphologicalProjector:
    """Create a ready-to-use morphology pipeline."""
    analyzer = MorphemeAnalyzer()
    builder = AlignmentMatrixBuilder(analyzer)
    return MorphologicalProjector(builder)


if __name__ == "__main__":
    # Quick test
    analyzer = MorphemeAnalyzer()
    
    test_words = [
        "yapamayacakmış",  # yap-ama-yacak-mış
        "gelemedim",       # gel-eme-di-m
        "gözlükçü",        # göz-lük-çü
        "evlerden",        # ev-ler-den
    ]
    
    print("Turkish Morphological Analysis Examples:\n")
    for word in test_words:
        analysis = analyzer.analyze_word(word)
        print(f"Word: {word}")
        print(f"  Morphemes: {' + '.join(analysis.morpheme_surfaces)}")
        for m in analysis.morphemes:
            print(f"    - {m.surface} [{m.pos}]")
        print()
