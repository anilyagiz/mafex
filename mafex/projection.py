"""
MAFEX Projection Pipeline

Implements the core MAFEX framework:
- CausalRegularizer: Causal ablation for attribution verification
- MAFEXPipeline: End-to-end pipeline combining projection + causal regularization

Key equation (Eq. 5):
    S* = λ·φ_morph + (1-λ)·φ_causal
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .morphology import MorphemeAnalyzer, AlignmentMatrixBuilder, MorphologicalProjector
from .attribution import IntegratedGradients, BaseAttributor


@dataclass
class MAFEXResult:
    """Container for MAFEX attribution results."""
    # Input info
    text: str
    tokens: List[str]
    morphemes: List[str]
    
    # Attributions
    token_attributions: np.ndarray      # φ_tok ∈ R^T
    morpheme_attributions: np.ndarray   # φ_morph ∈ R^K (projected)
    causal_attributions: np.ndarray     # φ_causal ∈ R^K
    final_attributions: np.ndarray      # S* ∈ R^K (Eq. 5)
    
    # Matrices
    alignment_matrix: np.ndarray        # A ∈ R^{K×T}
    
    # Metadata
    lambda_value: float
    target_class: int
    model_output: float
    
    def get_top_morphemes(self, k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k attributed morphemes."""
        indices = np.argsort(self.final_attributions)[::-1][:k]
        return [(self.morphemes[i], self.final_attributions[i]) for i in indices]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "tokens": self.tokens,
            "morphemes": self.morphemes,
            "final_attributions": self.final_attributions.tolist(),
            "top_morphemes": self.get_top_morphemes(),
            "lambda": self.lambda_value,
        }


class CausalRegularizer:
    """
    Causal ablation for morpheme-level attribution verification.
    
    Computes φ_causal by measuring output change when morphemes are masked:
        φ_causal^(k) = F(x) - F(x_{\\μ_k})
    
    This provides ground-truth causal importance that filters false positives
    from gradient-based methods.
    """
    
    def __init__(
        self,
        mask_token_id: int = 0,  # Usually [PAD] or [MASK]
        baseline_type: str = "mask"  # "mask", "zero", or "delete"
    ):
        self.mask_token_id = mask_token_id
        self.baseline_type = baseline_type
    
    def compute_causal_effects(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        alignment_matrix: np.ndarray,
        morphemes: List[str],
        target_idx: int,
        tokens: List[str]
    ) -> np.ndarray:
        """
        Compute causal importance for each morpheme via ablation.
        
        Args:
            model: The LLM model
            input_ids: Token IDs [1, seq_len]
            attention_mask: Attention mask
            alignment_matrix: A matrix [K, T]
            morphemes: List of morpheme strings
            target_idx: Target class index
            tokens: Token strings
            
        Returns:
            causal_effects: [K] array of causal importance scores
        """
        device = input_ids.device
        K = len(morphemes)
        
        # Get baseline prediction
        with torch.no_grad():
            base_output = self._forward(model, input_ids, attention_mask, target_idx)
        
        causal_effects = np.zeros(K)
        
        for k in range(K):
            # Find tokens that belong to morpheme k
            token_indices = np.where(alignment_matrix[k] > 0.5)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Create ablated input
            ablated_ids = input_ids.clone()
            
            if self.baseline_type == "mask":
                ablated_ids[0, token_indices] = self.mask_token_id
            elif self.baseline_type == "zero":
                ablated_ids[0, token_indices] = 0
            # "delete" would require reshaping - not implemented
            
            # Ablated prediction
            with torch.no_grad():
                ablated_output = self._forward(
                    model, ablated_ids, attention_mask, target_idx
                )
            
            # Causal effect = change in output
            causal_effects[k] = base_output - ablated_output
        
        return causal_effects
    
    def _forward(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int
    ) -> float:
        """Forward pass returning target probability."""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Handle sequence classification vs. LM
        if logits.dim() == 3:  # [batch, seq, vocab]
            probs = F.softmax(logits[:, -1, :], dim=-1)
        else:  # [batch, num_classes]
            probs = F.softmax(logits, dim=-1)
        
        return probs[0, target_idx].item()


class MAFEXPipeline:
    """
    Complete MAFEX (Morpheme-Aligned Faithful Explanations) Pipeline.
    
    Stages:
    1. Morphological Analysis: Parse input into morphemes
    2. Alignment: Build token-morpheme alignment matrix A
    3. Token Attribution: Compute φ_tok via IG
    4. Projection: φ_morph = A · φ_tok
    5. Causal Regularization: S* = λ·φ_morph + (1-λ)·φ_causal
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        lambda_causal: float = 0.7,
        ig_steps: int = 50,
        mask_token_id: Optional[int] = None
    ):
        """
        Initialize MAFEX pipeline.
        
        Args:
            model: The LLM model (encoder or decoder)
            tokenizer: HuggingFace tokenizer
            lambda_causal: Trade-off λ in Eq. 5 (default: 0.7)
            ig_steps: Number of IG interpolation steps
            mask_token_id: Token ID for masking (default: tokenizer's pad_token_id)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_causal = lambda_causal
        
        # Initialize components
        self.morpheme_analyzer = MorphemeAnalyzer()
        self.alignment_builder = AlignmentMatrixBuilder(self.morpheme_analyzer)
        self.projector = MorphologicalProjector(self.alignment_builder)
        self.ig = IntegratedGradients(n_steps=ig_steps)
        self.causal_reg = CausalRegularizer(
            mask_token_id=mask_token_id or tokenizer.pad_token_id or 0
        )
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def explain(
        self,
        text: str,
        target_idx: Optional[int] = None,
        return_intermediates: bool = False
    ) -> MAFEXResult:
        """
        Generate morpheme-level explanation for text.
        
        Args:
            text: Input Turkish text
            target_idx: Target class/token index (auto-detected if None)
            return_intermediates: Include intermediate results
            
        Returns:
            MAFEXResult with all attribution information
        """
        device = next(self.model.parameters()).device
        
        # Step 1: Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Get token strings and offsets
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        offsets = encoding.get("offset_mapping", None)
        if offsets is not None:
            offsets = offsets.squeeze().tolist()
        
        # Step 2: Auto-detect target if needed
        if target_idx is None:
            target_idx = self._get_predicted_class(input_ids, attention_mask)
        
        # Step 3: Compute token-level attributions (φ_tok)
        token_attributions = self.ig.attribute(
            self.model,
            input_ids,
            target_idx,
            attention_mask=attention_mask
        )
        
        # Step 4: Build alignment matrix and project (φ_morph = A · φ_tok)
        morpheme_attributions, morphemes, A = self.projector.project(
            token_attributions,
            text,
            tokens,
            offsets
        )
        
        # Step 5: Compute causal effects (φ_causal)
        causal_attributions = self.causal_reg.compute_causal_effects(
            self.model,
            input_ids,
            attention_mask,
            A,
            morphemes,
            target_idx,
            tokens
        )
        
        # Step 6: Final attribution (Eq. 5)
        # S* = λ·φ_morph + (1-λ)·φ_causal
        final_attributions = (
            self.lambda_causal * morpheme_attributions +
            (1 - self.lambda_causal) * causal_attributions
        )
        
        # Get model output probability
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            if logits.dim() == 3:
                probs = F.softmax(logits[:, -1, :], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            model_output = probs[0, target_idx].item()
        
        return MAFEXResult(
            text=text,
            tokens=tokens,
            morphemes=morphemes,
            token_attributions=token_attributions,
            morpheme_attributions=morpheme_attributions,
            causal_attributions=causal_attributions,
            final_attributions=final_attributions,
            alignment_matrix=A,
            lambda_value=self.lambda_causal,
            target_class=target_idx,
            model_output=model_output
        )
    
    def _get_predicted_class(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> int:
        """Get the model's predicted class."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if logits.dim() == 3:  # LM head
                predicted = logits[:, -1, :].argmax(dim=-1).item()
            else:  # Classification head
                predicted = logits.argmax(dim=-1).item()
        
        return predicted
    
    def explain_batch(
        self,
        texts: List[str],
        target_indices: Optional[List[int]] = None,
        show_progress: bool = True
    ) -> List[MAFEXResult]:
        """
        Generate explanations for multiple texts.
        
        Args:
            texts: List of input texts
            target_indices: Optional target indices per text
            show_progress: Show progress bar
            
        Returns:
            List of MAFEXResult objects
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(texts, desc="MAFEX") if show_progress else texts
        
        for i, text in enumerate(iterator):
            target = target_indices[i] if target_indices else None
            result = self.explain(text, target)
            results.append(result)
        
        return results


class TokenBaselinePipeline:
    """
    Baseline pipeline without morphological alignment.
    Used for comparison with MAFEX.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        ig_steps: int = 50
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(n_steps=ig_steps)
        self.model.eval()
    
    def explain(
        self,
        text: str,
        target_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate token-level explanation."""
        device = next(self.model.parameters()).device
        
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        if target_idx is None:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if logits.dim() == 3:
                    target_idx = logits[:, -1, :].argmax(dim=-1).item()
                else:
                    target_idx = logits.argmax(dim=-1).item()
        
        attributions = self.ig.attribute(
            self.model,
            input_ids,
            target_idx,
            attention_mask=attention_mask
        )
        
        return {
            "text": text,
            "tokens": tokens,
            "attributions": attributions,
            "target_class": target_idx
        }


def compare_methods(
    model,
    tokenizer,
    text: str,
    target_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compare MAFEX with token-level baseline.
    
    Returns:
        Dictionary with both methods' results
    """
    # Run MAFEX
    mafex = MAFEXPipeline(model, tokenizer)
    mafex_result = mafex.explain(text, target_idx)
    
    # Run baseline
    baseline = TokenBaselinePipeline(model, tokenizer)
    baseline_result = baseline.explain(text, target_idx)
    
    return {
        "text": text,
        "mafex": {
            "morphemes": mafex_result.morphemes,
            "attributions": mafex_result.final_attributions.tolist(),
            "top_features": mafex_result.get_top_morphemes(5)
        },
        "baseline": {
            "tokens": baseline_result["tokens"],
            "attributions": baseline_result["attributions"].tolist(),
        }
    }
