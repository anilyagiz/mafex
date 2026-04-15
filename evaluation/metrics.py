"""
ERASER Evaluation Metrics for MAFEX

Implements faithfulness metrics from DeYoung et al. (2020):
- Comprehensiveness: Does removing important features hurt performance?
- Sufficiency: Are important features enough to maintain performance?

Also includes additional metrics for morphological alignment evaluation.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    comprehensiveness: float
    sufficiency: float
    comprehensiveness_auc: float
    sufficiency_auc: float
    
    # Optional metrics
    faithfulness_correlation: Optional[float] = None
    plausibility_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "comprehensiveness": self.comprehensiveness,
            "sufficiency": self.sufficiency,
            "comp_auc": self.comprehensiveness_auc,
            "suff_auc": self.sufficiency_auc,
            "faith_corr": self.faithfulness_correlation,
            "plausibility": self.plausibility_score
        }


class ERASEREvaluator:
    """
    ERASER-style faithfulness evaluation.
    
    Metrics:
    - Comprehensiveness: P(y|x) - P(y|x \ top-k)
      Higher = removing important features hurts more = good
    
    - Sufficiency: P(y|x) - P(y|top-k only)
      Lower = important features alone are enough = good
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        top_k_ratios: List[float] = [0.1, 0.2, 0.3],
        mask_token_id: Optional[int] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k_ratios = top_k_ratios
        self.mask_token_id = mask_token_id or tokenizer.pad_token_id or 0
        
        self.model.eval()
        self.device = next(model.parameters()).device
    
    def evaluate(
        self,
        text: str,
        attributions: np.ndarray,
        tokens: List[str],
        target_idx: int
    ) -> EvaluationResult:
        """
        Evaluate attribution quality using ERASER metrics.
        
        Args:
            text: Original input text
            attributions: Attribution scores (token-level or morpheme-level)
            tokens: Token list (aligned with attributions)
            target_idx: Target class index
            
        Returns:
            EvaluationResult with comprehensiveness and sufficiency
        """
        # Encode
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Get base prediction
        base_prob = self._get_probability(input_ids, attention_mask, target_idx)
        
        comp_scores = []
        suff_scores = []
        
        for k_ratio in self.top_k_ratios:
            k = max(1, int(len(attributions) * k_ratio))
            top_k_indices = np.argsort(np.abs(attributions))[::-1][:k]
            
            # Comprehensiveness: remove top-k
            comp_prob = self._evaluate_removal(
                input_ids, attention_mask, top_k_indices, target_idx
            )
            comp_scores.append(base_prob - comp_prob)
            
            # Sufficiency: keep only top-k
            suff_prob = self._evaluate_keep_only(
                input_ids, attention_mask, top_k_indices, target_idx
            )
            suff_scores.append(base_prob - suff_prob)
        
        # Compute AUC (area under the curve across k values)
        comp_auc = np.trapz(comp_scores, self.top_k_ratios)
        suff_auc = np.trapz(suff_scores, self.top_k_ratios)
        
        return EvaluationResult(
            comprehensiveness=np.mean(comp_scores),
            sufficiency=np.mean(suff_scores),
            comprehensiveness_auc=comp_auc,
            sufficiency_auc=suff_auc
        )
    
    def _get_probability(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int
    ) -> float:
        """Get probability for target class."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            
            if logits.dim() == 3:
                probs = F.softmax(logits[:, -1, :], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            
            return probs[0, target_idx].item()
    
    def _evaluate_removal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        indices_to_remove: np.ndarray,
        target_idx: int
    ) -> float:
        """Evaluate after removing specified indices."""
        masked_ids = input_ids.clone()
        
        for idx in indices_to_remove:
            if idx < masked_ids.shape[1]:
                masked_ids[0, idx] = self.mask_token_id
        
        return self._get_probability(masked_ids, attention_mask, target_idx)
    
    def _evaluate_keep_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        indices_to_keep: np.ndarray,
        target_idx: int
    ) -> float:
        """Evaluate keeping only specified indices."""
        masked_ids = torch.full_like(input_ids, self.mask_token_id)
        
        for idx in indices_to_keep:
            if idx < input_ids.shape[1]:
                masked_ids[0, idx] = input_ids[0, idx]
        
        # Keep special tokens
        special_tokens = [
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ]
        
        for i, token_id in enumerate(input_ids[0]):
            if token_id.item() in special_tokens:
                masked_ids[0, i] = token_id
        
        return self._get_probability(masked_ids, attention_mask, target_idx)


def compute_faithfulness_correlation(
    gradient_attributions: np.ndarray,
    causal_effects: np.ndarray
) -> float:
    """
    Compute correlation between gradient-based and causal attributions.
    Higher correlation indicates more faithful gradients.
    """
    from scipy.stats import spearmanr
    
    if len(gradient_attributions) != len(causal_effects):
        raise ValueError("Attribution lengths must match")
    
    corr, _ = spearmanr(gradient_attributions, causal_effects)
    return corr if not np.isnan(corr) else 0.0


def compare_methods(
    evaluator: ERASEREvaluator,
    text: str,
    mafex_result: Any,
    baseline_result: Dict
) -> Dict[str, EvaluationResult]:
    """
    Compare MAFEX with baseline on same input.
    
    Returns:
        Dictionary with evaluation results for each method
    """
    target_idx = mafex_result.target_class
    
    # Evaluate MAFEX
    mafex_eval = evaluator.evaluate(
        text,
        mafex_result.final_attributions,
        mafex_result.morphemes,
        target_idx
    )
    
    # Evaluate baseline
    baseline_eval = evaluator.evaluate(
        text,
        baseline_result["attributions"],
        baseline_result["tokens"],
        target_idx
    )
    
    return {
        "mafex": mafex_eval,
        "baseline": baseline_eval,
        "improvement": {
            "comp": mafex_eval.comprehensiveness - baseline_eval.comprehensiveness,
            "suff": baseline_eval.sufficiency - mafex_eval.sufficiency,  # Lower is better
        }
    }


class BenchmarkRunner:
    """
    Run full benchmark evaluation across multiple samples and methods.
    """
    
    def __init__(
        self,
        mafex_pipeline,
        evaluator: ERASEREvaluator
    ):
        self.mafex = mafex_pipeline
        self.evaluator = evaluator
    
    def run(
        self,
        samples: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run benchmark on samples.
        
        Args:
            samples: List of {"text": str, "label": int}
            show_progress: Show progress bar
            
        Returns:
            Aggregated results
        """
        results = {
            "mafex_comp": [],
            "mafex_suff": [],
            "baseline_comp": [],
            "baseline_suff": [],
        }
        
        iterator = tqdm(samples, desc="Evaluating") if show_progress else samples
        
        for sample in iterator:
            text = sample["text"]
            target = sample.get("label", None)
            
            try:
                # Run MAFEX
                mafex_result = self.mafex.explain(text, target)
                
                # Calculate ERASER metrics
                mafex_eval = self.evaluator.evaluate(
                    text,
                    mafex_result.final_attributions,
                    mafex_result.morphemes,
                    mafex_result.target_class
                )
                
                # Token baseline
                baseline_eval = self.evaluator.evaluate(
                    text,
                    mafex_result.token_attributions,
                    mafex_result.tokens,
                    mafex_result.target_class
                )
                
                results["mafex_comp"].append(mafex_eval.comprehensiveness)
                results["mafex_suff"].append(mafex_eval.sufficiency)
                results["baseline_comp"].append(baseline_eval.comprehensiveness)
                results["baseline_suff"].append(baseline_eval.sufficiency)
                
            except Exception as e:
                print(f"Error on sample: {e}")
                continue
        
        # Aggregate
        return {
            "mafex": {
                "comprehensiveness": np.mean(results["mafex_comp"]),
                "sufficiency": np.mean(results["mafex_suff"]),
                "comp_std": np.std(results["mafex_comp"]),
                "suff_std": np.std(results["mafex_suff"]),
            },
            "baseline": {
                "comprehensiveness": np.mean(results["baseline_comp"]),
                "sufficiency": np.mean(results["baseline_suff"]),
                "comp_std": np.std(results["baseline_comp"]),
                "suff_std": np.std(results["baseline_suff"]),
            },
            "n_samples": len(results["mafex_comp"]),
            "improvement": {
                "comp_gain": (
                    np.mean(results["mafex_comp"]) - 
                    np.mean(results["baseline_comp"])
                ) / np.mean(results["baseline_comp"]) * 100
            }
        }
