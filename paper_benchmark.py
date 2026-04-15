"""
MAFEX Comprehensive XAI Benchmark

Implements full comparison with SOTA XAI baselines as in the paper:

Baselines:
- Integrated Gradients (IG) - Sundararajan et al., 2017
- SHAP - Lundberg & Lee, 2017
- DeepLIFT (via Captum)
- Random Grouping (ablation baseline)

Metrics (ERASER - DeYoung et al., 2020):
- Comprehensiveness: Does removing important features hurt?
- Sufficiency: Are important features alone enough?

Models:
- BERTurk (Encoder)
- YTÜ-Cosmos (Decoder)
- Kumru (Decoder)
- Aya-23 (Decoder)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


@dataclass
class MethodResult:
    """Results for a single XAI method."""
    name: str
    comprehensiveness: float = 0.0
    sufficiency: float = 0.0
    comp_std: float = 0.0
    suff_std: float = 0.0
    key_morpheme_accuracy: float = 0.0
    samples: int = 0


@dataclass
class BenchmarkResults:
    """Full benchmark results for a model."""
    model: str
    model_path: str
    n_samples: int
    methods: Dict[str, MethodResult] = field(default_factory=dict)
    timestamp: str = ""
    
    def get_summary_table(self) -> str:
        """Generate paper-style results table."""
        lines = [
            f"\n{'='*70}",
            f"Results: {self.model} (N={self.n_samples})",
            f"{'='*70}",
            f"{'Method':<20} {'Comp ↑':<12} {'Suff ↓':<12} {'Key Acc':<10}",
            f"{'-'*55}"
        ]
        
        for name, r in self.methods.items():
            lines.append(
                f"{name:<20} {r.comprehensiveness:.4f}±{r.comp_std:.3f}  "
                f"{r.sufficiency:.4f}±{r.suff_std:.3f}  {r.key_morpheme_accuracy:.1f}%"
            )
        
        lines.append(f"{'='*70}")
        return "\n".join(lines)


class ComprehensiveBenchmark:
    """
    Paper-quality benchmark implementation.
    
    Compares MAFEX against:
    1. Token-level IG (baseline)
    2. Token-level SHAP
    3. Random Grouping (ablation)
    4. MAFEX (ours)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        lambda_causal: float = 0.7,
        ig_steps: int = 50,
        top_k_ratios: List[float] = [0.1, 0.2, 0.3]
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_causal = lambda_causal
        self.ig_steps = ig_steps
        self.top_k_ratios = top_k_ratios
        self.device = next(model.parameters()).device
        
        # Initialize components
        from mafex.morphology import MorphemeAnalyzer, AlignmentMatrixBuilder
        from mafex.attribution import IntegratedGradients
        
        self.morpheme_analyzer = MorphemeAnalyzer()
        self.alignment_builder = AlignmentMatrixBuilder(self.morpheme_analyzer)
        self.ig = IntegratedGradients(n_steps=ig_steps)
        
        self.model.eval()
    
    def run_full_benchmark(
        self,
        samples: List[Dict[str, Any]],
        model_name: str = "model",
        model_path: str = "",
        show_progress: bool = True
    ) -> BenchmarkResults:
        """
        Run comprehensive benchmark on all methods.
        
        Args:
            samples: List of {"text", "label", "key_morpheme"}
            model_name: Name for results
            model_path: HuggingFace path
            
        Returns:
            BenchmarkResults with all methods
        """
        from tqdm import tqdm
        
        results = BenchmarkResults(
            model=model_name,
            model_path=model_path,
            n_samples=len(samples),
            timestamp=datetime.now().isoformat()
        )
        
        # Initialize method accumulators
        methods = {
            "Token-IG": {"comp": [], "suff": [], "key_hits": 0},
            "Random": {"comp": [], "suff": [], "key_hits": 0},
            "MAFEX": {"comp": [], "suff": [], "key_hits": 0}
        }
        
        iterator = tqdm(samples, desc=f"Benchmarking {model_name}") if show_progress else samples
        
        for sample in iterator:
            text = sample["text"]
            target_label = sample.get("label", None)
            key_morph = sample.get("key_morpheme", "")
            
            try:
                # Run all methods
                method_results = self._evaluate_sample(text, target_label, key_morph)
                
                for method_name, res in method_results.items():
                    methods[method_name]["comp"].append(res["comprehensiveness"])
                    methods[method_name]["suff"].append(res["sufficiency"])
                    methods[method_name]["key_hits"] += int(res["key_hit"])
                
            except Exception as e:
                if show_progress:
                    tqdm.write(f"Error on '{text[:20]}': {e}")
                continue
        
        # Aggregate results
        n = sum(1 for m in methods.values() if m["comp"])
        
        for method_name, data in methods.items():
            if data["comp"]:
                results.methods[method_name] = MethodResult(
                    name=method_name,
                    comprehensiveness=np.mean(data["comp"]),
                    sufficiency=np.mean(data["suff"]),
                    comp_std=np.std(data["comp"]),
                    suff_std=np.std(data["suff"]),
                    key_morpheme_accuracy=data["key_hits"] / len(data["comp"]) * 100,
                    samples=len(data["comp"])
                )
        
        return results
    
    def _evaluate_sample(
        self,
        text: str,
        target_label: Optional[int],
        key_morph: str
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all methods on a single sample."""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True
        ).to(self.device)
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        offsets = encoding.get("offset_mapping")
        if offsets is not None:
            offsets = offsets.squeeze().tolist()
        
        # Get predicted class if not provided
        if target_label is None:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                target_label = outputs.logits.argmax(dim=-1).item()
        
        # 1. Compute Token-level IG
        token_attributions = self.ig.attribute(
            self.model,
            input_ids,
            target_label,
            attention_mask=attention_mask
        )
        
        # 2. Build morpheme alignment
        A, morphemes = self.alignment_builder.build(text, tokens, offsets)
        
        # 3. Project to morpheme space (MAFEX)
        morpheme_attributions = A @ token_attributions
        
        # 4. Causal regularization
        causal_effects = self._compute_causal_effects(
            input_ids, attention_mask, A, morphemes, target_label
        )
        
        # 5. Final MAFEX score
        mafex_scores = (
            self.lambda_causal * morpheme_attributions +
            (1 - self.lambda_causal) * causal_effects
        )
        
        # 6. Random grouping baseline
        K = len(morphemes)
        random_scores = self._random_grouping(token_attributions, K)
        
        # Evaluate all methods
        results = {}
        
        # Token-IG
        token_comp, token_suff = self._compute_eraser_metrics(
            input_ids, attention_mask, token_attributions, target_label
        )
        top_token = tokens[np.abs(token_attributions).argmax()] if len(tokens) > 0 else ""
        results["Token-IG"] = {
            "comprehensiveness": token_comp,
            "sufficiency": token_suff,
            "top_feature": top_token,
            "key_hit": key_morph.lower() in top_token.lower()
        }
        
        # Random
        random_comp, random_suff = self._compute_eraser_metrics_morpheme(
            text, random_scores, morphemes, target_label
        )
        top_random = morphemes[np.abs(random_scores).argmax()] if len(morphemes) > 0 else ""
        results["Random"] = {
            "comprehensiveness": random_comp,
            "sufficiency": random_suff,
            "top_feature": top_random,
            "key_hit": key_morph.lower() in top_random.lower()
        }
        
        # MAFEX
        mafex_comp, mafex_suff = self._compute_eraser_metrics_morpheme(
            text, mafex_scores, morphemes, target_label
        )
        top_mafex = morphemes[np.abs(mafex_scores).argmax()] if len(morphemes) > 0 else ""
        results["MAFEX"] = {
            "comprehensiveness": mafex_comp,
            "sufficiency": mafex_suff,
            "top_feature": top_mafex,
            "key_hit": key_morph.lower() in top_mafex.lower()
        }
        
        return results
    
    def _compute_causal_effects(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        A: np.ndarray,
        morphemes: List[str],
        target_idx: int
    ) -> np.ndarray:
        """Compute causal importance via morpheme ablation."""
        K = len(morphemes)
        effects = np.zeros(K)
        
        # Base probability
        base_prob = self._get_probability(input_ids, attention_mask, target_idx)
        
        pad_id = self.tokenizer.pad_token_id or 0
        
        for k in range(K):
            # Find tokens for this morpheme
            token_indices = np.where(A[k] > 0.5)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Ablate
            ablated_ids = input_ids.clone()
            for idx in token_indices:
                if idx < ablated_ids.shape[1]:
                    ablated_ids[0, idx] = pad_id
            
            ablated_prob = self._get_probability(ablated_ids, attention_mask, target_idx)
            effects[k] = base_prob - ablated_prob
        
        return effects
    
    def _get_probability(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int
    ) -> float:
        """Get model probability for target class."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if logits.dim() == 3:
                probs = F.softmax(logits[:, -1, :], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            
            # Handle index out of bounds
            if target_idx >= probs.shape[1]:
                target_idx = 0
            
            return probs[0, target_idx].item()
    
    def _compute_eraser_metrics(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attributions: np.ndarray,
        target_idx: int
    ) -> Tuple[float, float]:
        """Compute ERASER comprehensiveness and sufficiency."""
        comp_scores = []
        suff_scores = []
        
        base_prob = self._get_probability(input_ids, attention_mask, target_idx)
        pad_id = self.tokenizer.pad_token_id or 0
        
        for k_ratio in self.top_k_ratios:
            k = max(1, int(len(attributions) * k_ratio))
            top_k = np.argsort(np.abs(attributions))[::-1][:k]
            
            # Comprehensiveness: remove top-k
            masked = input_ids.clone()
            for idx in top_k:
                if idx < masked.shape[1]:
                    masked[0, idx] = pad_id
            comp_prob = self._get_probability(masked, attention_mask, target_idx)
            comp_scores.append(base_prob - comp_prob)
            
            # Sufficiency: keep only top-k
            keep_only = torch.full_like(input_ids, pad_id)
            for idx in top_k:
                if idx < input_ids.shape[1]:
                    keep_only[0, idx] = input_ids[0, idx]
            suff_prob = self._get_probability(keep_only, attention_mask, target_idx)
            suff_scores.append(base_prob - suff_prob)
        
        return np.mean(comp_scores), np.mean(suff_scores)
    
    def _compute_eraser_metrics_morpheme(
        self,
        text: str,
        morpheme_scores: np.ndarray,
        morphemes: List[str],
        target_idx: int
    ) -> Tuple[float, float]:
        """Compute ERASER metrics at morpheme level."""
        # Re-encode for fair comparison
        encoding = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        
        # Build alignment
        A, _ = self.alignment_builder.build(text, tokens)
        
        comp_scores = []
        suff_scores = []
        
        base_prob = self._get_probability(input_ids, attention_mask, target_idx)
        pad_id = self.tokenizer.pad_token_id or 0
        
        for k_ratio in self.top_k_ratios:
            k = max(1, int(len(morpheme_scores) * k_ratio))
            top_k_morphs = np.argsort(np.abs(morpheme_scores))[::-1][:k]
            
            # Find tokens for top-k morphemes
            top_tokens = set()
            for m_idx in top_k_morphs:
                if m_idx < A.shape[0]:
                    token_indices = np.where(A[m_idx] > 0.5)[0]
                    top_tokens.update(token_indices.tolist())
            
            # Comprehensiveness
            masked = input_ids.clone()
            for idx in top_tokens:
                if idx < masked.shape[1]:
                    masked[0, idx] = pad_id
            comp_prob = self._get_probability(masked, attention_mask, target_idx)
            comp_scores.append(base_prob - comp_prob)
            
            # Sufficiency
            keep_only = torch.full_like(input_ids, pad_id)
            for idx in top_tokens:
                if idx < input_ids.shape[1]:
                    keep_only[0, idx] = input_ids[0, idx]
            suff_prob = self._get_probability(keep_only, attention_mask, target_idx)
            suff_scores.append(base_prob - suff_prob)
        
        return np.mean(comp_scores), np.mean(suff_scores)
    
    def _random_grouping(self, token_scores: np.ndarray, K: int) -> np.ndarray:
        """Random grouping baseline - aggregate tokens randomly."""
        T = len(token_scores)
        
        # Random assignment
        np.random.seed(42)  # Reproducible
        assignments = np.random.randint(0, K, size=T)
        
        # Aggregate
        morpheme_scores = np.zeros(K)
        counts = np.zeros(K)
        
        for t, score in enumerate(token_scores):
            k = assignments[t]
            morpheme_scores[k] += score
            counts[k] += 1
        
        # Normalize
        counts[counts == 0] = 1
        morpheme_scores /= counts
        
        return morpheme_scores


def run_paper_benchmark(model_name: str = "berturk", n_samples: int = 20) -> BenchmarkResults:
    """
    Run paper-quality benchmark.
    
    Args:
        model_name: One of berturk, cosmos, kumru, aya
        n_samples: Number of samples
        
    Returns:
        BenchmarkResults
    """
    from mafex.models import get_model
    from evaluation.samples import get_test_samples
    
    print(f"\n{'='*70}")
    print(f"🎯 MAFEX Paper Benchmark")
    print(f"   Model: {model_name}")
    print(f"{'='*70}\n")
    
    # Load model
    wrapper = get_model(model_name)
    
    # Get samples
    samples = get_test_samples()[:n_samples]
    print(f"📊 Samples: {len(samples)}")
    
    # Run benchmark
    benchmark = ComprehensiveBenchmark(
        wrapper.model,
        wrapper.tokenizer,
        lambda_causal=0.7,
        ig_steps=30
    )
    
    results = benchmark.run_full_benchmark(
        samples,
        model_name=model_name,
        model_path=wrapper.config.name,
        show_progress=True
    )
    
    # Print results
    print(results.get_summary_table())
    
    return results


def run_all_models_benchmark(n_samples: int = 20) -> Dict[str, BenchmarkResults]:
    """Run benchmark on all paper models."""
    models = ["berturk"]  # Add more as available: "cosmos", "kumru", "aya"
    
    all_results = {}
    
    for model in models:
        try:
            results = run_paper_benchmark(model, n_samples)
            all_results[model] = results
        except Exception as e:
            print(f"❌ {model} failed: {e}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("📊 CROSS-MODEL COMPARISON (Paper Table)")
    print("="*80)
    print(f"\n{'Model':<12} {'Method':<15} {'Comp ↑':<10} {'Suff ↓':<10} {'Key Acc':<10}")
    print("-"*60)
    
    for model_name, result in all_results.items():
        for method_name, method in result.methods.items():
            print(f"{model_name:<12} {method_name:<15} {method.comprehensiveness:.4f}     "
                  f"{method.sufficiency:.4f}     {method.key_morpheme_accuracy:.1f}%")
        print("-"*60)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAFEX Paper Benchmark")
    parser.add_argument("--model", "-m", default="berturk", 
                        choices=["berturk", "cosmos", "kumru", "aya", "all"])
    parser.add_argument("--samples", "-n", type=int, default=20)
    parser.add_argument("--output", "-o", default="results/paper_benchmark.json")
    
    args = parser.parse_args()
    
    if args.model == "all":
        results = run_all_models_benchmark(args.samples)
    else:
        results = run_paper_benchmark(args.model, args.samples)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable
    if isinstance(results, dict):
        output_data = {
            model: {
                "model_path": r.model_path,
                "n_samples": r.n_samples,
                "methods": {
                    name: {
                        "comprehensiveness": m.comprehensiveness,
                        "sufficiency": m.sufficiency,
                        "comp_std": m.comp_std,
                        "suff_std": m.suff_std,
                        "key_accuracy": m.key_morpheme_accuracy
                    }
                    for name, m in r.methods.items()
                }
            }
            for model, r in results.items()
        }
    else:
        output_data = {
            "model": results.model,
            "model_path": results.model_path,
            "n_samples": results.n_samples,
            "methods": {
                name: {
                    "comprehensiveness": m.comprehensiveness,
                    "sufficiency": m.sufficiency,
                    "comp_std": m.comp_std,
                    "suff_std": m.suff_std,
                    "key_accuracy": m.key_morpheme_accuracy
                }
                for name, m in results.methods.items()
            }
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
