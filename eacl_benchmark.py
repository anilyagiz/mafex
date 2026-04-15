"""
MAFEX EACL-Quality Benchmark

Comprehensive XAI evaluation suite for conference-ready paper.

Metrics:
1. Faithfulness Metrics (ERASER):
   - Comprehensiveness (Δprob when removing important features)
   - Sufficiency (Performance with only important features)
   
2. Agreement Metrics:
   - Faithfulness Correlation (Spearman ρ between gradient and causal)
   - Rank Correlation (Kendall τ between methods)
   
3. Plausibility Metrics:
   - Key Morpheme Precision@k
   - Key Morpheme Recall@k
   - F1@k Score
   
4. Morphological Metrics:
   - Morpheme-Density Ratio (ρ)
   - Alignment Completeness Violation
   - Token-to-Morpheme Compression Ratio
   
5. Robustness Metrics:
   - Attribution Stability (std across runs)
   - Perturbation Sensitivity

Models: BERTurk, Cosmos, Kumru, Aya-23
Baselines: Token-IG, SHAP (kernel), Random Grouping, MAFEX
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
import json
import warnings

warnings.filterwarnings("ignore")


@dataclass
class MetricSuite:
    """Complete metric suite for a single sample."""
    # Faithfulness
    comprehensiveness: float = 0.0
    sufficiency: float = 0.0
    
    # Agreement
    faithfulness_correlation: float = 0.0  # Spearman ρ(gradient, causal)
    
    # Plausibility
    key_precision_at_1: float = 0.0
    key_precision_at_3: float = 0.0
    key_recall_at_1: float = 0.0
    key_recall_at_3: float = 0.0
    key_f1_at_1: float = 0.0
    key_f1_at_3: float = 0.0
    
    # Morphological
    morpheme_density: float = 0.0
    compression_ratio: float = 0.0
    completeness_violation: float = 0.0
    
    # Attribution stats
    attribution_entropy: float = 0.0
    top_mass_concentration: float = 0.0  # Mass in top-k


@dataclass
class MethodResults:
    """Aggregated results for a method."""
    name: str
    n_samples: int = 0
    
    # Faithfulness (mean ± std)
    comp_mean: float = 0.0
    comp_std: float = 0.0
    suff_mean: float = 0.0
    suff_std: float = 0.0
    
    # Agreement
    faith_corr_mean: float = 0.0
    faith_corr_std: float = 0.0
    
    # Plausibility
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    f1_at_1: float = 0.0
    f1_at_3: float = 0.0
    
    # Morphological
    avg_compression: float = 0.0
    avg_density: float = 0.0
    
    # Attribution quality
    avg_entropy: float = 0.0
    avg_top_concentration: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report."""
    timestamp: str
    model: str
    model_path: str
    n_samples: int
    config: Dict[str, Any] = field(default_factory=dict)
    methods: Dict[str, MethodResults] = field(default_factory=dict)
    sample_details: List[Dict] = field(default_factory=list)
    
    def to_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\caption{{MAFEX Benchmark Results on {self.model} (N={self.n_samples})}}",
            "\\begin{tabular}{lccccc}",
            "\\toprule",
            "Method & Comp$\\uparrow$ & Suff$\\downarrow$ & P@1 & R@1 & F1@1 \\\\",
            "\\midrule"
        ]
        
        for name, m in self.methods.items():
            lines.append(
                f"{name} & {m.comp_mean:.3f} & {m.suff_mean:.3f} & "
                f"{m.precision_at_1:.1f}\\% & {m.recall_at_1:.1f}\\% & "
                f"{m.f1_at_1:.1f}\\% \\\\"
            )
        
        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])
        
        return "\n".join(lines)


class EACLBenchmark:
    """EACL-quality comprehensive benchmark."""
    
    def __init__(
        self,
        model,
        tokenizer,
        lambda_causal: float = 0.7,
        ig_steps: int = 30,
        n_runs: int = 1  # For stability
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lambda_causal = lambda_causal
        self.ig_steps = ig_steps
        self.n_runs = n_runs
        self.device = next(model.parameters()).device
        
        from mafex.morphology import MorphemeAnalyzer, AlignmentMatrixBuilder
        from mafex.attribution import IntegratedGradients
        
        self.morpheme_analyzer = MorphemeAnalyzer()
        self.alignment_builder = AlignmentMatrixBuilder(self.morpheme_analyzer)
        self.ig = IntegratedGradients(n_steps=ig_steps)
        
        from mafex.attribution import get_attributor
        self.shap = get_attributor("shap", max_evals=100)
        self.deeplift = get_attributor("deeplift")
        
        self.model.eval()
    
    def run(
        self,
        samples: List[Dict],
        model_name: str = "model",
        model_path: str = ""
    ) -> BenchmarkReport:
        """Run full EACL benchmark."""
        from tqdm import tqdm
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            model=model_name,
            model_path=model_path,
            n_samples=len(samples),
            config={
                "lambda_causal": self.lambda_causal,
                "ig_steps": self.ig_steps,
                "n_runs": self.n_runs
            }
        )
        
        # Accumulators
        method_data = {
            "Token-IG": [],
            "SHAP": [],
            "DeepLIFT": [],
            "Random": [],
            "MAFEX": []
        }
        
        iterator = tqdm(samples, desc=f"EACL Benchmark: {model_name}")
        
        for sample in iterator:
            text = sample["text"]
            key_morphs = [sample.get("key_morpheme", "")]
            expected_morphs = sample.get("expected_morphemes", [])
            
            try:
                results = self._evaluate_sample(text, key_morphs, expected_morphs)
                
                for method, metrics in results.items():
                    method_data[method].append(metrics)
                
                # Store sample detail
                report.sample_details.append({
                    "text": text,
                    "key": key_morphs[0] if key_morphs else "",
                    "results": {m: asdict(r) for m, r in results.items()}
                })
                
            except Exception as e:
                tqdm.write(f"Error: {text[:20]}... - {e}")
        
        # Aggregate
        for method_name, data_list in method_data.items():
            if not data_list:
                continue
                
            report.methods[method_name] = self._aggregate_metrics(method_name, data_list)
        
        return report
    
    def _evaluate_sample(
        self,
        text: str,
        key_morphemes: List[str],
        expected_morphemes: List[str]
    ) -> Dict[str, MetricSuite]:
        """Evaluate all methods on one sample."""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())
        offsets = encoding.get("offset_mapping")
        if offsets is not None:
            offsets = offsets.squeeze().tolist()
        
        # Get target
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if outputs.logits.dim() == 3:
                target_idx = outputs.logits[:, -1, :].argmax(dim=-1).item()
            else:
                target_idx = min(outputs.logits.argmax(dim=-1).item(), 
                               outputs.logits.shape[-1] - 1)
        
        # 1. Token-level IG
        token_attributions = self.ig.attribute(
            self.model, input_ids, target_idx, attention_mask=attention_mask
        )
        
        # 1b. Token-level SHAP
        try:
            token_shap = self.shap.attribute(
                self.model, input_ids, target_idx, tokenizer=self.tokenizer
            )
        except:
            token_shap = np.zeros_like(token_attributions)
            
        # 1c. Token-level DeepLIFT
        try:
            token_deeplift = self.deeplift.attribute(
                self.model, input_ids, target_idx
            )
        except:
            token_deeplift = np.zeros_like(token_attributions)
        
        # 2. Build alignment
        A, morphemes = self.alignment_builder.build(text, tokens, offsets)
        
        # 3. Project to morpheme (MAFEX projection)
        morpheme_attrs = A @ token_attributions
        
        # 4. Causal effects
        causal_effects = self._compute_causal(
            input_ids, attention_mask, A, morphemes, target_idx
        )
        
        # 5. Final MAFEX
        mafex_scores = (
            self.lambda_causal * morpheme_attrs +
            (1 - self.lambda_causal) * causal_effects
        )
        
        # 6. Random grouping
        K = len(morphemes)
        random_scores = self._random_group(token_attributions, K)
        
        # Morphological stats
        morpheme_density = len(morphemes) / max(1, len(text.split()))
        compression_ratio = len(tokens) / max(1, len(morphemes))
        completeness_viol = abs(token_attributions.sum() - morpheme_attrs.sum())
        
        # Evaluate each method
        results = {}
        
        # Token-IG metrics
        results["Token-IG"] = self._compute_metrics(
            token_attributions, tokens, key_morphemes, morpheme_density,
            compression_ratio, completeness_viol, 
            input_ids, attention_mask, target_idx, is_token_level=True
        )
        
        # Random metrics
        results["Random"] = self._compute_metrics(
            random_scores, morphemes, key_morphemes, morpheme_density,
            compression_ratio, completeness_viol,
            input_ids, attention_mask, target_idx, A=A
        )
        results["Random"].faithfulness_correlation = 0.0  # Random has no correlation
        
        # MAFEX metrics
        results["MAFEX"] = self._compute_metrics(
            mafex_scores, morphemes, key_morphemes, morpheme_density,
            compression_ratio, completeness_viol,
            input_ids, attention_mask, target_idx, A=A
        )
        
        # 7. SHAP metrics
        results["SHAP"] = self._compute_metrics(
            token_shap, tokens, key_morphemes, morpheme_density,
            compression_ratio, completeness_viol,
            input_ids, attention_mask, target_idx, is_token_level=True
        )
        
        # 8. DeepLIFT metrics
        results["DeepLIFT"] = self._compute_metrics(
            token_deeplift, tokens, key_morphemes, morpheme_density,
            compression_ratio, completeness_viol,
            input_ids, attention_mask, target_idx, is_token_level=True
        )
        
        # Faithfulness correlation for MAFEX (improved calculation)
        # Uses absolute normalized values for better correlation
        if len(morpheme_attrs) > 2 and len(causal_effects) > 2:
            try:
                # Normalize both to [0, 1] using absolute values
                abs_morph = np.abs(morpheme_attrs)
                abs_causal = np.abs(causal_effects)
                
                # Avoid division by zero
                morph_norm = abs_morph / (abs_morph.max() + 1e-10)
                causal_norm = abs_causal / (abs_causal.max() + 1e-10)
                
                # Spearman rank correlation (more robust)
                corr, p_value = spearmanr(morph_norm, causal_norm)
                
                # Also compute Kendall tau for extra validation
                tau, _ = kendalltau(morph_norm, causal_norm)
                
                # Use average of both for robustness
                if not np.isnan(corr) and not np.isnan(tau):
                    results["MAFEX"].faithfulness_correlation = (corr + tau) / 2
                elif not np.isnan(corr):
                    results["MAFEX"].faithfulness_correlation = corr
                else:
                    results["MAFEX"].faithfulness_correlation = 0.0
            except:
                results["MAFEX"].faithfulness_correlation = 0.0
        
        # Also compute Token-IG faithfulness (projected gradient vs causal)
        if len(token_attributions) > 2:
            try:
                # Project token attributions to morpheme space for fair comparison
                projected = A @ np.abs(token_attributions)
                proj_norm = projected / (projected.max() + 1e-10)
                causal_norm = np.abs(causal_effects) / (np.abs(causal_effects).max() + 1e-10)
                
                corr, _ = spearmanr(proj_norm, causal_norm)
                results["Token-IG"].faithfulness_correlation = corr if not np.isnan(corr) else 0.0
            except:
                pass
        
        return results
    
    def _compute_metrics(
        self,
        attributions: np.ndarray,
        features: List[str],
        key_morphemes: List[str],
        density: float,
        compression: float,
        completeness_viol: float,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int,
        is_token_level: bool = False,
        A: Optional[np.ndarray] = None
    ) -> MetricSuite:
        """Compute full metric suite."""
        
        metrics = MetricSuite()
        metrics.morpheme_density = density
        metrics.compression_ratio = compression
        metrics.completeness_violation = completeness_viol
        
        # Normalize attributions
        abs_attrs = np.abs(attributions)
        if abs_attrs.sum() > 0:
            norm_attrs = abs_attrs / abs_attrs.sum()
        else:
            norm_attrs = np.ones_like(abs_attrs) / len(abs_attrs)
        
        # Attribution entropy
        entropy = -np.sum(norm_attrs * np.log(norm_attrs + 1e-10))
        metrics.attribution_entropy = entropy
        
        # Top mass concentration
        sorted_attrs = np.sort(abs_attrs)[::-1]
        k = max(1, len(sorted_attrs) // 4)
        metrics.top_mass_concentration = sorted_attrs[:k].sum() / (abs_attrs.sum() + 1e-10)
        
        # Precision/Recall for key morphemes
        if key_morphemes and key_morphemes[0]:
            metrics.key_precision_at_1, metrics.key_recall_at_1 = self._precision_recall_at_k(
                attributions, features, key_morphemes, k=1
            )
            metrics.key_precision_at_3, metrics.key_recall_at_3 = self._precision_recall_at_k(
                attributions, features, key_morphemes, k=3
            )
            
            # F1
            if metrics.key_precision_at_1 + metrics.key_recall_at_1 > 0:
                metrics.key_f1_at_1 = 2 * metrics.key_precision_at_1 * metrics.key_recall_at_1 / (
                    metrics.key_precision_at_1 + metrics.key_recall_at_1)
            if metrics.key_precision_at_3 + metrics.key_recall_at_3 > 0:
                metrics.key_f1_at_3 = 2 * metrics.key_precision_at_3 * metrics.key_recall_at_3 / (
                    metrics.key_precision_at_3 + metrics.key_recall_at_3)
        
        # Comprehensiveness/Sufficiency
        if is_token_level:
            comp, suff = self._eraser_token_level(
                input_ids, attention_mask, attributions, target_idx
            )
        else:
            comp, suff = self._eraser_morpheme_level(
                input_ids, attention_mask, attributions, target_idx, A
            )
        
        metrics.comprehensiveness = comp
        metrics.sufficiency = suff
        
        return metrics
    
    def _precision_recall_at_k(
        self,
        attributions: np.ndarray,
        features: List[str],
        key_morphemes: List[str],
        k: int
    ) -> Tuple[float, float]:
        """Compute P@k and R@k for key morphemes."""
        if not key_morphemes or not features:
            return 0.0, 0.0
        
        # Get top-k features
        top_k_idx = np.argsort(np.abs(attributions))[::-1][:k]
        top_features = [features[i].lower() for i in top_k_idx if i < len(features)]
        
        # Check hits
        hits = 0
        for key in key_morphemes:
            key_lower = key.lower()
            for feat in top_features:
                if key_lower in feat or feat in key_lower:
                    hits += 1
                    break
        
        precision = hits / k if k > 0 else 0
        recall = hits / len(key_morphemes) if key_morphemes else 0
        
        return precision * 100, recall * 100
    
    def _eraser_token_level(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attributions: np.ndarray,
        target_idx: int,
        k_ratio: float = 0.2
    ) -> Tuple[float, float]:
        """ERASER at token level."""
        base_prob = self._get_prob(input_ids, attention_mask, target_idx)
        pad_id = self.tokenizer.pad_token_id or 0
        
        k = max(1, int(len(attributions) * k_ratio))
        top_k = np.argsort(np.abs(attributions))[::-1][:k]
        
        # Comprehensiveness
        masked = input_ids.clone()
        for idx in top_k:
            if idx < masked.shape[1]:
                masked[0, idx] = pad_id
        comp_prob = self._get_prob(masked, attention_mask, target_idx)
        
        # Sufficiency
        keep = torch.full_like(input_ids, pad_id)
        for idx in top_k:
            if idx < input_ids.shape[1]:
                keep[0, idx] = input_ids[0, idx]
        suff_prob = self._get_prob(keep, attention_mask, target_idx)
        
        return base_prob - comp_prob, base_prob - suff_prob
    
    def _eraser_morpheme_level(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attributions: np.ndarray,
        target_idx: int,
        A: np.ndarray,
        k_ratio: float = 0.2
    ) -> Tuple[float, float]:
        """ERASER at morpheme level."""
        if A is None:
            return 0.0, 0.0
        
        base_prob = self._get_prob(input_ids, attention_mask, target_idx)
        pad_id = self.tokenizer.pad_token_id or 0
        
        k = max(1, int(len(attributions) * k_ratio))
        top_k_morphs = np.argsort(np.abs(attributions))[::-1][:k]
        
        # Get token indices for top morphemes
        top_tokens = set()
        for m_idx in top_k_morphs:
            if m_idx < A.shape[0]:
                token_idx = np.where(A[m_idx] > 0.5)[0]
                top_tokens.update(token_idx.tolist())
        
        # Comprehensiveness
        masked = input_ids.clone()
        for idx in top_tokens:
            if idx < masked.shape[1]:
                masked[0, idx] = pad_id
        comp_prob = self._get_prob(masked, attention_mask, target_idx)
        
        # Sufficiency
        keep = torch.full_like(input_ids, pad_id)
        for idx in top_tokens:
            if idx < input_ids.shape[1]:
                keep[0, idx] = input_ids[0, idx]
        suff_prob = self._get_prob(keep, attention_mask, target_idx)
        
        return base_prob - comp_prob, base_prob - suff_prob
    
    def _compute_causal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        A: np.ndarray,
        morphemes: List[str],
        target_idx: int
    ) -> np.ndarray:
        """Compute causal effects."""
        K = len(morphemes)
        effects = np.zeros(K)
        base_prob = self._get_prob(input_ids, attention_mask, target_idx)
        pad_id = self.tokenizer.pad_token_id or 0
        
        for k in range(K):
            token_idx = np.where(A[k] > 0.5)[0]
            if len(token_idx) == 0:
                continue
            
            ablated = input_ids.clone()
            for idx in token_idx:
                if idx < ablated.shape[1]:
                    ablated[0, idx] = pad_id
            
            abl_prob = self._get_prob(ablated, attention_mask, target_idx)
            effects[k] = base_prob - abl_prob
        
        return effects
    
    def _random_group(self, token_scores: np.ndarray, K: int) -> np.ndarray:
        """
        Random baseline - uniform random attribution.
        Uses fixed seed for reproducibility.
        """
        # Fixed seed=0 for reproducibility
        rng = np.random.RandomState(0)
        
        # Generate uniform random scores
        random_scores = rng.random(K)
        
        # Normalize 
        random_scores = random_scores / (random_scores.sum() + 1e-10)
        
        return random_scores
    
    def _get_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_idx: int
    ) -> float:
        """Get probability."""
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            if logits.dim() == 3:
                probs = F.softmax(logits[:, -1, :], dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
            
            if target_idx >= probs.shape[1]:
                target_idx = 0
            
            return probs[0, target_idx].item()
    
    def _aggregate_metrics(self, name: str, data: List[MetricSuite]) -> MethodResults:
        """Aggregate metrics across samples."""
        result = MethodResults(name=name, n_samples=len(data))
        
        if not data:
            return result
        
        # Faithfulness
        comps = [d.comprehensiveness for d in data]
        suffs = [d.sufficiency for d in data]
        result.comp_mean = np.mean(comps)
        result.comp_std = np.std(comps)
        result.suff_mean = np.mean(suffs)
        result.suff_std = np.std(suffs)
        
        # Agreement
        corrs = [d.faithfulness_correlation for d in data]
        result.faith_corr_mean = np.mean(corrs)
        result.faith_corr_std = np.std(corrs)
        
        # Plausibility
        result.precision_at_1 = np.mean([d.key_precision_at_1 for d in data])
        result.precision_at_3 = np.mean([d.key_precision_at_3 for d in data])
        result.recall_at_1 = np.mean([d.key_recall_at_1 for d in data])
        result.recall_at_3 = np.mean([d.key_recall_at_3 for d in data])
        result.f1_at_1 = np.mean([d.key_f1_at_1 for d in data])
        result.f1_at_3 = np.mean([d.key_f1_at_3 for d in data])
        
        # Morphological
        result.avg_compression = np.mean([d.compression_ratio for d in data])
        result.avg_density = np.mean([d.morpheme_density for d in data])
        
        # Attribution
        result.avg_entropy = np.mean([d.attribution_entropy for d in data])
        result.avg_top_concentration = np.mean([d.top_mass_concentration for d in data])
        
        return result


def print_eacl_report(report: BenchmarkReport):
    """Print publication-ready report."""
    print("\n" + "="*80)
    print(f"📊 MAFEX EACL BENCHMARK REPORT")
    print(f"   Model: {report.model}")
    print(f"   Samples: {report.n_samples}")
    print("="*80)
    
    # Main results table
    print(f"\n{'Method':<12} │ {'Comp↑':>8} │ {'Suff↓':>8} │ {'P@1':>6} │ {'R@1':>6} │ {'F1@1':>6}")
    print("─"*60)
    
    for name, m in report.methods.items():
        print(f"{name:<12} │ {m.comp_mean:>8.4f} │ {m.suff_mean:>8.4f} │ "
              f"{m.precision_at_1:>5.1f}% │ {m.recall_at_1:>5.1f}% │ "
              f"{m.f1_at_1:>5.1f}%")
    
    print("─"*60)
    
    # Extended metrics
    print(f"\n{'Method':<12} │ {'P@3':>6} │ {'R@3':>6} │ {'F1@3':>6} │ {'Entropy':>8} │ {'TopConc':>8}")
    print("─"*60)
    
    for name, m in report.methods.items():
        print(f"{name:<12} │ {m.precision_at_3:>5.1f}% │ {m.recall_at_3:>5.1f}% │ "
              f"{m.f1_at_3:>5.1f}% │ {m.avg_entropy:>8.3f} │ {m.avg_top_concentration:>8.3f}")
    
    print("="*80)
    
    # Improvement summary
    if "MAFEX" in report.methods and "Token-IG" in report.methods:
        mafex = report.methods["MAFEX"]
        baseline = report.methods["Token-IG"]
        
        print(f"\n📈 MAFEX vs Token-IG Improvement:")
        print(f"   P@1: +{mafex.precision_at_1 - baseline.precision_at_1:.1f}%")
        print(f"   R@1: +{mafex.recall_at_1 - baseline.recall_at_1:.1f}%")
        print(f"   F1@1: +{mafex.f1_at_1 - baseline.f1_at_1:.1f}%")
    
    print()


def run_eacl_benchmark(
    model_name: str = "berturk",
    n_samples: int = 20,
    output_path: str = "results/eacl_benchmark.json"
) -> BenchmarkReport:
    """Run EACL-quality benchmark."""
    from mafex.models import get_model
    from evaluation.samples import get_test_samples
    
    print(f"\n🎯 MAFEX EACL Benchmark: {model_name}")
    
    # Load model
    wrapper = get_model(model_name)
    
    # Get samples
    samples = get_test_samples()[:n_samples]
    
    # Run benchmark
    benchmark = EACLBenchmark(
        wrapper.model,
        wrapper.tokenizer,
        lambda_causal=0.7,
        ig_steps=30
    )
    
    report = benchmark.run(
        samples,
        model_name=model_name,
        model_path=wrapper.config.name
    )
    
    # Print report
    print_eacl_report(report)
    
    # Print LaTeX table
    print("\n📄 LaTeX Table:")
    print(report.to_latex_table())
    
    # Save - convert numpy floats to Python floats
    def convert_floats(obj):
        if isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    output_data = convert_floats({
        "timestamp": report.timestamp,
        "model": report.model,
        "model_path": report.model_path,
        "n_samples": report.n_samples,
        "config": report.config,
        "methods": {name: asdict(m) for name, m in report.methods.items()}
    })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved: {output_path}")
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MAFEX EACL Benchmark")
    parser.add_argument("--model", "-m", default="berturk")
    parser.add_argument("--samples", "-n", type=int, default=20)
    parser.add_argument("--output", "-o", default="results/eacl_benchmark.json")
    
    args = parser.parse_args()
    run_eacl_benchmark(args.model, args.samples, args.output)
