"""
MAFEX Production Benchmark

Runs comprehensive evaluation across all Turkish LLMs:
- BERTurk (Encoder)
- YTÜ-Cosmos (Decoder) 
- Kumru (Decoder)
- Aya-23 (Decoder)

Usage:
    python benchmark.py                    # All models
    python benchmark.py --model berturk    # Specific model
    python benchmark.py --quick            # Demo model only
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

warnings.filterwarnings("ignore")


def run_mafex_benchmark(model_name: str, samples: List[Dict], ig_steps: int = 20) -> Dict[str, Any]:
    """Run MAFEX benchmark on a single model."""
    from mafex.models import get_model
    from mafex.projection import MAFEXPipeline
    
    print(f"\n{'='*60}")
    print(f"🚀 Benchmarking: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load model
    try:
        wrapper = get_model(model_name)
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return {"model": model_name, "error": str(e)}
    
    # Create pipeline
    mafex = MAFEXPipeline(
        wrapper.model,
        wrapper.tokenizer,
        lambda_causal=0.7,
        ig_steps=ig_steps
    )
    
    # Run evaluation
    results = {
        "model": model_name,
        "model_path": wrapper.config.name,
        "n_samples": 0,
        "mafex_hits": 0,
        "baseline_hits": 0,
        "details": []
    }
    
    print(f"\n📊 Evaluating {len(samples)} samples...\n")
    
    for i, sample in enumerate(samples):
        text = sample["text"]
        key = sample["key_morpheme"]
        
        try:
            # Run MAFEX
            result = mafex.explain(text)
            
            # Get top morpheme
            top_morphs = result.get_top_morphemes(3)
            top_morph = top_morphs[0][0] if top_morphs else ""
            top_score = top_morphs[0][1] if top_morphs else 0
            
            # Get top token from baseline
            top_tok_idx = abs(result.token_attributions).argmax()
            top_token = result.tokens[top_tok_idx] if top_tok_idx < len(result.tokens) else ""
            
            # Check hits
            mafex_hit = key.lower() in top_morph.lower()
            baseline_hit = key.lower() in top_token.lower()
            
            results["mafex_hits"] += int(mafex_hit)
            results["baseline_hits"] += int(baseline_hit)
            results["n_samples"] += 1
            
            # Store detail
            results["details"].append({
                "text": text,
                "key": key,
                "mafex_top": top_morph,
                "mafex_score": float(top_score),
                "baseline_top": top_token,
                "mafex_hit": mafex_hit,
                "baseline_hit": baseline_hit
            })
            
            # Print progress
            m_mark = "✅" if mafex_hit else "❌"
            b_mark = "✅" if baseline_hit else "❌"
            print(f"  {i+1:2}. {text[:25]:25} | key='{key}' | MAFEX:{m_mark} Baseline:{b_mark}")
            
        except Exception as e:
            print(f"  {i+1:2}. {text[:25]:25} | ERROR: {e}")
    
    # Calculate metrics
    n = results["n_samples"]
    if n > 0:
        results["mafex_accuracy"] = results["mafex_hits"] / n * 100
        results["baseline_accuracy"] = results["baseline_hits"] / n * 100
        results["improvement"] = results["mafex_accuracy"] - results["baseline_accuracy"]
    
    return results


def print_summary(all_results: List[Dict]):
    """Print summary table of all results."""
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    print(f"\n{'Model':<15} {'MAFEX Acc':<12} {'Baseline Acc':<14} {'Improvement':<12}")
    print("-"*55)
    
    for r in all_results:
        if "error" in r:
            print(f"{r['model']:<15} ERROR: {r['error'][:30]}")
        else:
            mafex = r.get("mafex_accuracy", 0)
            baseline = r.get("baseline_accuracy", 0)
            improvement = r.get("improvement", 0)
            print(f"{r['model']:<15} {mafex:>6.1f}%      {baseline:>6.1f}%        +{improvement:>5.1f}%")
    
    print("\n" + "="*70)
    
    # Calculate averages
    valid = [r for r in all_results if "error" not in r and r["n_samples"] > 0]
    if valid:
        avg_mafex = sum(r["mafex_accuracy"] for r in valid) / len(valid)
        avg_baseline = sum(r["baseline_accuracy"] for r in valid) / len(valid)
        avg_improvement = sum(r["improvement"] for r in valid) / len(valid)
        
        print(f"\n📈 AVERAGE IMPROVEMENT: +{avg_improvement:.1f}%")
        print(f"   MAFEX: {avg_mafex:.1f}% vs Baseline: {avg_baseline:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="MAFEX Production Benchmark")
    parser.add_argument("--model", "-m", type=str, default="all",
                        choices=["all", "berturk", "cosmos", "kumru", "aya", "demo"],
                        help="Model to benchmark")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Quick mode with demo model only")
    parser.add_argument("--samples", "-n", type=int, default=20,
                        help="Number of samples")
    parser.add_argument("--output", "-o", type=str, default="results/benchmark.json",
                        help="Output file for results")
    parser.add_argument("--ig-steps", type=int, default=20,
                        help="Integrated Gradients steps")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🎯 MAFEX PRODUCTION BENCHMARK")
    print("   Morpheme-Aligned Faithful Explanations for Turkish")
    print("="*70)
    
    # Get samples
    from evaluation.samples import get_test_samples
    samples = get_test_samples()[:args.samples]
    print(f"\n📋 Using {len(samples)} embedded Turkish samples")
    
    # Determine models to run
    if args.quick:
        models = ["demo"]
    elif args.model == "all":
        models = ["berturk", "cosmos", "kumru", "aya"]
    else:
        models = [args.model]
    
    print(f"📦 Models: {', '.join(models)}")
    
    # Run benchmarks
    all_results = []
    for model in models:
        result = run_mafex_benchmark(model, samples, args.ig_steps)
        all_results.append(result)
    
    # Print summary
    print_summary(all_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_samples": len(samples),
            "ig_steps": args.ig_steps,
            "lambda_causal": 0.7
        },
        "results": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print()


if __name__ == "__main__":
    main()
