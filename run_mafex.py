"""
MAFEX Main Runner

Run MAFEX evaluation on trained models.

Usage:
    python run_mafex.py --model berturk --text "Gelemedim"
    python run_mafex.py --model berturk --eval --samples 100
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def setup_logging():
    """Setup logging configuration."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_single_explanation(args):
    """Run MAFEX on a single text."""
    logger = setup_logging()
    
    logger.info(f"Loading model: {args.model}")
    
    # Load model
    if args.model == "demo":
        from mafex.models import DemoModelWrapper
        wrapper = DemoModelWrapper()
    else:
        from mafex.models import get_model
        wrapper = get_model(args.model)
    
    wrapper.load()
    
    # Create MAFEX pipeline
    from mafex.projection import MAFEXPipeline
    
    mafex = MAFEXPipeline(
        wrapper.model,
        wrapper.tokenizer,
        lambda_causal=args.lambda_val,
        ig_steps=args.ig_steps
    )
    
    # Run explanation
    logger.info(f"Explaining: '{args.text}'")
    result = mafex.explain(args.text)
    
    # Print results
    print("\n" + "="*60)
    print(f"MAFEX EXPLANATION")
    print("="*60)
    print(f"\nInput: {result.text}")
    print(f"Target class: {result.target_class}")
    print(f"Model probability: {result.model_output:.4f}")
    print(f"lambda: {result.lambda_value}")
    
    print(f"\nMorphemes ({len(result.morphemes)}):")
    for i, (morph, score) in enumerate(zip(result.morphemes, result.final_attributions)):
        bar = "#" * int(abs(score) * 30) if abs(score) > 0.01 else ""
        sign = "+" if score > 0 else "-"
        print(f"   {i+1}. {morph:12} [{sign}] {score:+.4f} {bar}")
    
    print(f"\nTop attributed morphemes:")
    for morph, score in result.get_top_morphemes(3):
        print(f"   - {morph}: {score:.4f}")
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved to: {output_path}")
    
    return result


def run_evaluation(args):
    """Run evaluation on multiple samples."""
    logger = setup_logging()
    
    logger.info(f"Running evaluation with {args.samples} samples")
    
    # Load model
    if args.model == "demo":
        from mafex.models import DemoModelWrapper
        wrapper = DemoModelWrapper()
    else:
        from mafex.models import get_model
        wrapper = get_model(args.model)
    
    wrapper.load()
    
    # Create pipelines
    from mafex.projection import MAFEXPipeline
    from evaluation.metrics import ERASEREvaluator, BenchmarkRunner
    from evaluation.samples import get_test_samples
    
    mafex = MAFEXPipeline(
        wrapper.model,
        wrapper.tokenizer,
        lambda_causal=args.lambda_val,
        ig_steps=args.ig_steps
    )
    
    evaluator = ERASEREvaluator(
        wrapper.model,
        wrapper.tokenizer,
        top_k_ratios=[0.1, 0.2, 0.3]
    )
    
    # Get test samples from embedded dataset
    all_samples = get_test_samples()
    samples = all_samples[:min(args.samples, len(all_samples))]
    
    logger.info(f"Using {len(samples)} samples from embedded Turkish dataset")
    
    # Run benchmark
    runner = BenchmarkRunner(mafex, evaluator)
    results = runner.run(samples, show_progress=True)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nMAFEX:")
    print(f"   Comprehensiveness: {results['mafex']['comprehensiveness']:.4f} (+/-{results['mafex']['comp_std']:.4f})")
    print(f"   Sufficiency: {results['mafex']['sufficiency']:.4f} (+/-{results['mafex']['suff_std']:.4f})")
    
    print(f"\nToken Baseline:")
    print(f"   Comprehensiveness: {results['baseline']['comprehensiveness']:.4f} (+/-{results['baseline']['comp_std']:.4f})")
    print(f"   Sufficiency: {results['baseline']['sufficiency']:.4f} (+/-{results['baseline']['suff_std']:.4f})")
    
    print(f"\nImprovement:")
    print(f"   Comprehensiveness: +{results['improvement']['comp_gain']:.1f}%")
    
    print(f"\nSamples evaluated: {results['n_samples']}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results["timestamp"] = datetime.now().isoformat()
        results["model"] = args.model
        results["config"] = {
            "lambda": args.lambda_val,
            "ig_steps": args.ig_steps
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="MAFEX - Morpheme-Aligned Faithful Explanations"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="demo",
        choices=["demo", "berturk", "cosmos", "kumru", "aya"],
        help="Model to use"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to explain"
    )
    
    parser.add_argument(
        "--eval", "-e",
        action="store_true",
        help="Run evaluation mode"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="Number of samples for evaluation"
    )
    
    parser.add_argument(
        "--lambda-val", "-l",
        type=float,
        default=0.7,
        help="Lambda value for causal regularization (Eq. 5)"
    )
    
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=50,
        help="Number of Integrated Gradients steps"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for results"
    )
    
    args = parser.parse_args()
    
    if args.eval:
        run_evaluation(args)
    elif args.text:
        run_single_explanation(args)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
