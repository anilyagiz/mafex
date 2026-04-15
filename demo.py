"""
MAFEX Demo Script

Interactive demonstration of morpheme-aligned explanations.
Run this script to see MAFEX in action on Turkish text.

Usage:
    python demo.py --text "Gelemedim" --model berturk
    python demo.py --interactive
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header():
    """Print MAFEX header."""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███╗   ███╗ █████╗ ███████╗███████╗██╗  ██╗                ║
║   ████╗ ████║██╔══██╗██╔════╝██╔════╝╚██╗██╔╝                ║
║   ██╔████╔██║███████║█████╗  █████╗   ╚███╔╝                 ║
║   ██║╚██╔╝██║██╔══██║██╔══╝  ██╔══╝   ██╔██╗                 ║
║   ██║ ╚═╝ ██║██║  ██║██║     ███████╗██╔╝ ██╗                ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝                ║
║                                                               ║
║   Morpheme-Aligned Faithful Explanations                      ║
║   for Turkish NLP                                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def demo_morphology():
    """Demonstrate morphological analysis."""
    from mafex.morphology import MorphemeAnalyzer
    
    print("\n" + "="*60)
    print("📖 MORPHOLOGICAL ANALYSIS DEMO")
    print("="*60)
    
    analyzer = MorphemeAnalyzer()
    
    test_words = [
        ("yapamayacakmış", "reportedly, he will not be able to do"),
        ("gelemedim", "I could not come"),
        ("gözlükçü", "optician"),
        ("evlerden", "from the houses"),
        ("güzelleşiyor", "is becoming beautiful"),
        ("çalışkanlık", "diligence"),
    ]
    
    for word, meaning in test_words:
        analysis = analyzer.analyze_word(word)
        print(f"\n🔤 {word}")
        print(f"   Meaning: {meaning}")
        print(f"   Morphemes: {' + '.join(analysis.morpheme_surfaces)}")
        for m in analysis.morphemes:
            print(f"      • {m.surface} [{m.pos}]")


def demo_attribution_comparison():
    """Demonstrate MAFEX vs token-level attribution."""
    print("\n" + "="*60)
    print("🎯 ATTRIBUTION COMPARISON DEMO")
    print("="*60)
    
    # Use demo model for quick testing
    from mafex.models import DemoModelWrapper
    from mafex.projection import MAFEXPipeline, TokenBaselinePipeline
    
    # Load demo model
    print("\n⏳ Loading demo model...")
    wrapper = DemoModelWrapper()
    wrapper.load()
    
    model = wrapper.model
    tokenizer = wrapper.tokenizer
    
    # Create pipelines
    mafex = MAFEXPipeline(model, tokenizer, lambda_causal=0.7, ig_steps=20)
    baseline = TokenBaselinePipeline(model, tokenizer, ig_steps=20)
    
    # Test sentences
    test_sentences = [
        "Gelemedim",  # I could not come
        "Yapamayacağız",  # We will not be able to do
        "Çok güzelmiş",  # It was reportedly very beautiful
    ]
    
    for text in test_sentences:
        print(f"\n📝 Input: '{text}'")
        print("-" * 40)
        
        # MAFEX explanation
        try:
            mafex_result = mafex.explain(text)
            
            print("\n🔵 MAFEX (Morpheme-level):")
            for morph, score in mafex_result.get_top_morphemes(5):
                bar = "█" * int(abs(score) * 20)
                sign = "+" if score > 0 else "-"
                print(f"   {morph:12} [{sign}] {bar} ({score:.3f})")
            
            # Baseline
            baseline_result = baseline.explain(text)
            
            print("\n🔴 Token-IG (Baseline):")
            tokens = baseline_result["tokens"]
            attrs = baseline_result["attributions"]
            
            # Sort by importance
            sorted_idx = sorted(range(len(attrs)), key=lambda i: abs(attrs[i]), reverse=True)
            for i in sorted_idx[:5]:
                tok = tokens[i]
                score = attrs[i]
                bar = "█" * int(abs(score) * 20)
                sign = "+" if score > 0 else "-"
                print(f"   {tok:12} [{sign}] {bar} ({score:.3f})")
                
        except Exception as e:
            print(f"   ⚠️ Error: {e}")


def demo_interactive():
    """Interactive demo mode."""
    from mafex.morphology import MorphemeAnalyzer
    
    print("\n" + "="*60)
    print("🎮 INTERACTIVE MODE")
    print("="*60)
    print("Enter Turkish text to analyze. Type 'quit' to exit.\n")
    
    analyzer = MorphemeAnalyzer()
    
    while True:
        try:
            text = input("📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                continue
            
            print(f"\n{'='*40}")
            print("Morphological Analysis:")
            print(f"{'='*40}")
            
            analyses = analyzer.analyze_text(text)
            
            for analysis in analyses:
                print(f"\n  📌 {analysis.word}")
                print(f"     Morphemes: {' + '.join(analysis.morpheme_surfaces)}")
                for m in analysis.morphemes:
                    print(f"        • {m.surface} [{m.pos}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")


def run_full_demo():
    """Run complete demonstration."""
    print_header()
    
    # Demo 1: Morphology
    demo_morphology()
    
    # Demo 2: Attribution
    print("\n" + "="*60)
    print("⚡ Running attribution demo (this may take a moment)...")
    print("="*60)
    
    try:
        demo_attribution_comparison()
    except Exception as e:
        print(f"⚠️ Attribution demo failed: {e}")
        print("   (This is expected if BERT is not downloaded)")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="MAFEX Demo - Morpheme-Aligned Explanations for Turkish"
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to analyze"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--morphology", "-m",
        action="store_true",
        help="Demo morphological analysis only"
    )
    
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full demonstration"
    )
    
    args = parser.parse_args()
    
    print_header()
    
    if args.interactive:
        demo_interactive()
    elif args.morphology:
        demo_morphology()
    elif args.text:
        from mafex.morphology import MorphemeAnalyzer
        analyzer = MorphemeAnalyzer()
        
        print(f"\n📝 Analyzing: '{args.text}'\n")
        
        for analysis in analyzer.analyze_text(args.text):
            print(f"  📌 {analysis.word}")
            print(f"     → {' + '.join(analysis.morpheme_surfaces)}")
    else:
        run_full_demo()


if __name__ == "__main__":
    main()
