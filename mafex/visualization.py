"""
Visualization utilities for MAFEX explanations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path


def create_attribution_heatmap(
    morphemes: List[str],
    attributions: np.ndarray,
    title: str = "MAFEX Attribution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 3),
    cmap: str = "RdYlGn"
) -> plt.Figure:
    """
    Create heatmap visualization of morpheme attributions.
    
    Args:
        morphemes: List of morpheme strings
        attributions: Attribution scores
        title: Figure title
        save_path: Optional path to save figure
        figsize: Figure size
        cmap: Colormap name
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize attributions to [-1, 1]
    max_abs = max(abs(attributions.min()), abs(attributions.max()))
    if max_abs > 0:
        norm_attr = attributions / max_abs
    else:
        norm_attr = attributions
    
    # Create heatmap data
    data = norm_attr.reshape(1, -1)
    
    # Plot
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(morphemes)))
    ax.set_xticklabels(morphemes, rotation=45, ha='right', fontsize=10)
    ax.set_yticks([])
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.2)
    cbar.set_label('Attribution Score', fontsize=10)
    
    # Title
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comparison_plot(
    text: str,
    mafex_morphemes: List[str],
    mafex_attributions: np.ndarray,
    baseline_tokens: List[str],
    baseline_attributions: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create side-by-side comparison of MAFEX vs baseline.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    
    # Baseline (top)
    ax1 = axes[0]
    colors1 = plt.cm.Reds(np.abs(baseline_attributions) / (np.abs(baseline_attributions).max() + 1e-8))
    bars1 = ax1.bar(range(len(baseline_tokens)), np.abs(baseline_attributions), color=colors1)
    ax1.set_xticks(range(len(baseline_tokens)))
    ax1.set_xticklabels(baseline_tokens, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('|Attribution|')
    ax1.set_title('Token-level IG (Baseline)', fontweight='bold')
    ax1.set_xlim(-0.5, len(baseline_tokens) - 0.5)
    
    # MAFEX (bottom)
    ax2 = axes[1]
    colors2 = plt.cm.Blues(np.abs(mafex_attributions) / (np.abs(mafex_attributions).max() + 1e-8))
    bars2 = ax2.bar(range(len(mafex_morphemes)), np.abs(mafex_attributions), color=colors2)
    ax2.set_xticks(range(len(mafex_morphemes)))
    ax2.set_xticklabels(mafex_morphemes, rotation=45, ha='right', fontsize=10, fontweight='bold')
    ax2.set_ylabel('|Attribution|')
    ax2.set_title('MAFEX (Morpheme-aligned)', fontweight='bold')
    ax2.set_xlim(-0.5, len(mafex_morphemes) - 0.5)
    
    # Main title
    fig.suptitle(f'Input: "{text}"', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_paper_figure_1(
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Recreate Figure 1 from the paper - The Fidelity Bottleneck.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'hspace': 0.4})
    
    # Token space (noisy)
    tokens = ['yap', '##ama', '##yacak', '##mış']
    token_scores = [0.1, 0.6, 0.2, 0.1]
    
    ax1 = axes[0]
    colors = plt.cm.Reds([s * 0.8 + 0.2 for s in token_scores])
    ax1.bar(range(len(tokens)), token_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xticks(range(len(tokens)))
    ax1.set_xticklabels(tokens, fontsize=11, fontfamily='monospace')
    ax1.set_ylabel('IG Score', fontsize=10)
    ax1.set_title('Token Space (Noisy)', fontsize=12, color='red', fontweight='bold')
    ax1.set_ylim(0, 0.8)
    
    # Add score labels
    for i, (t, s) in enumerate(zip(tokens, token_scores)):
        ax1.annotate(f'{s}', (i, s + 0.03), ha='center', fontsize=9)
    
    # Morpheme space (faithful)
    morphemes = ['yap', '-ma', '-yacak', '-mış']
    morph_labels = ['Root', 'NEGATION', 'Future', 'Evidential']
    morph_scores = [0.05, 0.9, 0.05, 0.0]
    
    ax2 = axes[1]
    colors = plt.cm.Greens([s * 0.8 + 0.1 for s in morph_scores])
    ax2.bar(range(len(morphemes)), morph_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(morphemes)))
    ax2.set_xticklabels([f'{m}\n({l})' for m, l in zip(morphemes, morph_labels)], fontsize=10)
    ax2.set_ylabel('MAFEX Score', fontsize=10)
    ax2.set_title('Morpheme Space (Faithful)', fontsize=12, color='blue', fontweight='bold')
    ax2.set_ylim(0, 1.0)
    
    # Highlight negation
    ax2.patches[1].set_edgecolor('darkgreen')
    ax2.patches[1].set_linewidth(3)
    
    # Add projection annotation
    fig.text(0.5, 0.48, '↓  Morphological Projection  ↓', 
             ha='center', va='center', fontsize=11, 
             fontweight='bold', color='gray',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_benchmark_figure(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create benchmark comparison figure (like Figure 3 in paper).
    """
    models = list(results.keys())
    
    # Extract scores
    token_ig = [results[m].get('token_ig', 0.4) for m in models]
    random = [results[m].get('random', 0.5) for m in models]
    mafex = [results[m].get('mafex', 0.7) for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width, token_ig, width, label='Token-IG', color='#ff6b6b', edgecolor='black')
    bars2 = ax.bar(x, random, width, label='Random', color='#868e96', edgecolor='black')
    bars3 = ax.bar(x + width, mafex, width, label='MAFEX', color='#4dabf7', edgecolor='black')
    
    # Labels and title
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Comprehensiveness Score (↑)', fontsize=12)
    ax.set_title('Cross-Model Faithfulness Evaluation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def format_html_explanation(
    morphemes: List[str],
    attributions: np.ndarray,
    title: str = "MAFEX Explanation"
) -> str:
    """
    Generate HTML visualization of attribution.
    """
    # Normalize for color intensity
    max_abs = max(abs(attributions.min()), abs(attributions.max()), 1e-8)
    norm_scores = attributions / max_abs
    
    html_parts = [f"<h3>{title}</h3><p>"]
    
    for morph, score in zip(morphemes, norm_scores):
        if score > 0:
            # Positive: green
            intensity = int(score * 200)
            color = f"rgb(0, {100 + intensity}, 0)"
        else:
            # Negative: red
            intensity = int(abs(score) * 200)
            color = f"rgb({100 + intensity}, 0, 0)"
        
        html_parts.append(
            f'<span style="background-color: {color}; color: white; '
            f'padding: 2px 6px; margin: 2px; border-radius: 3px; '
            f'font-family: monospace;">{morph}</span>'
        )
    
    html_parts.append("</p>")
    
    return "".join(html_parts)


if __name__ == "__main__":
    # Generate example figures
    print("Generating example figures...")
    
    Path("figures").mkdir(exist_ok=True)
    
    # Figure 1
    fig1 = create_paper_figure_1("figures/figure_1_fidelity_bottleneck.png")
    print("✓ Saved figure_1_fidelity_bottleneck.png")
    
    # Benchmark figure
    results = {
        "BERTurk": {"token_ig": 0.42, "random": 0.50, "mafex": 0.68},
        "Cosmos": {"token_ig": 0.39, "random": 0.47, "mafex": 0.65},
        "Kumru": {"token_ig": 0.45, "random": 0.53, "mafex": 0.71},
        "Aya-23": {"token_ig": 0.41, "random": 0.49, "mafex": 0.69},
    }
    
    fig2 = create_benchmark_figure(results, "figures/figure_3_benchmark.png")
    print("✓ Saved figure_3_benchmark.png")
    
    print("\nAll figures saved to figures/ directory")
    plt.show()
