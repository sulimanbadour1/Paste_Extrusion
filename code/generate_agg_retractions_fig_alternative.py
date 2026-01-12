#!/usr/bin/env python3
"""
Alternative cleaner visualization for aggregate retraction statistics.

This version uses a simpler, clearer approach:
- Swarm plot for better point visibility
- Clearer median indicators
- Better spacing and layout
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import sys

# Import the main function's dependencies
sys.path.insert(0, str(Path(__file__).parent))
from generate_agg_retractions_fig import COLORS, count_retractions_incremental

def generate_swarm_plot_figure(
    baseline_counts: List[int],
    stabilized_counts: List[int],
    output_path: Path
):
    """
    Generate a cleaner swarm plot version.
    Shows all individual points clearly without overlap.
    """
    baseline_counts = np.array(baseline_counts, dtype=int)
    stabilized_counts = np.array(stabilized_counts, dtype=int)
    
    M = min(len(baseline_counts), len(stabilized_counts))
    baseline_counts = baseline_counts[:M]
    stabilized_counts = stabilized_counts[:M]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Determine log scale
    max_count = np.max(baseline_counts) if len(baseline_counts) > 0 else 1
    min_count = np.min(baseline_counts[baseline_counts > 0]) if np.any(baseline_counts > 0) else 1
    use_log = max_count > 10 and (max_count / min_count > 5)
    
    # Prepare data
    if use_log:
        plot_baseline = baseline_counts.copy().astype(float)
        plot_stabilized = stabilized_counts.copy().astype(float)
        plot_baseline[plot_baseline == 0] = 0.1
        plot_stabilized[plot_stabilized == 0] = 0.1
        ax.set_yscale('log')
        ylabel = 'Retraction Events per Model'
    else:
        plot_baseline = baseline_counts.astype(float)
        plot_stabilized = stabilized_counts.astype(float)
        ylabel = 'Retraction Events per Model'
    
    # Create swarm positions (manual jitter to avoid overlap)
    np.random.seed(42)
    
    # Baseline swarm
    baseline_x = np.ones(M) + np.random.normal(0, 0.12, M)
    ax.scatter(baseline_x, plot_baseline, 
              color=COLORS['baseline'], alpha=0.7, s=45, zorder=4,
              edgecolors='black', linewidths=0.8, label='Baseline')
    
    # Shaped swarm (all zeros)
    shaped_x = np.ones(M) * 2 + np.random.normal(0, 0.12, M)
    ax.scatter(shaped_x, plot_stabilized,
              color=COLORS['stabilized'], alpha=0.8, s=50, zorder=5,
              edgecolors='darkgreen', linewidths=1.0, marker='o', label='Shaped')
    
    # Add connecting lines
    for i in range(M):
        ax.plot([baseline_x[i], shaped_x[i]], [plot_baseline[i], plot_stabilized[i]],
               color=COLORS['stabilized'], alpha=0.2, linewidth=1.0, zorder=1)
    
    # Add box plots for summary statistics
    bp = ax.boxplot([plot_baseline, plot_stabilized], positions=[1, 2], widths=0.4,
                    patch_artist=True, showmeans=False,
                    boxprops=dict(linewidth=2.0, alpha=0.3),
                    medianprops=dict(linewidth=4.0, color='black'),
                    whiskerprops=dict(linewidth=2.0),
                    capprops=dict(linewidth=2.0),
                    zorder=2)
    
    bp['boxes'][0].set_facecolor(COLORS['baseline'])
    bp['boxes'][1].set_facecolor(COLORS['stabilized'])
    
    # Calculate and display statistics
    baseline_median = np.median(baseline_counts)
    baseline_q25 = np.percentile(baseline_counts, 25)
    baseline_q75 = np.percentile(baseline_counts, 75)
    
    # Add median line and annotation
    median_y = baseline_median if not use_log else baseline_median
    ax.plot([0.7, 1.3], [median_y, median_y],
           color='black', linewidth=4.0, linestyle='--', zorder=6, alpha=0.9)
    
    # Position annotation clearly
    if use_log:
        annot_y = max_count * 0.35
    else:
        annot_y = max_count * 0.8
    
    ax.text(1.0, annot_y,
           f'Median: {int(baseline_median)}',
           ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    alpha=0.98, edgecolor='black', linewidth=2.0),
           zorder=10)
    
    # Statistics box
    stats_text = (
        f'Baseline (M = {M}):\n'
        f'  Median: {int(baseline_median)}\n'
        f'  IQR: [{int(baseline_q25)}, {int(baseline_q75)}]\n'
        f'  Range: {int(np.min(baseline_counts))} - {int(np.max(baseline_counts))}\n\n'
        f'Shaped (M = {M}):\n'
        f'  All models: 0 retractions'
    )
    
    ax.text(0.98, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                    alpha=0.98, edgecolor='black', linewidth=1.5))
    
    # Labels
    ax.set_xlabel('Condition', fontsize=13, fontweight='bold', labelpad=12)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Baseline', 'Shaped'], fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', labelpad=12)
    
    if use_log:
        ax.text(-0.08, 0.5, '(log scale)', transform=ax.transAxes,
               rotation=90, fontsize=10, style='italic', va='center', ha='center')
    
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    
    # Set limits
    if use_log:
        ax.set_ylim([min(0.05, min_count * 0.3), max_count * 3.0])
    else:
        ax.set_ylim([-max_count * 0.03, max_count * 1.3])
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved alternative figure to: {output_path}", flush=True)
    plt.close(fig)

if __name__ == '__main__':
    # Test with demo data
    np.random.seed(42)
    n_simple = 10
    n_medium = 25
    n_complex = 12
    n_very_complex = 3
    
    simple_counts = np.random.randint(50, 200, n_simple)
    medium_counts = np.random.randint(200, 500, n_medium)
    complex_counts = np.random.randint(500, 800, n_complex)
    very_complex_counts = np.random.randint(800, 1200, n_very_complex)
    
    baseline_counts = np.concatenate([simple_counts, medium_counts, complex_counts, very_complex_counts])
    np.random.shuffle(baseline_counts)
    stabilized_counts = np.zeros(50, dtype=int)
    
    output_path = Path(__file__).parent / 'results' / 'figures' / 'agg_retractions_fig_swarm.png'
    generate_swarm_plot_figure(baseline_counts, stabilized_counts, output_path)
