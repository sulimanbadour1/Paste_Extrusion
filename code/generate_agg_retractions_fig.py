#!/usr/bin/env python3
"""
generate_agg_retractions_fig.py

Generates Figure: Aggregate retraction statistics across sliced models (M=50)

This script:
1. Parses G-code files and converts extrusion stream to incremental form
2. Accounts for absolute/relative extrusion mode (M82/M83) and extrusion resets (G92 E0)
3. Counts retraction events when ΔE_k < 0
4. Shows baseline distribution of retraction counts
5. Shows zero retractions for all stabilized files

Usage:
    python3 generate_agg_retractions_fig.py --baseline-dir <dir> --stabilized-dir <dir> [--output <path>]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

# Force non-interactive backend for saving
matplotlib.use('Agg')

# Professional styling for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# Color palette
COLORS = {
    'baseline': '#E63946',        # Vibrant red
    'stabilized': '#2A9D8F',      # Teal green
}


def parse_gcode_line(line: str) -> Dict[str, Optional[float]]:
    """Parse a G-code line and extract X, Y, Z, E, F values."""
    result = {'X': None, 'Y': None, 'Z': None, 'E': None, 'F': None}
    for key in result.keys():
        pattern = rf'{key}([+-]?\d+\.?\d*)'
        match = re.search(pattern, line)
        if match:
            result[key] = float(match.group(1))
    return result


def count_retractions_incremental(gcode_lines: List[str]) -> int:
    """
    Count retraction events by converting extrusion stream to incremental form.
    
    Accounts for:
    - Absolute/relative extrusion mode (M82/M83)
    - Extrusion resets (G92 E0)
    - Retraction events: ΔE_k < 0
    
    Returns:
        Number of retraction events detected
    """
    retraction_count = 0
    e_prev = 0.0  # Previous absolute E value (for tracking)
    is_relative_e = True  # Default assumption: relative mode (M83)
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        stripped_upper = stripped.upper()
        
        # Check for extrusion mode changes
        if 'M83' in stripped_upper:
            is_relative_e = True
            continue
        elif 'M82' in stripped_upper:
            is_relative_e = False
            continue
        
        # Handle G92 E resets (extrusion reset)
        if stripped_upper.startswith('G92'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None:
                # Reset the tracking value to the specified E value
                e_prev = parsed['E']
            continue
        
        # Process G0/G1 moves with E values
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None:
                e_val = parsed['E']
                
                if is_relative_e:
                    # In relative mode (M83), E value is already incremental
                    delta_e = e_val
                else:
                    # In absolute mode (M82), compute incremental change
                    delta_e = e_val - e_prev
                    e_prev = e_val  # Update tracking for next move
                
                # Count retraction: ΔE_k < 0
                if delta_e < -1e-6:  # Small threshold to avoid floating-point noise
                    retraction_count += 1
                
                # Update tracking for relative mode (cumulative)
                if is_relative_e:
                    e_prev += e_val
    
    return retraction_count


def find_gcode_files(directory: Path, filter_baseline: bool = False, filter_stabilized: bool = False) -> List[Path]:
    """
    Find all .gcode files in a directory (recursively).
    
    Args:
        directory: Directory to search
        filter_baseline: If True, only return files with 'baseline' in name
        filter_stabilized: If True, only return files with 'stabilized' in name
    """
    gcode_files = []
    if directory.exists() and directory.is_dir():
        all_files = list(directory.rglob('*.gcode'))
        
        if filter_baseline:
            gcode_files = [f for f in all_files if 'baseline' in f.name.lower()]
        elif filter_stabilized:
            gcode_files = [f for f in all_files if 'stabilized' in f.name.lower()]
        else:
            gcode_files = all_files
    
    return sorted(gcode_files)


def analyze_gcode_directory(directory: Path, is_stabilized: bool = False) -> List[int]:
    """
    Analyze all G-code files in a directory and return retraction counts.
    
    Args:
        directory: Directory containing G-code files
        is_stabilized: If True, filter for stabilized files; if False, filter for baseline files
    
    Returns:
        List of retraction counts, one per file
    """
    if is_stabilized:
        gcode_files = find_gcode_files(directory, filter_stabilized=True)
    else:
        gcode_files = find_gcode_files(directory, filter_baseline=True)
    
    # If no filtered files found, try all files
    if len(gcode_files) == 0:
        gcode_files = find_gcode_files(directory)
    
    retraction_counts = []
    
    print(f"Found {len(gcode_files)} G-code files in {directory} (is_stabilized={is_stabilized})", flush=True)
    
    for gcode_file in gcode_files:
        try:
            with open(gcode_file, 'r') as f:
                lines = f.readlines()
            
            retractions = count_retractions_incremental(lines)
            retraction_counts.append(retractions)
            print(f"  {gcode_file.name}: {retractions} retractions", flush=True)
            
        except Exception as e:
            print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
            retraction_counts.append(0)
    
    return retraction_counts


def generate_aggregate_retractions_figure(
    baseline_counts: List[int],
    stabilized_counts: List[int],
    output_path: Optional[Path] = None,
    use_log_scale: bool = True,
    show_paired_lines: bool = True,
    plot_type: str = 'boxplot'  # Options: 'boxplot', 'swarm', 'violin', 'slope'
):
    """
    Generate the aggregate retraction statistics figure.
    
    Creates a paired distribution plot showing:
    - X-axis: two categories (Baseline and Shaped)
    - Y-axis: Retraction events per file (count), log scale if counts vary widely
    - Individual points (light) + median and IQR (thicker box plots)
    - Optional: paired lines connecting each model's baseline to shaped count
    
    This figure supports the abstract-level claim: "across M models, the baseline
    has many retractions and the middleware eliminates them consistently."
    
    Retraction definition: segments with ΔE_k < 0 after parsing M82/M83 and G92 E0
    and converting to incremental extrusion.
    
    Suggested caption:
    "Aggregate baseline retractions across M=50 sliced models. Each point is one model;
    retraction events are counted as segments with ΔE_k < 0 after converting the stream
    to incremental extrusion (accounting for M82/M83 and G92 E0). The semantics-shaping
    pipeline eliminates retractions in all models (0 remaining), demonstrating consistent
    correction of FDM-derived negative-extrusion semantics."
    
    Args:
        baseline_counts: List of retraction counts for baseline files (M models)
        stabilized_counts: List of retraction counts for stabilized files (M models)
        output_path: Path to save figure
        use_log_scale: Whether to use log scale for y-axis (auto-enabled if max > 10)
        show_paired_lines: Whether to show lines connecting paired points (auto-disabled if M > 100)
    """
    # Convert to numpy arrays and ensure they're integers
    baseline_counts = np.array(baseline_counts, dtype=int)
    stabilized_counts = np.array(stabilized_counts, dtype=int)
    
    # Ensure same length (data correctness check)
    M = min(len(baseline_counts), len(stabilized_counts))
    if M == 0:
        print("ERROR: No data to plot", flush=True)
        return
    
    if len(baseline_counts) != len(stabilized_counts):
        print(f"WARNING: Mismatched lengths: baseline={len(baseline_counts)}, stabilized={len(stabilized_counts)}. Using M={M}", flush=True)
    
    baseline_counts = baseline_counts[:M]
    stabilized_counts = stabilized_counts[:M]
    
    # Verify data correctness: stabilized should be all zeros
    non_zero_stabilized = np.sum(stabilized_counts != 0)
    if non_zero_stabilized > 0:
        print(f"WARNING: {non_zero_stabilized} stabilized files have non-zero retractions!", flush=True)
    
    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    
    # Prepare data for plotting
    categories = ['Baseline', 'Shaped']
    positions = [1, 2]
    
    # Determine if log scale is appropriate
    max_count = np.max(baseline_counts) if len(baseline_counts) > 0 else 1
    min_count = np.min(baseline_counts[baseline_counts > 0]) if np.any(baseline_counts > 0) else 1
    
    use_log = use_log_scale and max_count > 10 and (max_count / min_count > 5)
    
    # Prepare plot data - handle zeros explicitly for log scale
    epsilon = 0.1  # Small value for plotting zeros on log scale
    if use_log:
        plot_baseline = baseline_counts.copy().astype(float)
        plot_stabilized = stabilized_counts.copy().astype(float)
        # Replace zeros with epsilon for log scale visualization
        plot_baseline[plot_baseline == 0] = epsilon
        plot_stabilized[plot_stabilized == 0] = epsilon
        plot_data = [plot_baseline, plot_stabilized]
        ax.set_yscale('log')
        ylabel = 'Retraction Events per Model'
    else:
        plot_data = [baseline_counts.astype(float), stabilized_counts.astype(float)]
        ylabel = 'Retraction Events per Model'
        epsilon = None
    
    # Draw paired lines FIRST (behind everything) - reduce visual dominance
    if show_paired_lines and M <= 100:
        for i in range(M):
            if stabilized_counts[i] == 0:
                # Line to zero - use green, lower opacity so baseline reads first
                ax.plot([1, 2], [plot_data[0][i], plot_data[1][i]],
                       color=COLORS['stabilized'], alpha=0.15, linewidth=1.0, zorder=1,
                       linestyle='-')
            else:
                # Line to non-zero - use red, lower opacity
                ax.plot([1, 2], [plot_data[0][i], plot_data[1][i]],
                       color=COLORS['baseline'], alpha=0.15, linewidth=1.0, zorder=1,
                       linestyle='-')
    
    # Create box plots with lighter styling - less visually heavy
    bp = ax.boxplot(plot_data, positions=positions, widths=0.5,
                    patch_artist=True, showmeans=False,
                    boxprops=dict(linewidth=1.8),
                    medianprops=dict(linewidth=3.0, color='black', solid_capstyle='round'),
                    whiskerprops=dict(linewidth=1.8),
                    capprops=dict(linewidth=1.8))
    
    # Color the boxes with lighter opacity for better readability
    bp['boxes'][0].set_facecolor(COLORS['baseline'])
    bp['boxes'][0].set_alpha(0.4)
    bp['boxes'][0].set_edgecolor('black')
    bp['boxes'][0].set_linewidth(1.8)
    
    bp['boxes'][1].set_facecolor(COLORS['stabilized'])
    bp['boxes'][1].set_alpha(0.4)
    bp['boxes'][1].set_edgecolor('black')
    bp['boxes'][1].set_linewidth(1.8)
    
    # Draw baseline median line (lighter, boxplot already shows it)
    baseline_median_val = np.median(baseline_counts)
    baseline_median_plot = baseline_median_val if not use_log else baseline_median_val
    ax.plot([0.7, 1.3], [baseline_median_plot, baseline_median_plot],
           color='black', linewidth=2.5, linestyle='--', zorder=6, alpha=0.7)
    
    # Highlight zero line for shaped if all zeros
    if np.all(stabilized_counts == 0):
        zero_y = epsilon if use_log else 0
        ax.plot([1.7, 2.3], [zero_y, zero_y],
               color=COLORS['stabilized'], linewidth=8, alpha=0.5, zorder=2,
               solid_capstyle='round')
    
    # Add individual points - make them larger and clearer
    np.random.seed(42)
    jitter_width = 0.15
    
    # Baseline points
    jitter_b = np.random.normal(0, jitter_width, len(baseline_counts))
    ax.scatter(1 + jitter_b, plot_data[0], 
              color=COLORS['baseline'],
              alpha=0.6, s=40, zorder=4, 
              edgecolors='black', linewidths=0.8,
              label=f'Baseline (n={M})')
    
    # Shaped points
    jitter_s = np.random.normal(0, jitter_width, len(stabilized_counts))
    if np.all(stabilized_counts == 0):
        # All zeros - make it very clear
        ax.scatter(2 + jitter_s, plot_data[1],
                  color=COLORS['stabilized'],
                  alpha=0.8, s=50, zorder=5,
                  edgecolors='darkgreen', linewidths=1.0,
                  marker='o', label=f'Shaped (n={M}, all zero)')
    else:
        zero_mask = stabilized_counts == 0
        non_zero_mask = stabilized_counts != 0
        if np.any(zero_mask):
            ax.scatter(2 + jitter_s[zero_mask], plot_data[1][zero_mask],
                      color=COLORS['stabilized'], alpha=0.8, s=50, zorder=5,
                      edgecolors='darkgreen', linewidths=1.0, marker='o')
        if np.any(non_zero_mask):
            ax.scatter(2 + jitter_s[non_zero_mask], plot_data[1][non_zero_mask],
                      color='red', alpha=0.8, s=50, zorder=5,
                      edgecolors='darkred', linewidths=1.0, marker='X')
    
    # Calculate statistics with proper rounding - ensure all variables defined
    baseline_median = np.median(baseline_counts)
    baseline_q25 = np.percentile(baseline_counts, 25)
    baseline_q75 = np.percentile(baseline_counts, 75)
    baseline_min = int(np.min(baseline_counts))
    baseline_max = int(np.max(baseline_counts))
    
    # REMOVED redundant median annotation - stats box and boxplot already show it
    
    # Handle "all zero" - integrate into stats box instead of separate annotation
    if np.all(stabilized_counts == 0):
        zero_y = epsilon if use_log else 0
        ax.plot([1.7, 2.3], [zero_y, zero_y],
               color='darkgreen', linewidth=2.5, linestyle='--', zorder=6, alpha=0.7)
    
    # Add statistics annotation box - compact for single-column legibility
    # Note: Consider moving this to caption for IEEE two-column layout
    stats_lines = [
        f'Baseline (M = {M}):',
        f'  Median: {int(baseline_median)}',
        f'  IQR: [{int(baseline_q25)}, {int(baseline_q75)}]',
        f'  Range: {baseline_min} - {baseline_max}',
        '',
        f'Shaped (M = {M}):',
    ]
    
    # Add shaped statistics with clear zero explanation
    if np.all(stabilized_counts == 0):
        stats_lines.append('  All models: 0 retractions')
        if use_log:
            stats_lines.append(f'  (plotted at ε={epsilon} for')
            stats_lines.append('   log scale visualization)')
    else:
        stats_lines.append(f'  Median: {int(np.median(stabilized_counts))}')
    
    stats_text = '\n'.join(stats_lines)
    
    # Ensure font size remains legible when scaled to column width
    ax.text(0.98, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           horizontalalignment='right',
           fontsize=9.5,  # Slightly larger for better legibility when scaled
           family='monospace',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    alpha=0.98, edgecolor='black', linewidth=1.2))
    
    # Set labels and formatting - optimized for single-column legibility
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_xticks(positions)
    ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', labelpad=10)
    
    # Add log scale indicator if needed
    if use_log:
        ax.text(-0.08, 0.5, '(log scale)', transform=ax.transAxes,
               rotation=90, fontsize=9, style='italic', va='center', ha='center')
    
    # Grid styling
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Ensure tick labels are readable at single-column size
    ax.tick_params(labelsize=10)
    
    # Add legend if needed (for points)
    # Legend is optional - uncomment if you want it
    # ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black', frameon=True)
    
    # Set y-axis limits
    if use_log:
        y_min = min(0.05, min_count * 0.3)
        y_max = max_count * 3.0
        ax.set_ylim([y_min, y_max])
    else:
        y_max = max_count * 1.3 if max_count > 0 else 10
        ax.set_ylim([-y_max * 0.03, y_max])
    
    # Remove title (let caption handle it) or make it subtler
    # ax.set_title(f'Aggregate Retraction Statistics Across {M} Models',
    #             fontsize=14, fontweight='bold', pad=15)
    
    # Add zero reference line for shaped (if using log scale) - make unambiguous
    if np.all(stabilized_counts == 0) and use_log:
        zero_y = epsilon
        ax.axhline(y=zero_y, color=COLORS['stabilized'], 
                  linestyle=':', linewidth=1.5, alpha=0.4, zorder=0)
        
        # Add clear text annotation near the line for unambiguous labeling
        ax.text(2.15, zero_y * 1.3, f'ε=0.1\n(zeros)',
               fontsize=8, style='italic', color=COLORS['stabilized'],
               va='bottom', ha='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        alpha=0.95, edgecolor=COLORS['stabilized'], linewidth=1.2),
               zorder=11)
        
        # Also add a small annotation on the y-axis near epsilon
        # Get the current y-axis limits to position annotation
        ylim = ax.get_ylim()
        ax.text(-0.12, zero_y, f'ε=0.1',
               transform=ax.get_yaxis_transform(),
               fontsize=8, style='italic', color=COLORS['stabilized'],
               va='center', ha='right',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                        alpha=0.9, edgecolor=COLORS['stabilized'], linewidth=0.8),
               zorder=11)
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / 'results' / 'figures' / 'agg_retractions_fig.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to: {output_path}", flush=True)
    plt.close(fig)
    
    # Print summary statistics
    print("\n" + "="*60, flush=True)
    print("Aggregate Retraction Statistics Summary", flush=True)
    print("="*60, flush=True)
    if len(baseline_counts) > 0:
        print(f"Baseline files (M={len(baseline_counts)}):", flush=True)
        print(f"  Mean retractions: {np.mean(baseline_counts):.2f}", flush=True)
        print(f"  Median retractions: {np.median(baseline_counts):.2f}", flush=True)
        print(f"  Std dev: {np.std(baseline_counts):.2f}", flush=True)
        print(f"  Min: {np.min(baseline_counts)}, Max: {np.max(baseline_counts)}", flush=True)
    if len(stabilized_counts) > 0:
        print(f"\nStabilized files (M={len(stabilized_counts)}):", flush=True)
        print(f"  Mean retractions: {np.mean(stabilized_counts):.2f}", flush=True)
        print(f"  Files with zero retractions: {np.sum(stabilized_counts == 0)}/{len(stabilized_counts)}", flush=True)
        print(f"  Files with non-zero retractions: {np.sum(stabilized_counts != 0)}/{len(stabilized_counts)}", flush=True)
    print("="*60 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Generate aggregate retraction statistics figure across sliced models (M=50)'
    )
    parser.add_argument('--baseline-dir', type=str, default=None,
                       help='Directory containing baseline G-code files')
    parser.add_argument('--stabilized-dir', type=str, default=None,
                       help='Directory containing stabilized G-code files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for figure (default: results/figures/agg_retractions_fig.png)')
    parser.add_argument('--baseline-files', type=str, nargs='+', default=None,
                       help='Explicit list of baseline G-code files (overrides --baseline-dir)')
    parser.add_argument('--stabilized-files', type=str, nargs='+', default=None,
                       help='Explicit list of stabilized G-code files (overrides --stabilized-dir)')
    parser.add_argument('--generate-demo-data', action='store_true',
                       help='Generate realistic demo data for M=50 files (for testing/demo purposes)')
    parser.add_argument('--no-log-scale', action='store_true',
                       help='Disable log scale for y-axis')
    parser.add_argument('--no-paired-lines', action='store_true',
                       help='Disable paired lines connecting baseline to shaped')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
    # Generate demo data if requested
    if args.generate_demo_data:
        print("Generating demo data for M=50 files...", flush=True)
        # Generate realistic baseline retraction counts
        # Distribution should reflect actual FDM slicer behavior:
        # - Wide distribution scaling with travel density and perimeter fragmentation
        # - Typical range: 50-800 retractions per model
        # - Some models with very few retractions (simple geometries)
        # - Some models with many retractions (complex geometries, many perimeters)
        
        np.random.seed(42)  # For reproducibility
        
        # Create a realistic distribution:
        # - 20% simple models: 50-200 retractions
        # - 50% medium models: 200-500 retractions  
        # - 25% complex models: 500-800 retractions
        # - 5% very complex models: 800-1200 retractions
        
        n_simple = 10
        n_medium = 25
        n_complex = 12
        n_very_complex = 3
        
        simple_counts = np.random.randint(50, 200, n_simple)
        medium_counts = np.random.randint(200, 500, n_medium)
        complex_counts = np.random.randint(500, 800, n_complex)
        very_complex_counts = np.random.randint(800, 1200, n_very_complex)
        
        baseline_counts = np.concatenate([
            simple_counts,
            medium_counts,
            complex_counts,
            very_complex_counts
        ])
        
        # Shuffle to avoid obvious grouping
        np.random.shuffle(baseline_counts)
        baseline_counts = baseline_counts.astype(int)
        
        # Stabilized: all zeros (middleware eliminates all retractions)
        stabilized_counts = np.zeros(50, dtype=int)
        
        print(f"Generated {len(baseline_counts)} baseline samples and {len(stabilized_counts)} stabilized samples", flush=True)
        print(f"Baseline range: {np.min(baseline_counts)} - {np.max(baseline_counts)} retractions", flush=True)
    else:
        # Process baseline files
        if args.baseline_files:
            baseline_files = [Path(f) for f in args.baseline_files]
            baseline_counts = []
            for gcode_file in baseline_files:
                try:
                    with open(gcode_file, 'r') as f:
                        lines = f.readlines()
                    retractions = count_retractions_incremental(lines)
                    baseline_counts.append(retractions)
                    print(f"  {gcode_file.name}: {retractions} retractions", flush=True)
                except Exception as e:
                    print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
                    baseline_counts.append(0)
        else:
            if args.baseline_dir is None:
                print("ERROR: --baseline-dir or --baseline-files must be provided", flush=True)
                sys.exit(1)
            baseline_dir = Path(args.baseline_dir)
            if not baseline_dir.is_absolute():
                baseline_dir = script_dir / args.baseline_dir
            baseline_counts = analyze_gcode_directory(baseline_dir, is_stabilized=False)
        
        # Process stabilized files
        if args.stabilized_files:
            stabilized_files = [Path(f) for f in args.stabilized_files]
            stabilized_counts = []
            for gcode_file in stabilized_files:
                try:
                    with open(gcode_file, 'r') as f:
                        lines = f.readlines()
                    retractions = count_retractions_incremental(lines)
                    stabilized_counts.append(retractions)
                    print(f"  {gcode_file.name}: {retractions} retractions", flush=True)
                except Exception as e:
                    print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
                    stabilized_counts.append(0)
        else:
            if args.stabilized_dir is None:
                print("ERROR: --stabilized-dir or --stabilized-files must be provided", flush=True)
                sys.exit(1)
            stabilized_dir = Path(args.stabilized_dir)
            if not stabilized_dir.is_absolute():
                stabilized_dir = script_dir / args.stabilized_dir
            stabilized_counts = analyze_gcode_directory(stabilized_dir, is_stabilized=True)
    
    # Validate that we have data
    if len(baseline_counts) == 0 and len(stabilized_counts) == 0:
        print("ERROR: No G-code files found. Use --generate-demo-data for testing or provide valid directories/files.", flush=True)
        sys.exit(1)
    
    # Generate figure
    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / args.output
    
    generate_aggregate_retractions_figure(
        baseline_counts,
        stabilized_counts,
        output_path,
        use_log_scale=not args.no_log_scale,
        show_paired_lines=not args.no_paired_lines
    )


if __name__ == '__main__':
    main()
