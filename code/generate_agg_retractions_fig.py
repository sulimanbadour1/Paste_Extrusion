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


def find_gcode_files(directory: Path) -> List[Path]:
    """Find all .gcode files in a directory (recursively)."""
    gcode_files = []
    if directory.exists() and directory.is_dir():
        gcode_files = list(directory.rglob('*.gcode'))
    return sorted(gcode_files)


def analyze_gcode_directory(directory: Path) -> List[int]:
    """
    Analyze all G-code files in a directory and return retraction counts.
    
    Returns:
        List of retraction counts, one per file
    """
    gcode_files = find_gcode_files(directory)
    retraction_counts = []
    
    print(f"Found {len(gcode_files)} G-code files in {directory}", flush=True)
    
    for gcode_file in gcode_files:
        try:
            with open(gcode_file, 'r') as f:
                lines = f.readlines()
            
            retractions = count_retractions_incremental(lines)
            retraction_counts.append(retractions)
            
        except Exception as e:
            print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
            retraction_counts.append(0)
    
    return retraction_counts


def generate_aggregate_retractions_figure(
    baseline_counts: List[int],
    stabilized_counts: List[int],
    output_path: Optional[Path] = None
):
    """
    Generate the aggregate retraction statistics figure.
    
    Shows:
    - Distribution of retraction counts for baseline files
    - Zero retractions for all stabilized files
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.0))
    
    # Convert to numpy arrays
    baseline_counts = np.array(baseline_counts)
    stabilized_counts = np.array(stabilized_counts)
    
    # Left panel: Baseline retraction distribution
    if len(baseline_counts) > 0:
        # Create histogram
        max_retractions = int(np.max(baseline_counts)) if len(baseline_counts) > 0 else 100
        bins = np.arange(0, max_retractions + 20, 20)  # Bin width of 20
        
        counts, bin_edges, patches = ax1.hist(
            baseline_counts,
            bins=bins,
            color=COLORS['baseline'],
            alpha=0.7,
            edgecolor='black',
            linewidth=1.2,
            label=f'Baseline (M={len(baseline_counts)})'
        )
        
        # Add statistics text
        mean_retractions = np.mean(baseline_counts)
        median_retractions = np.median(baseline_counts)
        std_retractions = np.std(baseline_counts)
        
        stats_text = (
            f'Mean: {mean_retractions:.1f}\n'
            f'Median: {median_retractions:.1f}\n'
            f'Std: {std_retractions:.1f}\n'
            f'Range: [{np.min(baseline_counts)}, {np.max(baseline_counts)}]'
        )
        
        ax1.text(0.98, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                fontsize=10,
                family='monospace')
        
        ax1.set_xlabel('Retraction Count per File', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Baseline Slicer Output', fontsize=13, fontweight='bold', pad=10)
        ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
        ax1.set_axisbelow(True)
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
    else:
        ax1.text(0.5, 0.5, 'No baseline data available',
                transform=ax1.transAxes,
                ha='center', va='center',
                fontsize=14)
        ax1.set_xlabel('Retraction Count per File', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Baseline Slicer Output', fontsize=13, fontweight='bold', pad=10)
    
    # Right panel: Stabilized retraction distribution (should be all zeros)
    if len(stabilized_counts) > 0:
        # All should be zero, but show histogram anyway
        unique_counts, unique_counts_freq = np.unique(stabilized_counts, return_counts=True)
        
        # Create bar plot for unique values (should be just [0])
        bars = ax2.bar(unique_counts, unique_counts_freq,
                      color=COLORS['stabilized'],
                      alpha=0.7,
                      edgecolor='black',
                      linewidth=1.2,
                      width=0.8,
                      label=f'Stabilized (M={len(stabilized_counts)})')
        
        # Add value labels on bars
        for bar, count in zip(bars, unique_counts_freq):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom',
                    fontweight='bold', fontsize=11)
        
        # Verify all zeros
        non_zero_count = np.sum(stabilized_counts != 0)
        if non_zero_count > 0:
            warning_text = f'Warning: {non_zero_count} files have non-zero retractions'
            ax2.text(0.5, 0.95, warning_text,
                    transform=ax2.transAxes,
                    ha='center', va='top',
                    fontsize=10,
                    color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        else:
            success_text = f'✓ All {len(stabilized_counts)} files have zero retractions'
            ax2.text(0.5, 0.95, success_text,
                    transform=ax2.transAxes,
                    ha='center', va='top',
                    fontsize=11,
                    fontweight='bold',
                    color='green',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax2.set_xlabel('Retraction Count per File', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
        ax2.set_title('(b) After Middleware Processing', fontsize=13, fontweight='bold', pad=10)
        ax2.set_xticks(unique_counts)
        ax2.set_xticklabels([str(int(x)) for x in unique_counts])
        ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)
        ax2.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
    else:
        ax2.text(0.5, 0.5, 'No stabilized data available',
                transform=ax2.transAxes,
                ha='center', va='center',
                fontsize=14)
        ax2.set_xlabel('Retraction Count per File', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Files', fontsize=12, fontweight='bold')
        ax2.set_title('(b) After Middleware Processing', fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        script_dir = Path(__file__).parent
        output_path = script_dir / 'results' / 'figures' / 'agg_retractions_fig.png'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved figure to: {output_path}", flush=True)
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
    parser.add_argument('--baseline-dir', type=str, required=True,
                       help='Directory containing baseline G-code files')
    parser.add_argument('--stabilized-dir', type=str, required=True,
                       help='Directory containing stabilized G-code files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for figure (default: results/figures/agg_retractions_fig.png)')
    parser.add_argument('--baseline-files', type=str, nargs='+', default=None,
                       help='Explicit list of baseline G-code files (overrides --baseline-dir)')
    parser.add_argument('--stabilized-files', type=str, nargs='+', default=None,
                       help='Explicit list of stabilized G-code files (overrides --stabilized-dir)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    
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
            except Exception as e:
                print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
                baseline_counts.append(0)
    else:
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.is_absolute():
            baseline_dir = script_dir / args.baseline_dir
        baseline_counts = analyze_gcode_directory(baseline_dir)
    
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
            except Exception as e:
                print(f"Warning: Error processing {gcode_file}: {e}", flush=True)
                stabilized_counts.append(0)
    else:
        stabilized_dir = Path(args.stabilized_dir)
        if not stabilized_dir.is_absolute():
            stabilized_dir = script_dir / args.stabilized_dir
        stabilized_counts = analyze_gcode_directory(stabilized_dir)
    
    # Generate figure
    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = script_dir / args.output
    
    generate_aggregate_retractions_figure(
        baseline_counts,
        stabilized_counts,
        output_path
    )


if __name__ == '__main__':
    main()
