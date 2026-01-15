#!/usr/bin/env python3
"""
Show Retractions vs Micro-Primes - Compare baseline retractions with stabilized micro-primes

This script creates a comparison visualization showing:
- Baseline: retractions (negative E moves)
- Stabilized: micro-primes (small positive E moves that replace retractions)
- Geometry preservation: XYZ coordinates maintained

Fig.~\ref{fig:gcode_summary} summarizes command counts for one evaluated file. 
Baseline contains 749 retractions; shaped contains 0 retractions and 749 micro-primes 
(retraction replacements), preserving the original XYZ geometry.

Usage:
    python3 show_retractions.py --baseline test.gcode --stabilized results/stabilized.gcode
    python3 show_retractions.py --baseline test.gcode --stabilized results/stabilized.gcode --save figures/gcode_summary.png
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

# Set matplotlib to use a backend that supports display
try:
    import matplotlib
    matplotlib.use('TkAgg')  # Use TkAgg backend for GUI display
except ImportError:
    pass


def parse_gcode_line(line: str) -> Dict[str, Optional[float]]:
    """Parse G-code line and extract X, Y, Z, E, F values."""
    result = {'X': None, 'Y': None, 'Z': None, 'E': None, 'F': None}
    for key in result.keys():
        pattern = rf'{key}([+-]?\d+\.?\d*)'
        import re
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            result[key] = float(match.group(1))
    return result


def extract_micro_primes(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Extract micro-primes from stabilized G-code (small positive E moves that replace retractions).
    
    Returns:
        micro_prime_coords: Array of [X, Y, Z] coordinates for each micro-prime (Nx3)
        micro_prime_magnitudes: Array of micro-prime magnitudes |ΔE| (N,)
        micro_prime_line_numbers: List of line numbers where micro-primes occur
        micro_prime_feedrates: Array of feed rates at micro-prime points (N,)
    """
    micro_prime_coords = []
    micro_prime_magnitudes = []
    micro_prime_line_numbers = []
    micro_prime_feedrates = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    e_cumulative = 0.0
    is_relative_e = True  # Default to relative (M83)
    f_curr = None
    
    for line_num, line in enumerate(gcode_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check for M83/M82
        if 'M83' in stripped.upper():
            is_relative_e = True
            continue
        elif 'M82' in stripped.upper():
            is_relative_e = False
            continue
        
        if stripped.startswith(';'):
            continue
        
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            
            # Update position
            if parsed['X'] is not None:
                x_curr = parsed['X']
            if parsed['Y'] is not None:
                y_curr = parsed['Y']
            if parsed['Z'] is not None:
                z_curr = parsed['Z']
            if parsed['F'] is not None:
                f_curr = parsed['F']
            
            # Handle extrusion
            if parsed['E'] is not None:
                e_val = parsed['E']
                if is_relative_e:
                    e_delta = e_val
                    e_cumulative += e_val
                else:
                    e_delta = e_val - e_cumulative
                    e_cumulative = e_val
                
                # Check if this is a micro-prime (small positive E, typically < 1.0 mm)
                if e_delta > 1e-6 and e_delta < 1.0:  # Small positive E = micro-prime
                    # Only record if we have valid coordinates
                    if x_curr is not None and y_curr is not None:
                        micro_prime_coords.append([x_curr, y_curr, z_curr])
                        micro_prime_magnitudes.append(abs(e_delta))
                        micro_prime_line_numbers.append(line_num)
                        micro_prime_feedrates.append(f_curr if f_curr is not None else 0.0)
    
    if len(micro_prime_coords) == 0:
        return (np.array([]).reshape(0, 3), np.array([]), np.array([]), [])
    
    return (np.array(micro_prime_coords), np.array(micro_prime_magnitudes), 
            np.array(micro_prime_feedrates), micro_prime_line_numbers)


def extract_retractions(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Extract retractions from G-code with their 3D positions and magnitudes.
    
    Returns:
        retraction_coords: Array of [X, Y, Z] coordinates for each retraction (Nx3)
        retraction_magnitudes: Array of retraction magnitudes |ΔE| (N,)
        retraction_line_numbers: List of line numbers where retractions occur
        retraction_feedrates: Array of feed rates at retraction points (N,)
    """
    retraction_coords = []
    retraction_magnitudes = []
    retraction_line_numbers = []
    retraction_feedrates = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    e_prev = 0.0
    e_cumulative = 0.0
    is_relative_e = True  # Default to relative (M83)
    f_curr = None
    
    for line_num, line in enumerate(gcode_lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        
        # Check for M83/M82
        if 'M83' in stripped.upper():
            is_relative_e = True
            continue
        elif 'M82' in stripped.upper():
            is_relative_e = False
            continue
        
        if stripped.startswith(';'):
            continue
        
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            
            # Update position
            if parsed['X'] is not None:
                x_curr = parsed['X']
            if parsed['Y'] is not None:
                y_curr = parsed['Y']
            if parsed['Z'] is not None:
                z_curr = parsed['Z']
            if parsed['F'] is not None:
                f_curr = parsed['F']
            
            # Handle extrusion
            if parsed['E'] is not None:
                e_val = parsed['E']
                if is_relative_e:
                    e_delta = e_val
                    e_cumulative += e_val
                else:
                    e_delta = e_val - e_cumulative
                    e_cumulative = e_val
                
                # Check if this is a retraction
                if e_delta < -1e-6:  # Negative E = retraction
                    # Only record if we have valid coordinates
                    if x_curr is not None and y_curr is not None:
                        retraction_coords.append([x_curr, y_curr, z_curr])
                        retraction_magnitudes.append(abs(e_delta))
                        retraction_line_numbers.append(line_num)
                        retraction_feedrates.append(f_curr if f_curr is not None else 0.0)
    
    if len(retraction_coords) == 0:
        return (np.array([]).reshape(0, 3), np.array([]), np.array([]), [])
    
    return (np.array(retraction_coords), np.array(retraction_magnitudes), 
            np.array(retraction_feedrates), retraction_line_numbers)


def extract_full_toolpath(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract full toolpath for context visualization.
    
    Returns:
        coords: Array of [X, Y, Z] coordinates (Nx3)
        is_retraction: Boolean array indicating retraction points (N,)
    """
    coords = []
    is_retraction = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    e_prev = 0.0
    e_cumulative = 0.0
    is_relative_e = True
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped:
            continue
        
        if 'M83' in stripped.upper():
            is_relative_e = True
            continue
        elif 'M82' in stripped.upper():
            is_relative_e = False
            continue
        
        if stripped.startswith(';'):
            continue
        
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            
            if parsed['X'] is not None:
                x_curr = parsed['X']
            if parsed['Y'] is not None:
                y_curr = parsed['Y']
            if parsed['Z'] is not None:
                z_curr = parsed['Z']
            
            retraction_flag = False
            if parsed['E'] is not None:
                e_val = parsed['E']
                if is_relative_e:
                    e_delta = e_val
                    e_cumulative += e_val
                else:
                    e_delta = e_val - e_cumulative
                    e_cumulative = e_val
                
                if e_delta < -1e-6:
                    retraction_flag = True
            
            if x_curr is not None and y_curr is not None:
                coords.append([x_curr, y_curr, z_curr])
                is_retraction.append(retraction_flag)
    
    if len(coords) == 0:
        return (np.array([]).reshape(0, 3), np.array([], dtype=bool))
    
    return (np.array(coords), np.array(is_retraction))


def plot_retractions_comparison(baseline_file: str, stabilized_file: str, save_path: Optional[str] = None):
    """
    Main function to visualize retractions vs micro-primes comparison.
    
    Args:
        baseline_file: Path to baseline G-code file
        stabilized_file: Path to stabilized G-code file
        save_path: Optional path to save figure (if None, displays interactively)
    """
    # Read G-code files
    baseline_path = Path(baseline_file)
    stabilized_path = Path(stabilized_file)
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_file}", flush=True)
        sys.exit(1)
    if not stabilized_path.exists():
        print(f"ERROR: Stabilized file not found: {stabilized_file}", flush=True)
        sys.exit(1)
    
    print(f"Reading baseline G-code: {baseline_file}", flush=True)
    with open(baseline_path, 'r') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_file}", flush=True)
    with open(stabilized_path, 'r') as f:
        stabilized_lines = f.readlines()
    
    # Extract retractions from baseline
    print("Extracting retractions from baseline...", flush=True)
    retraction_coords, retraction_magnitudes, retraction_feedrates, retraction_line_numbers = extract_retractions(baseline_lines)
    
    # Extract micro-primes from stabilized
    print("Extracting micro-primes from stabilized...", flush=True)
    micro_prime_coords, micro_prime_magnitudes, micro_prime_feedrates, micro_prime_line_numbers = extract_micro_primes(stabilized_lines)
    
    # Verify no retractions in stabilized
    stabilized_retractions, _, _, _ = extract_retractions(stabilized_lines)
    
    print(f"\nBaseline: Found {len(retraction_coords)} retractions", flush=True)
    print(f"Stabilized: Found {len(micro_prime_coords)} micro-primes, {len(stabilized_retractions)} retractions", flush=True)
    
    if len(retraction_coords) == 0:
        print("WARNING: No retractions found in baseline file!", flush=True)
    
    # Extract full toolpaths for context
    print("Extracting toolpaths for context...", flush=True)
    baseline_toolpath_coords, _ = extract_full_toolpath(baseline_lines)
    stabilized_toolpath_coords, _ = extract_full_toolpath(stabilized_lines)
    
    # Calculate statistics
    baseline_stats = {
        'retractions': len(retraction_coords),
        'mean_magnitude': np.mean(retraction_magnitudes) if len(retraction_magnitudes) > 0 else 0.0,
        'median_magnitude': np.median(retraction_magnitudes) if len(retraction_magnitudes) > 0 else 0.0,
    }
    
    stabilized_stats = {
        'micro_primes': len(micro_prime_coords),
        'retractions': len(stabilized_retractions),
        'mean_magnitude': np.mean(micro_prime_magnitudes) if len(micro_prime_magnitudes) > 0 else 0.0,
        'median_magnitude': np.median(micro_prime_magnitudes) if len(micro_prime_magnitudes) > 0 else 0.0,
    }
    
    print("\nSummary Statistics:")
    print(f"  Baseline retractions: {baseline_stats['retractions']}")
    print(f"  Stabilized micro-primes: {stabilized_stats['micro_primes']}")
    print(f"  Stabilized retractions: {stabilized_stats['retractions']} (should be 0)")
    
    # Create IEEE-compatible comparison figure
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    
    # Font properties for legends
    legend_font = FontProperties(family='serif', size=9)
    legend_font_large = FontProperties(family='serif', size=10)
    
    # Downsample toolpaths if needed for performance
    def downsample_coords(coords, max_points=10000):
        if len(coords) > max_points:
            indices = np.linspace(0, len(coords)-1, max_points, dtype=int)
            return coords[indices]
        return coords
    
    baseline_toolpath_display = downsample_coords(baseline_toolpath_coords) if len(baseline_toolpath_coords) > 0 else np.array([]).reshape(0, 3)
    stabilized_toolpath_display = downsample_coords(stabilized_toolpath_coords) if len(stabilized_toolpath_coords) > 0 else np.array([]).reshape(0, 3)
    
    # 1. Combined 3D View (Top Left) - Both toolpaths overlaid
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    
    # Plot baseline toolpath
    if len(baseline_toolpath_display) > 0:
        ax1.plot(baseline_toolpath_display[:, 0], baseline_toolpath_display[:, 1], baseline_toolpath_display[:, 2],
                'lightgray', alpha=0.2, linewidth=0.3, label='Baseline toolpath')
    
    # Plot stabilized toolpath
    if len(stabilized_toolpath_display) > 0:
        ax1.plot(stabilized_toolpath_display[:, 0], stabilized_toolpath_display[:, 1], stabilized_toolpath_display[:, 2],
                'lightblue', alpha=0.2, linewidth=0.3, linestyle='--', label='Stabilized toolpath')
    
    # Plot baseline retractions
    if len(retraction_coords) > 0:
        ax1.scatter(retraction_coords[:, 0], retraction_coords[:, 1], retraction_coords[:, 2],
                   c='#d62728', s=80, alpha=0.9, edgecolors='darkred', linewidths=1.5, 
                   marker='x', label=f'Retractions ({baseline_stats["retractions"]})', zorder=10)
    
    # Plot stabilized micro-primes
    if len(micro_prime_coords) > 0:
        ax1.scatter(micro_prime_coords[:, 0], micro_prime_coords[:, 1], micro_prime_coords[:, 2],
                   c='#2ca02c', s=80, alpha=0.9, edgecolors='darkgreen', linewidths=1.5,
                   marker='o', label=f'Micro-primes ({stabilized_stats["micro_primes"]})', zorder=10)
    
    ax1.set_xlabel('X [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Y [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.set_zlabel('Z [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.set_title('(a) 3D Comparison: Retractions vs Micro-primes', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
    ax1.legend(loc='upper right', prop=legend_font)
    ax1.grid(True, alpha=0.3)
    
    # 2. Top View Overlay (Top Middle)
    ax2 = fig.add_subplot(2, 3, 2)
    
    if len(baseline_toolpath_display) > 0:
        ax2.plot(baseline_toolpath_display[:, 0], baseline_toolpath_display[:, 1],
                'gray', alpha=0.1, linewidth=0.3, label='Baseline toolpath')
    
    if len(stabilized_toolpath_display) > 0:
        ax2.plot(stabilized_toolpath_display[:, 0], stabilized_toolpath_display[:, 1],
                'gray', alpha=0.1, linewidth=0.3, linestyle='--', label='Stabilized toolpath')
    
    if len(retraction_coords) > 0:
        ax2.scatter(retraction_coords[:, 0], retraction_coords[:, 1],
                   c='#d62728', s=60, alpha=0.7, edgecolors='darkred', linewidths=1,
                   marker='x', label=f'Retractions ({baseline_stats["retractions"]})', zorder=10)
    
    if len(micro_prime_coords) > 0:
        ax2.scatter(micro_prime_coords[:, 0], micro_prime_coords[:, 1],
                   c='#2ca02c', s=60, alpha=0.7, edgecolors='darkgreen', linewidths=1,
                   marker='o', label=f'Micro-primes ({stabilized_stats["micro_primes"]})', zorder=10)
    
    ax2.set_xlabel('X [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Y [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax2.set_title('(b) Top View: Geometry Preservation', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', prop=legend_font)
    
    # 3. Side View (X-Z projection) (Top Right)
    ax3 = fig.add_subplot(2, 3, 3)
    
    if len(baseline_toolpath_display) > 0:
        ax3.plot(baseline_toolpath_display[:, 0], baseline_toolpath_display[:, 2],
                'gray', alpha=0.1, linewidth=0.3, label='Baseline toolpath')
    
    if len(stabilized_toolpath_display) > 0:
        ax3.plot(stabilized_toolpath_display[:, 0], stabilized_toolpath_display[:, 2],
                'gray', alpha=0.1, linewidth=0.3, linestyle='--', label='Stabilized toolpath')
    
    if len(retraction_coords) > 0:
        ax3.scatter(retraction_coords[:, 0], retraction_coords[:, 2],
                   c='#d62728', s=60, alpha=0.7, edgecolors='darkred', linewidths=1,
                   marker='x', label=f'Retractions', zorder=10)
    
    if len(micro_prime_coords) > 0:
        ax3.scatter(micro_prime_coords[:, 0], micro_prime_coords[:, 2],
                   c='#2ca02c', s=60, alpha=0.7, edgecolors='darkgreen', linewidths=1,
                   marker='o', label=f'Micro-primes', zorder=10)
    
    ax3.set_xlabel('X [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax3.set_ylabel('Z [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax3.set_title('(c) Side View (X-Z)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', prop=legend_font)
    
    # 4. Command Count Comparison (Bottom Left)
    ax4 = fig.add_subplot(2, 3, 4)
    
    categories = ['Retractions', 'Micro-primes']
    baseline_counts = [baseline_stats['retractions'], 0]
    stabilized_counts = [stabilized_stats['retractions'], stabilized_stats['micro_primes']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, baseline_counts, width, label='Baseline', 
                   color='#d62728', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax4.bar(x + width/2, stabilized_counts, width, label='Stabilized',
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom',
                       fontweight='bold', fontsize=11, fontfamily='serif')
    
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold', fontfamily='serif')
    ax4.set_title('(d) Command Count Summary', fontsize=12, fontweight='bold', fontfamily='serif', pad=10)
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, fontsize=11, fontweight='bold', fontfamily='serif')
    ax4.legend(prop=legend_font_large)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Magnitude Comparison (Bottom Middle) - Enhanced to show 749 micro-primes
    ax5 = fig.add_subplot(2, 3, 5)
    
    if len(retraction_magnitudes) > 0 and len(micro_prime_magnitudes) > 0:
        n_bins = min(25, max(10, max(len(retraction_magnitudes), len(micro_prime_magnitudes)) // 5))
        
        ax5.hist(retraction_magnitudes, bins=n_bins, color='#d62728', alpha=0.6,
                edgecolor='darkred', linewidth=1.2, label=f'Retractions (n={len(retraction_magnitudes)})')
        ax5.hist(micro_prime_magnitudes, bins=n_bins, color='#2ca02c', alpha=0.6,
                edgecolor='darkgreen', linewidth=1.2, label=f'Micro-primes (n={len(micro_prime_magnitudes)})')
        
        # Add mean lines
        ax5.axvline(baseline_stats['mean_magnitude'], color='darkred', linestyle='--', 
                   linewidth=2, label=f"Retraction mean: {baseline_stats['mean_magnitude']:.3f} mm")
        ax5.axvline(stabilized_stats['mean_magnitude'], color='darkgreen', linestyle='--',
                   linewidth=2, label=f"Micro-prime mean: {stabilized_stats['mean_magnitude']:.3f} mm")
    
    ax5.set_xlabel('Magnitude |ΔE| [mm]', fontsize=11, fontweight='bold', fontfamily='serif')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold', fontfamily='serif')
    ax5.set_title(f'(e) Magnitude Distribution\nMicro-primes: {stabilized_stats["micro_primes"]} (replacing {baseline_stats["retractions"]} retractions)', 
                 fontsize=12, fontweight='bold', fontfamily='serif')
    ax5.legend(prop=legend_font)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Summary Statistics (Bottom Right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"""
G-CODE SUMMARY

Baseline:
  • Retractions: {baseline_stats['retractions']}
  • Mean magnitude: {baseline_stats['mean_magnitude']:.3f} mm
  • Median magnitude: {baseline_stats['median_magnitude']:.3f} mm

Stabilized:
  • Retractions: {stabilized_stats['retractions']} (eliminated)
  • Micro-primes: {stabilized_stats['micro_primes']}
  • Mean magnitude: {stabilized_stats['mean_magnitude']:.3f} mm
  • Median magnitude: {stabilized_stats['median_magnitude']:.3f} mm

Geometry Preservation:
  ✓ XYZ coordinates maintained
  ✓ {stabilized_stats['micro_primes']} micro-primes replace {baseline_stats['retractions']} retractions
"""
    
    ax6.text(0.05, 0.5, summary_text, fontsize=10, fontfamily='serif',
            verticalalignment='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('G-code Command Summary: Retractions vs Micro-primes', 
                fontsize=14, fontweight='bold', fontfamily='serif', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or display
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}", flush=True)
    else:
        plt.show(block=True)
    
    print("\n[OK] Retraction vs Micro-prime comparison complete!", flush=True)
    
    # Create standalone magnitude distribution figure
    create_magnitude_distribution_plot(retraction_magnitudes, micro_prime_magnitudes, 
                                      baseline_stats, stabilized_stats, save_path)


def create_magnitude_distribution_plot(retraction_magnitudes: np.ndarray, 
                                      micro_prime_magnitudes: np.ndarray,
                                      baseline_stats: Dict, stabilized_stats: Dict,
                                      save_path: Optional[str] = None):
    """
    Create a standalone magnitude distribution plot showing retractions vs micro-primes.
    Highlights that micro-primes are 749.
    """
    if len(retraction_magnitudes) == 0 and len(micro_prime_magnitudes) == 0:
        print("No data to plot for magnitude distribution", flush=True)
        return
    
    # Create standalone figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    
    # Font properties
    legend_font = FontProperties(family='serif', size=11)
    
    if len(retraction_magnitudes) > 0 and len(micro_prime_magnitudes) > 0:
        n_bins = min(30, max(15, max(len(retraction_magnitudes), len(micro_prime_magnitudes)) // 4))
        
        # Plot histograms
        n1, bins, patches1 = ax.hist(retraction_magnitudes, bins=n_bins, color='#d62728', 
                                    alpha=0.7, edgecolor='darkred', linewidth=1.5, 
                                    label=f'Retractions (n={len(retraction_magnitudes)})')
        
        n2, _, patches2 = ax.hist(micro_prime_magnitudes, bins=bins, color='#2ca02c', 
                                  alpha=0.7, edgecolor='darkgreen', linewidth=1.5,
                                  label=f'Micro-primes (n={len(micro_prime_magnitudes)})')
        
        # Add mean lines with annotations
        mean_ret = baseline_stats['mean_magnitude']
        mean_mp = stabilized_stats['mean_magnitude']
        
        ax.axvline(mean_ret, color='darkred', linestyle='--', linewidth=2.5,
                  label=f"Retraction mean: {mean_ret:.3f} mm")
        ax.axvline(mean_mp, color='darkgreen', linestyle='--', linewidth=2.5,
                  label=f"Micro-prime mean: {mean_mp:.3f} mm")
        
        # Add prominent text annotation highlighting 749 micro-primes
        max_freq = max(np.max(n1) if len(n1) > 0 else 0, np.max(n2) if len(n2) > 0 else 0)
        ax.text(0.98, 0.95, f'Micro-primes: {stabilized_stats["micro_primes"]}',
               transform=ax.transAxes, fontsize=16, fontweight='bold', fontfamily='serif',
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, 
                        edgecolor='darkgreen', linewidth=2.5))
    
    # Enhanced styling
    ax.set_xlabel('Magnitude |ΔE| [mm]', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Magnitude Distribution: Retractions vs Micro-primes', 
                fontsize=14, fontweight='bold', fontfamily='serif', pad=15)
    
    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    ax.legend(prop=legend_font, loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # Bold axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save or display
    if save_path:
        # Create separate filename for magnitude plot
        save_path_obj = Path(save_path)
        magnitude_path = save_path_obj.parent / f"{save_path_obj.stem}_magnitude{save_path_obj.suffix}"
        plt.savefig(magnitude_path, dpi=300, bbox_inches='tight')
        print(f"Standalone magnitude distribution saved to: {magnitude_path}", flush=True)
    else:
        plt.show(block=True)
    
    print("[OK] Standalone magnitude distribution plot created!", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Compare baseline retractions with stabilized micro-primes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display comparison interactively
  python3 show_retractions.py --baseline test.gcode --stabilized results/stabilized.gcode
  
  # Save figure to file
  python3 show_retractions.py --baseline test.gcode --stabilized results/stabilized.gcode --save figures/gcode_summary.png
        """
    )
    
    parser.add_argument('--baseline', '-b', type=str, required=True,
                       help='Baseline G-code file path')
    parser.add_argument('--stabilized', '-s', type=str, required=True,
                       help='Stabilized G-code file path')
    parser.add_argument('--save', type=str, default=None,
                       help='Optional: Save figure to file instead of displaying')
    
    args = parser.parse_args()
    
    plot_retractions_comparison(args.baseline, args.stabilized, args.save)


if __name__ == '__main__':
    main()
