#!/usr/bin/env python3
"""
3D Toolpath Comparison Visualization
Creates a high-quality side-by-side 3D visualization showing the difference
between baseline and stabilized G-code toolpaths.

# Usage:
# Display interactively
python3 3d_map.py --baseline test.gcode --stabilized results/stabilized.gcode

# Save to file (high resolution)
python3 3d_map.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/3d_comparison.png

# Custom DPI
python3 3d_map.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/3d_comparison.png --dpi 600
Key Features:
- Retractions highlighted in red (baseline)
- Micro-primes highlighted in green (stabilized)
- Extrusion moves color-coded by rate
- Travel moves in light colors
- Synchronized axis limits for easy comparison
- Publication-quality styling
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# Determine if we're saving or displaying
# Check args early to set backend correctly
_save_mode = False
if len(sys.argv) > 1:
    _save_mode = '--output' in sys.argv or '-o' in sys.argv

import matplotlib
if _save_mode:
    matplotlib.use('Agg')  # Non-interactive for saving
else:
    # Try interactive backends in order
    try:
        matplotlib.use('TkAgg')  # Best for macOS/Linux
    except:
        try:
            matplotlib.use('Qt5Agg')  # Fallback
        except:
            matplotlib.use('Agg')  # Last resort

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors

# ============================================================================
# G-code Parsing
# ============================================================================

def parse_gcode_line(line: str) -> Dict[str, Optional[float]]:
    """Parse a G-code line and extract X, Y, Z, E, F values."""
    result = {'X': None, 'Y': None, 'Z': None, 'E': None, 'F': None}
    
    for key in result.keys():
        pattern = rf'{key}([+-]?\d+\.?\d*)'
        match = re.search(pattern, line)
        if match:
            result[key] = float(match.group(1))
    
    return result


def extract_3d_toolpath(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D toolpath coordinates and extrusion information from G-code.
    Returns: (coords [Nx3], e_deltas [N], extrusion_flags [N], retraction_flags [N], feed_rates [N])
    """
    coords = []
    e_deltas = []
    extrusion_flags = []
    retraction_flags = []
    feed_rates = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    x_prev, y_prev, z_prev = None, None, 0.0
    e_cumulative = 0.0
    is_relative_e = True  # Default to relative (M83)
    f_curr = None
    
    for line in gcode_lines:
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
            
            # Store previous position
            x_prev = x_curr
            y_prev = y_curr
            z_prev = z_curr
            
            # Update position
            if parsed['X'] is not None:
                x_curr = parsed['X']
            if parsed['Y'] is not None:
                y_curr = parsed['Y']
            if parsed['Z'] is not None:
                z_curr = parsed['Z']
            if parsed['F'] is not None:
                f_curr = parsed['F']
            
            # Use previous position if current move doesn't specify coordinates
            x_to_use = x_curr if x_curr is not None else x_prev
            y_to_use = y_curr if y_curr is not None else y_prev
            z_to_use = z_curr if z_curr is not None else (z_prev if z_prev is not None else 0.0)
            
            # Handle extrusion
            e_delta = 0.0
            if parsed['E'] is not None:
                e_val = parsed['E']
                if is_relative_e:
                    e_delta = e_val
                    e_cumulative += e_val
                else:
                    e_delta = e_val - e_cumulative
                    e_cumulative = e_val
            
            # Record point if we have valid coordinates
            if x_to_use is not None and y_to_use is not None:
                coords.append([x_to_use, y_to_use, z_to_use])
                e_deltas.append(e_delta)
                extrusion_flags.append(e_delta > 1e-6)
                retraction_flags.append(e_delta < -1e-6)
                feed_rates.append(f_curr if f_curr is not None else 0.0)
    
    if len(coords) == 0:
        return (np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=bool), 
                np.array([], dtype=bool), np.array([]))
    
    return (np.array(coords), np.array(e_deltas), np.array(extrusion_flags), 
            np.array(retraction_flags), np.array(feed_rates))


def compute_extrusion_rate_3d(coords: np.ndarray, e_deltas: np.ndarray, feed_rates: np.ndarray) -> np.ndarray:
    """
    Compute extrusion rate (E/mm) for each segment.
    """
    rates = np.zeros(len(coords))
    
    for i in range(len(coords) - 1):
        if e_deltas[i+1] > 1e-6:  # Extrusion move
            # Compute distance
            dx = coords[i+1, 0] - coords[i, 0]
            dy = coords[i+1, 1] - coords[i, 1]
            dz = coords[i+1, 2] - coords[i, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist > 1e-6:
                rates[i+1] = e_deltas[i+1] / dist
            else:
                rates[i+1] = 0.0
    
    return rates


# ============================================================================
# 3D Visualization
# ============================================================================

def create_3d_comparison(baseline_lines: List[str], stabilized_lines: List[str], 
                         output_path: Optional[Path] = None, dpi: int = 300):
    """
    Create a high-quality side-by-side 3D comparison of baseline vs stabilized toolpaths.
    """
    print("Extracting 3D toolpaths...", flush=True)
    
    # Extract toolpaths
    baseline_coords, baseline_e, baseline_ext, baseline_ret, baseline_f = extract_3d_toolpath(baseline_lines)
    stabilized_coords, stabilized_e, stabilized_ext, stabilized_ret, stabilized_f = extract_3d_toolpath(stabilized_lines)
    
    print(f"Baseline: {len(baseline_coords)} points, {np.sum(baseline_ext)} extrusions, {np.sum(baseline_ret)} retractions", flush=True)
    print(f"Stabilized: {len(stabilized_coords)} points, {np.sum(stabilized_ext)} extrusions, {np.sum(stabilized_ret)} retractions", flush=True)
    
    if len(baseline_coords) == 0 or len(stabilized_coords) == 0:
        print("ERROR: Could not extract 3D toolpath data", flush=True)
        return
    
    # Compute extrusion rates
    baseline_rates = compute_extrusion_rate_3d(baseline_coords, baseline_e, baseline_f)
    stabilized_rates = compute_extrusion_rate_3d(stabilized_coords, stabilized_e, stabilized_f)
    
    # Normalize rates for colormap
    all_rates = np.concatenate([baseline_rates[baseline_rates > 0], stabilized_rates[stabilized_rates > 0]])
    max_rate = np.max(all_rates) if len(all_rates) > 0 else 1.0
    min_rate = np.min(all_rates) if len(all_rates) > 0 else 0.0
    if max_rate < 1e-6:
        max_rate = 1.0
    
    # Create figure with publication-quality styling
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
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'figure.titlesize': 14,
        'figure.titleweight': 'bold',
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.6,
        'grid.alpha': 0.4,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
    })
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('3D Toolpath Comparison: Baseline vs Stabilized\nAll Differences Highlighted for Paper', 
                 fontsize=17, fontweight='bold', y=0.98)
    
    # ========================================================================
    # BASELINE (Left Plot)
    # ========================================================================
    ax1 = fig.add_subplot(121, projection='3d')
    
    baseline_retraction_count = 0
    baseline_extrusion_count = 0
    plot_limit = min(5000, len(baseline_coords) - 1)
    
    # Plot all moves
    for i in range(plot_limit):
        has_e = (i+1 < len(baseline_e) and abs(baseline_e[i+1]) > 1e-6)
        is_retraction = (i+1 < len(baseline_ret) and baseline_ret[i+1])
        is_extrusion = (i+1 < len(baseline_ext) and baseline_ext[i+1])
        
        if is_retraction or (has_e and baseline_e[i+1] < 0):
            # RETRACTION - Highlight in bright red
            ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                    [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                    [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                    color='red', linewidth=3.0, linestyle='--', alpha=1.0, zorder=10)
            # Mark retraction point with large X marker
            ax1.scatter([baseline_coords[i+1, 0]], [baseline_coords[i+1, 1]], [baseline_coords[i+1, 2]],
                       color='red', marker='X', s=200, linewidths=3.0, edgecolors='darkred', zorder=15)
            baseline_retraction_count += 1
        elif is_extrusion or (has_e and baseline_e[i+1] > 0):
            # EXTRUSION - Color by rate
            color_val = (baseline_rates[i+1] - min_rate) / (max_rate - min_rate) if max_rate > min_rate else 0.7
            color_val = np.clip(color_val, 0.0, 1.0)
            ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                    [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                    [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                    color=plt.cm.viridis(color_val), linewidth=2.0, alpha=0.9, zorder=2)
            baseline_extrusion_count += 1
        else:
            # TRAVEL MOVE - Light gray
            ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                    [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                    [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                    color='lightgray', linewidth=0.8, alpha=0.5, zorder=1)
    
    # Add detailed annotation showing all differences
    annotation_text = f'BASELINE STATISTICS:\n'
    annotation_text += f'Retractions: {baseline_retraction_count}\n'
    annotation_text += f'Extrusions: {baseline_extrusion_count}\n'
    annotation_text += f'Total Moves: {len(baseline_coords)}\n'
    annotation_text += f'\nKEY DIFFERENCE:\n'
    annotation_text += f'High retraction count\ncauses flow issues'
    ax1.text2D(0.02, 0.98, annotation_text, 
              transform=ax1.transAxes, fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.85, edgecolor='red', linewidth=2.5),
              color='black', fontweight='bold')
    
    ax1.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax1.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Baseline - Retractions in Red', fontsize=13, pad=10, fontweight='bold')
    ax1.view_init(elev=25, azim=45)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # ========================================================================
    # STABILIZED (Right Plot)
    # ========================================================================
    ax2 = fig.add_subplot(122, projection='3d')
    
    stabilized_extrusion_count = 0
    micro_prime_count = 0
    remaining_retractions = 0
    plot_limit = min(5000, len(stabilized_coords) - 1)
    
    # Plot all moves
    for i in range(plot_limit):
        has_e = (i+1 < len(stabilized_e) and abs(stabilized_e[i+1]) > 1e-6)
        is_retraction = (i+1 < len(stabilized_ret) and stabilized_ret[i+1])
        is_extrusion = (i+1 < len(stabilized_ext) and stabilized_ext[i+1])
        
        if is_retraction or (has_e and stabilized_e[i+1] < 0):
            # REMAINING RETRACTION (shouldn't happen) - Orange
            ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                    [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                    [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                    color='orange', linewidth=3.0, linestyle='--', alpha=1.0, zorder=10)
            remaining_retractions += 1
        elif has_e and stabilized_e[i+1] > 0:
            # EXTRUSION or MICRO-PRIME
            if abs(stabilized_e[i+1]) < 1.0:  # Small E = micro-prime
                # MICRO-PRIME - Bright green
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color='lime', linewidth=3.0, alpha=1.0, zorder=8)
                # Mark micro-prime point
                if micro_prime_count < 150:  # Limit markers for performance
                    ax2.scatter([stabilized_coords[i+1, 0]], [stabilized_coords[i+1, 1]], [stabilized_coords[i+1, 2]],
                               color='lime', marker='o', s=120, edgecolors='darkgreen', linewidths=2.5, zorder=12, alpha=0.9)
                micro_prime_count += 1
                stabilized_extrusion_count += 1
            else:
                # REGULAR EXTRUSION - Color by rate
                color_val = (stabilized_rates[i+1] - min_rate) / (max_rate - min_rate) if max_rate > min_rate else 0.7
                color_val = np.clip(color_val, 0.0, 1.0)
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color=plt.cm.plasma(color_val), linewidth=2.0, alpha=0.9, zorder=2)
                stabilized_extrusion_count += 1
        elif is_extrusion:
            # Fallback to flag
            color_val = (stabilized_rates[i+1] - min_rate) / (max_rate - min_rate) if max_rate > min_rate else 0.7
            color_val = np.clip(color_val, 0.0, 1.0)
            ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                    [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                    [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                    color=plt.cm.plasma(color_val), linewidth=2.0, alpha=0.9, zorder=2)
            stabilized_extrusion_count += 1
        else:
            # TRAVEL MOVE - Light blue
            ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                    [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                    [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                    color='lightblue', linewidth=1.0, alpha=0.6, zorder=1)
    
    # Add detailed annotation showing improvements
    improvement_text = f'STABILIZED STATISTICS:\n'
    improvement_text += f'Micro-primes: {micro_prime_count}\n'
    improvement_text += f'Extrusions: {stabilized_extrusion_count}\n'
    improvement_text += f'Total Moves: {len(stabilized_coords)}\n'
    if remaining_retractions > 0:
        improvement_text += f'\nRemaining retractions: {remaining_retractions}'
    else:
        improvement_text += f'\n\nKEY IMPROVEMENT:\n'
        improvement_text += f'[OK] All retractions eliminated\n'
        improvement_text += f'[OK] {baseline_retraction_count} -> 0 retractions\n'
        improvement_text += f'[OK] Flow continuity restored'
    
    ax2.text2D(0.02, 0.98, improvement_text, 
              transform=ax2.transAxes, fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85, edgecolor='darkgreen', linewidth=2.5),
              color='black', fontweight='bold')
    
    ax2.set_xlabel('X (mm)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Y (mm)', fontsize=12, fontweight='bold')
    ax2.set_zlabel('Z (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Stabilized - Micro-primes in Green', fontsize=13, pad=10, fontweight='bold')
    ax2.view_init(elev=25, azim=45)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # ========================================================================
    # Synchronize Axis Limits
    # ========================================================================
    if len(baseline_coords) > 0 and len(stabilized_coords) > 0:
        all_x = np.concatenate([baseline_coords[:, 0], stabilized_coords[:, 0]])
        all_y = np.concatenate([baseline_coords[:, 1], stabilized_coords[:, 1]])
        all_z = np.concatenate([baseline_coords[:, 2], stabilized_coords[:, 2]])
        
        x_range = [np.min(all_x), np.max(all_x)] if len(all_x) > 0 else [0, 100]
        y_range = [np.min(all_y), np.max(all_y)] if len(all_y) > 0 else [0, 100]
        z_range = [np.min(all_z), np.max(all_z)] if len(all_z) > 0 else [0, 10]
        
        # Add padding
        x_pad = (x_range[1] - x_range[0]) * 0.1 if x_range[1] > x_range[0] else 10
        y_pad = (y_range[1] - y_range[0]) * 0.1 if y_range[1] > y_range[0] else 10
        z_pad = (z_range[1] - z_range[0]) * 0.1 if z_range[1] > z_range[0] else 1
        
        for ax in [ax1, ax2]:
            ax.set_xlim(x_range[0] - x_pad, x_range[1] + x_pad)
            ax.set_ylim(y_range[0] - y_pad, y_range[1] + y_pad)
            ax.set_zlim(max(0, z_range[0] - z_pad), z_range[1] + z_pad)
    
    # ========================================================================
    # Add Comprehensive Legend
    # ========================================================================
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linewidth=3, linestyle='--', label='Retractions (Baseline) - Problem'),
        Line2D([0], [0], color='lime', linewidth=3, label='Micro-primes (Stabilized) - Solution'),
        Line2D([0], [0], color=plt.cm.viridis(0.7), linewidth=2, label='Extrusion moves (Baseline)'),
        Line2D([0], [0], color=plt.cm.plasma(0.7), linewidth=2, label='Extrusion moves (Stabilized)'),
        Line2D([0], [0], color='lightgray', linewidth=1, alpha=0.5, label='Travel moves'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red', markersize=12, 
               markeredgecolor='darkred', markeredgewidth=2, label='Retraction markers'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, 
               markeredgecolor='darkgreen', markeredgewidth=2, label='Micro-prime markers'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, frameon=True, 
               fancybox=False, shadow=False, framealpha=1.0, bbox_to_anchor=(0.5, 0.01), 
               columnspacing=1.5, handlelength=2.5)
    
    # ========================================================================
    # Save or Display
    # ========================================================================
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Leave space for legend
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[OK] Saved 3D comparison to: {output_path}", flush=True)
        plt.close(fig)
    else:
        # Display interactively - BLOCK until window is closed
        print("=" * 70, flush=True)
        print("Displaying 3D comparison (BLOCKING MODE)", flush=True)
        print("Close the figure window to continue...", flush=True)
        print("=" * 70, flush=True)
        plt.show(block=True)  # This will block until window is closed
        print("[OK] 3D comparison window closed", flush=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create a high-quality 3D toolpath comparison visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display interactively
  python3 3d_map.py --baseline test.gcode --stabilized results/stabilized.gcode
  
  # Save to file
  python3 3d_map.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/3d_comparison.png
        """
    )
    
    parser.add_argument('--baseline', '--baseline-gcode', dest='baseline_gcode', required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized', '--stabilized-gcode', dest='stabilized_gcode', required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--output', '-o', dest='output', default=None,
                       help='Output path for saved figure (if not provided, displays interactively)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure (default: 300)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    baseline_path = Path(args.baseline_gcode)
    if not baseline_path.is_absolute():
        test_path = script_dir / args.baseline_gcode
        if test_path.exists():
            baseline_path = test_path
    
    stabilized_path = Path(args.stabilized_gcode)
    if not stabilized_path.is_absolute():
        test_path = script_dir / args.stabilized_gcode
        if test_path.exists():
            stabilized_path = test_path
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline G-code not found: {baseline_path}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    if not stabilized_path.exists():
        print(f"ERROR: Stabilized G-code not found: {stabilized_path}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Read G-code files
    print(f"Reading baseline G-code: {baseline_path}", flush=True)
    with open(baseline_path, 'r', encoding='utf-8', errors='replace') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_path}", flush=True)
    with open(stabilized_path, 'r', encoding='utf-8', errors='replace') as f:
        stabilized_lines = f.readlines()
    
    print(f"Loaded {len(baseline_lines)} baseline lines, {len(stabilized_lines)} stabilized lines", flush=True)
    
    # Create visualization
    output_path = Path(args.output) if args.output else None
    create_3d_comparison(baseline_lines, stabilized_lines, output_path, args.dpi)
    
    print("[OK] Done", flush=True)


if __name__ == '__main__':
    main()

