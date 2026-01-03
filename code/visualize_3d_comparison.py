#!/usr/bin/env python3
"""
Enhanced 3D visualization tool for before/after stabilization comparison.
Creates publication-quality 3D maps showing the transformation.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import sys

# Publication-quality settings
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 2.0

def parse_float_token(line: str, letter: str):
    """Extract float value for a G-code token."""
    m = re.search(rf"{letter}([-+]?\d*\.?\d+)", line, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))

def strip_comment(line: str):
    """Remove comments from G-code line."""
    if ";" in line:
        return line.split(";", 1)[0].strip()
    return line.strip()

def is_move(code: str):
    """Check if line is a G0 or G1 move command."""
    return bool(re.match(r"^(G0|G1)\s", code, re.IGNORECASE))

def extract_toolpath(gcode_file: Path):
    """Extract 3D coordinates and extrusion values from G-code."""
    coords = []
    e_values = []
    feed_rates = []
    segment_flags = []
    retraction_flags = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    e_cumulative = 0.0
    is_relative_e = True
    f_curr = None
    
    lines = gcode_file.read_text(encoding="utf-8", errors="replace").splitlines()
    
    for ln in lines:
        code = strip_comment(ln)
        if not code:
            continue
        
        if "M83" in code.upper():
            is_relative_e = True
        elif "M82" in code.upper():
            is_relative_e = False
        
        if "G28" in code.upper():
            x_curr, y_curr, z_curr = None, None, 0.0
        elif "G92" in code.upper():
            x_set = parse_float_token(code, "X")
            y_set = parse_float_token(code, "Y")
            z_set = parse_float_token(code, "Z")
            if x_set is not None:
                x_curr = x_set
            if y_set is not None:
                y_curr = y_set
            if z_set is not None:
                z_curr = z_set
        
        if is_move(code):
            x = parse_float_token(code, "X")
            y = parse_float_token(code, "Y")
            z = parse_float_token(code, "Z")
            e = parse_float_token(code, "E")
            f = parse_float_token(code, "F")
            
            x_prev, y_prev, z_prev = x_curr, y_curr, z_curr
            
            if x is not None:
                x_curr = x
            if y is not None:
                y_curr = y
            if z is not None:
                z_curr = z
            if f is not None:
                f_curr = f
            
            e_delta = 0.0
            has_extrusion = False
            is_retraction = False
            if e is not None:
                if is_relative_e:
                    e_delta = e
                    e_cumulative += e
                else:
                    e_delta = e - e_cumulative
                    e_cumulative = e
                has_extrusion = (e_delta > 1e-6)
                is_retraction = (e_delta < -1e-6)
            
            x_to_use = x_curr if x_curr is not None else x_prev
            y_to_use = y_curr if y_curr is not None else y_prev
            z_to_use = z_curr if z_curr is not None else (z_prev if z_prev is not None else 0.0)
            
            should_record = False
            if x_to_use is not None and y_to_use is not None:
                should_record = True
            elif (z is not None or e is not None) and (x_prev is not None or y_prev is not None):
                x_to_use = x_prev if x_to_use is None else x_to_use
                y_to_use = y_prev if y_to_use is None else y_to_use
                should_record = True
            
            if should_record and x_to_use is not None and y_to_use is not None:
                coords.append([x_to_use, y_to_use, z_to_use])
                e_values.append(e_delta)
                feed_rates.append(f_curr if f_curr is not None else 0.0)
                segment_flags.append(has_extrusion)
                retraction_flags.append(is_retraction)
    
    if len(coords) == 0:
        return None, None, None, None, None
    
    return (np.array(coords), np.array(e_values), np.array(feed_rates), 
            np.array(segment_flags), np.array(retraction_flags))

def compute_extrusion_rate(coords, e_values, feed_rates):
    """Compute extrusion rate (E per mm) along toolpath."""
    if len(coords) < 2:
        return np.zeros_like(e_values)
    
    distances = np.zeros(len(coords))
    distances[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    
    rates = np.zeros_like(e_values)
    mask = distances > 1e-6
    rates[mask] = np.abs(e_values[mask]) / distances[mask]
    
    return rates

def create_3d_comparison(original_file: Path, stabilized_file: Path, output_file: Path):
    """Create enhanced 3D before/after comparison visualization."""
    
    print(f"Reading original G-code: {original_file}")
    orig_coords, orig_e, orig_f, orig_flags, orig_retract = extract_toolpath(original_file)
    
    print(f"Reading stabilized G-code: {stabilized_file}")
    stab_coords, stab_e, stab_f, stab_flags, stab_retract = extract_toolpath(stabilized_file)
    
    if orig_coords is None or stab_coords is None:
        print("ERROR: Could not extract toolpaths")
        return False
    
    # Compute extrusion rates
    orig_rates = compute_extrusion_rate(orig_coords, orig_e, orig_f)
    stab_rates = compute_extrusion_rate(stab_coords, stab_e, stab_f)
    
    # Normalize rates for color mapping
    all_rates = np.concatenate([orig_rates[orig_rates > 0], stab_rates[stab_rates > 0]])
    if len(all_rates) > 0:
        vmin, vmax = 0, np.percentile(all_rates, 95)
    else:
        vmin, vmax = 0, 1
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 10))
    
    # ===== LEFT PANEL: ORIGINAL (BEFORE) =====
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot extrusion segments
    for i in range(len(orig_coords) - 1):
        if orig_flags[i] and orig_flags[i+1]:
            rate = (orig_rates[i] + orig_rates[i+1]) / 2
            if rate > 1e-6:
                color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                ax1.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                        [orig_coords[i, 1], orig_coords[i+1, 1]],
                        [orig_coords[i, 2], orig_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=2.5, alpha=0.9)
        elif not orig_flags[i] and not orig_flags[i+1]:
            ax1.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                    [orig_coords[i, 1], orig_coords[i+1, 1]],
                    [orig_coords[i, 2], orig_coords[i+1, 2]],
                    'gray', alpha=0.2, linewidth=0.5)
    
    # Highlight retractions with red markers
    retract_indices = np.where(orig_retract)[0]
    if len(retract_indices) > 0:
        ax1.scatter(orig_coords[retract_indices, 0],
                   orig_coords[retract_indices, 1],
                   orig_coords[retract_indices, 2],
                   c='red', marker='x', s=100, linewidths=3, 
                   label=f'Retractions ({len(retract_indices)})', zorder=10)
    
    # Add extrusion points
    extr_indices = np.where(orig_flags)[0]
    if len(extr_indices) > 0:
        ax1.scatter(orig_coords[extr_indices, 0],
                   orig_coords[extr_indices, 1],
                   orig_coords[extr_indices, 2],
                   c=orig_rates[extr_indices], cmap='viridis', s=20, alpha=0.6,
                   vmin=vmin, vmax=vmax, zorder=5)
    
    ax1.set_xlabel('X (mm)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Y (mm)', fontweight='bold', fontsize=12)
    ax1.set_zlabel('Z (mm)', fontweight='bold', fontsize=12)
    ax1.set_title('BEFORE Stabilization\n(Original G-code)', fontweight='bold', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    
    # Set equal aspect ratio
    max_range = np.array([orig_coords[:, 0].max() - orig_coords[:, 0].min(),
                          orig_coords[:, 1].max() - orig_coords[:, 1].min(),
                          orig_coords[:, 2].max() - orig_coords[:, 2].min()]).max() / 2.0
    mid_x = (orig_coords[:, 0].max() + orig_coords[:, 0].min()) * 0.5
    mid_y = (orig_coords[:, 1].max() + orig_coords[:, 1].min()) * 0.5
    mid_z = (orig_coords[:, 2].max() + orig_coords[:, 2].min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # ===== RIGHT PANEL: STABILIZED (AFTER) =====
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot extrusion segments
    for i in range(len(stab_coords) - 1):
        if stab_flags[i] and stab_flags[i+1]:
            rate = (stab_rates[i] + stab_rates[i+1]) / 2
            if rate > 1e-6:
                color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                ax2.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                        [stab_coords[i, 1], stab_coords[i+1, 1]],
                        [stab_coords[i, 2], stab_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=2.5, alpha=0.9)
        elif not stab_flags[i] and not stab_flags[i+1]:
            ax2.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                    [stab_coords[i, 1], stab_coords[i+1, 1]],
                    [stab_coords[i, 2], stab_coords[i+1, 2]],
                    'gray', alpha=0.2, linewidth=0.5)
    
    # Highlight micro-primes (pure E moves) with green markers
    # Find pure E moves (extrusion without XY movement)
    distances = np.zeros(len(stab_coords))
    distances[1:] = np.linalg.norm(np.diff(stab_coords, axis=0), axis=1)
    pure_e_indices = np.where((stab_flags) & (distances < 1e-6))[0]
    
    if len(pure_e_indices) > 0:
        ax2.scatter(stab_coords[pure_e_indices, 0],
                   stab_coords[pure_e_indices, 1],
                   stab_coords[pure_e_indices, 2],
                   c='green', marker='o', s=80, edgecolors='darkgreen', 
                   linewidths=2, label=f'Micro-primes ({len(pure_e_indices)})', zorder=10)
    
    # Add extrusion points
    extr_indices = np.where(stab_flags)[0]
    if len(extr_indices) > 0:
        ax2.scatter(stab_coords[extr_indices, 0],
                   stab_coords[extr_indices, 1],
                   stab_coords[extr_indices, 2],
                   c=stab_rates[extr_indices], cmap='viridis', s=20, alpha=0.6,
                   vmin=vmin, vmax=vmax, zorder=5)
    
    ax2.set_xlabel('X (mm)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Y (mm)', fontweight='bold', fontsize=12)
    ax2.set_zlabel('Z (mm)', fontweight='bold', fontsize=12)
    ax2.set_title('AFTER Stabilization\n(Stabilized G-code)', fontweight='bold', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    
    # Set equal aspect ratio
    max_range = np.array([stab_coords[:, 0].max() - stab_coords[:, 0].min(),
                          stab_coords[:, 1].max() - stab_coords[:, 1].min(),
                          stab_coords[:, 2].max() - stab_coords[:, 2].min()]).max() / 2.0
    mid_x = (stab_coords[:, 0].max() + stab_coords[:, 0].min()) * 0.5
    mid_y = (stab_coords[:, 1].max() + stab_coords[:, 1].min()) * 0.5
    mid_z = (stab_coords[:, 2].max() + stab_coords[:, 2].min()) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=[ax1, ax2], pad=0.1, aspect=30)
    cbar.set_label('Extrusion Rate (E/mm)', fontweight='bold', fontsize=11, rotation=270, labelpad=20)
    
    # Add overall title
    fig.suptitle('3D Toolpath Comparison: Before vs After Stabilization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
    fig.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved 3D comparison: {output_file.with_suffix('.pdf')}")
    print(f"✓ Saved 3D comparison: {output_file.with_suffix('.png')}")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Original: {len(orig_coords)} points, {orig_flags.sum()} extrusion moves, {orig_retract.sum()} retractions")
    print(f"  Stabilized: {len(stab_coords)} points, {stab_flags.sum()} extrusion moves, {stab_retract.sum()} retractions")
    print(f"  Micro-primes added: {len(pure_e_indices)}")
    
    return True

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Create enhanced 3D before/after comparison visualization.")
    ap.add_argument("--original", required=True, help="Original G-code file")
    ap.add_argument("--stabilized", required=True, help="Stabilized G-code file")
    ap.add_argument("--output", default="results/figures/3d_comparison", help="Output file path (without extension)")
    
    args = ap.parse_args()
    
    orig_path = Path(args.original)
    stab_path = Path(args.stabilized)
    out_path = Path(args.output)
    
    if not orig_path.exists():
        print(f"ERROR: Original file not found: {orig_path}")
        sys.exit(1)
    
    if not stab_path.exists():
        print(f"ERROR: Stabilized file not found: {stab_path}")
        sys.exit(1)
    
    create_3d_comparison(orig_path, stab_path, out_path)

if __name__ == "__main__":
    main()

