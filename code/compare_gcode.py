#!/usr/bin/env python3
"""
Comprehensive comparison tool for original vs stabilized G-code.
Shows 3D toolpaths, extrusion rates, and statistics.
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

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
    """Extract 3D coordinates and extrusion values from G-code.
    Returns: coords, e_values, feed_rates, segment_flags
    segment_flags: True=extrusion (E>0), False=travel or retraction (E<=0)
    """
    coords = []
    e_values = []
    feed_rates = []
    segment_flags = []
    retraction_flags = []  # Track retractions separately
    
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
            
            has_position_change = (x is not None or y is not None or z is not None) and \
                                 (x_prev != x_curr or y_prev != y_curr or z_prev != z_curr)
            
            # Record moves with position change OR any E value (including retractions)
            # Use last known X/Y if current move doesn't specify them
            x_to_use = x_curr if x_curr is not None else x_prev
            y_to_use = y_curr if y_curr is not None else y_prev
            z_to_use = z_curr if z_curr is not None else (z_prev if z_prev is not None else 0.0)
            
            # Record if we have valid coordinates (at least X and Y, or Z change with E)
            should_record = False
            if x_to_use is not None and y_to_use is not None:
                should_record = True
            elif (z is not None or e is not None) and (x_prev is not None or y_prev is not None):
                # Z-only or E-only move, but we have previous position
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

def plot_comparison(original_file: Path, stabilized_file: Path, output_dir: Path):
    """Create comprehensive comparison visualizations."""
    
    print(f"Reading original G-code: {original_file}")
    orig_coords, orig_e, orig_f, orig_flags, orig_retract = extract_toolpath(original_file)
    
    print(f"Reading stabilized G-code: {stabilized_file}")
    stab_coords, stab_e, stab_f, stab_flags, stab_retract = extract_toolpath(stabilized_file)
    
    if orig_coords is None or stab_coords is None:
        print("ERROR: Could not extract toolpaths")
        return
    
    # Compute rates
    orig_rates = compute_extrusion_rate(orig_coords, orig_e, orig_f)
    stab_rates = compute_extrusion_rate(stab_coords, stab_e, stab_f)
    
    # Categorize moves better
    def categorize_moves(coords, e_values, flags, rates, retract_flags):
        """Categorize moves into: pure_E, extrusion_with_XY, travel, retractions"""
        pure_E = []  # E > 0 but no XY movement (distance = 0)
        extrusion_XY = []  # E > 0 with XY movement
        retractions = []  # E < 0 (retractions)
        travel = []  # E = 0
        
        # Compute distances: distances[i] = distance from point i-1 to point i
        distances = np.zeros(len(coords))
        if len(coords) > 1:
            distances[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        
        for i in range(len(coords)):
            if retract_flags[i]:  # Retraction (negative E)
                retractions.append(i)
            elif flags[i]:  # Has extrusion (positive E)
                # Check if the move TO this point had XY movement
                # distances[i] = distance from point i-1 to point i (0 for first point)
                if i == 0:
                    # First point: assume it's not pure E (or check if same as next)
                    has_XY = True  # Default to having XY, will be corrected if needed
                    if len(coords) > 1:
                        # If next point is at same location, this might be pure E
                        if np.linalg.norm(coords[1] - coords[0]) < 1e-6:
                            has_XY = False
                else:
                    has_XY = (distances[i] > 1e-6)
                
                if not has_XY:  # Pure E move (no XY movement)
                    pure_E.append(i)
                else:  # Had XY movement
                    extrusion_XY.append(i)
            else:  # No extrusion, no retraction = travel
                travel.append(i)
        
        return np.array(pure_E), np.array(extrusion_XY), np.array(travel), np.array(retractions)
    
    orig_pure_E, orig_extrusion_XY, orig_travel, orig_retractions = categorize_moves(
        orig_coords, orig_e, orig_flags, orig_rates, orig_retract)
    stab_pure_E, stab_extrusion_XY, stab_travel, stab_retractions = categorize_moves(
        stab_coords, stab_e, stab_flags, stab_rates, stab_retract)
    
    # Statistics
    print("\n=== Statistics ===")
    print(f"Original: {len(orig_coords)} points")
    print(f"  - Extrusion with XY: {len(orig_extrusion_XY)}")
    print(f"  - Pure E moves: {len(orig_pure_E)}")
    print(f"  - Retractions: {len(orig_retractions)} (negative E moves)")
    print(f"  - Travel moves: {len(orig_travel)}")
    print(f"  - E>0: {(orig_e > 0).sum()}, E<0: {(orig_e < 0).sum()}, E=0: {(orig_e == 0).sum()}")
    if len(orig_retractions) > 0:
        print(f"  - Retraction E values: min={orig_e[orig_retractions].min():.4f}, "
              f"max={orig_e[orig_retractions].max():.4f}, mean={orig_e[orig_retractions].mean():.4f}")
    
    print(f"\nStabilized: {len(stab_coords)} points")
    print(f"  - Extrusion with XY: {len(stab_extrusion_XY)}")
    print(f"  - Pure E moves (micro-primes): {len(stab_pure_E)}")
    print(f"  - Retractions: {len(stab_retractions)} (should be 0)")
    print(f"  - Travel moves: {len(stab_travel)}")
    print(f"  - E>0: {(stab_e > 0).sum()}, E<0: {(stab_e < 0).sum()}, E=0: {(stab_e == 0).sum()}")
    
    if len(orig_extrusion_XY) > 0:
        print(f"\nOriginal extrusion rate (with XY): min={orig_rates[orig_extrusion_XY].min():.4f}, "
              f"max={orig_rates[orig_extrusion_XY].max():.4f}, "
              f"mean={orig_rates[orig_extrusion_XY].mean():.4f}")
    if len(stab_extrusion_XY) > 0:
        print(f"Stabilized extrusion rate (with XY): min={stab_rates[stab_extrusion_XY].min():.4f}, "
              f"max={stab_rates[stab_extrusion_XY].max():.4f}, "
              f"mean={stab_rates[stab_extrusion_XY].mean():.4f}")
    
    if len(stab_pure_E) > 0:
        print(f"\nPure E moves (micro-primes) in stabilized: {len(stab_pure_E)}")
        print(f"  E values: min={stab_e[stab_pure_E].min():.4f}, "
              f"max={stab_e[stab_pure_E].max():.4f}, "
              f"mean={stab_e[stab_pure_E].mean():.4f}")
    
    # Set color scale (use same for both)
    all_rates = []
    if orig_flags.any():
        all_rates.extend(orig_rates[orig_flags])
    if stab_flags.any():
        all_rates.extend(stab_rates[stab_flags])
    
    if len(all_rates) > 0:
        vmin, vmax = 0, np.percentile(all_rates, 95)
    else:
        vmin, vmax = 0, 1
    
    # Create sets for faster lookup (used in multiple plots)
    stab_pure_E_set = set(stab_pure_E)
    stab_extrusion_XY_set = set(stab_extrusion_XY)
    
    # === Plot 1: Side-by-side 3D comparison ===
    fig = plt.figure(figsize=(20, 8))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    for i in range(len(orig_coords) - 1):
        if orig_flags[i] and orig_flags[i+1]:
            # Extrusion moves
            rate = (orig_rates[i] + orig_rates[i+1]) / 2
            if rate > 1e-6:
                color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                ax1.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                        [orig_coords[i, 1], orig_coords[i+1, 1]],
                        [orig_coords[i, 2], orig_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=2.0, alpha=0.8)
        elif orig_retract[i] or orig_retract[i+1]:
            # Retraction moves - show in red
            ax1.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                    [orig_coords[i, 1], orig_coords[i+1, 1]],
                    [orig_coords[i, 2], orig_coords[i+1, 2]],
                    'red', alpha=0.4, linewidth=1.5, linestyle='--')
        elif not orig_flags[i] and not orig_flags[i+1]:
            # Travel moves
            ax1.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                    [orig_coords[i, 1], orig_coords[i+1, 1]],
                    [orig_coords[i, 2], orig_coords[i+1, 2]],
                    'gray', alpha=0.15, linewidth=0.3)
    
    # Show extrusion moves
    if orig_flags.any():
        scatter1 = ax1.scatter(orig_coords[orig_flags, 0], orig_coords[orig_flags, 1], 
                               orig_coords[orig_flags, 2],
                               c=orig_rates[orig_flags], cmap='viridis', s=20, alpha=0.9,
                               edgecolors='black', linewidths=0.3, vmin=vmin, vmax=vmax)
    
    # Show retractions
    if orig_retract.any():
        ax1.scatter(orig_coords[orig_retract, 0], orig_coords[orig_retract, 1], 
                   orig_coords[orig_retract, 2],
                   c='red', s=15, alpha=0.7, marker='x', linewidths=1.5, label='Retractions')
    
    ax1.set_xlabel('X (mm)', fontsize=10)
    ax1.set_ylabel('Y (mm)', fontsize=10)
    ax1.set_zlabel('Z (mm)', fontsize=10)
    title1 = f'Original G-code\n{len(orig_extrusion_XY)} extrusion, {len(orig_retractions)} retractions, {len(orig_travel)} travel'
    ax1.set_title(title1, fontsize=11)
    if orig_flags.any():
        cbar1 = plt.colorbar(scatter1, ax=ax1, label='Extrusion Rate', shrink=0.6)
        cbar1.ax.tick_params(labelsize=8)
    
    # Stabilized - Show micro-primes (replaced retractions) prominently
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot micro-primes (pure E moves) - these replaced retractions
    for i in range(len(stab_coords) - 1):
        # Check if either point is a micro-prime (pure E)
        is_microprime_i = i in stab_pure_E_set
        is_microprime_next = (i+1) in stab_pure_E_set
        
        if is_microprime_i or is_microprime_next:
            # Micro-prime moves - show in green (replacement for retractions)
            ax2.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                    [stab_coords[i, 1], stab_coords[i+1, 1]],
                    [stab_coords[i, 2], stab_coords[i+1, 2]],
                    'green', alpha=0.5, linewidth=1.5, linestyle='-')
        elif i in stab_extrusion_XY_set and (i+1) in stab_extrusion_XY_set:
            # Extrusion with XY movement
            rate = (stab_rates[i] + stab_rates[i+1]) / 2
            if rate > 1e-6:
                color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                ax2.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                        [stab_coords[i, 1], stab_coords[i+1, 1]],
                        [stab_coords[i, 2], stab_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=2.0, alpha=0.8)
        elif not stab_flags[i] and not stab_flags[i+1]:
            # Travel moves
            ax2.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                    [stab_coords[i, 1], stab_coords[i+1, 1]],
                    [stab_coords[i, 2], stab_coords[i+1, 2]],
                    'gray', alpha=0.15, linewidth=0.3)
    
    # Show micro-primes as green markers (replaced retractions)
    if len(stab_pure_E) > 0:
        ax2.scatter(stab_coords[stab_pure_E, 0], stab_coords[stab_pure_E, 1], 
                   stab_coords[stab_pure_E, 2],
                   c='green', s=15, alpha=0.8, marker='o', linewidths=1.0, 
                   edgecolors='darkgreen', label='Micro-primes (replaced retractions)')
    
    # Show extrusion with XY (if any) - less prominent
    if len(stab_extrusion_XY) > 0:
        scatter2 = ax2.scatter(stab_coords[stab_extrusion_XY, 0], 
                              stab_coords[stab_extrusion_XY, 1], 
                              stab_coords[stab_extrusion_XY, 2],
                              c=stab_rates[stab_extrusion_XY], cmap='viridis', s=20, alpha=0.7,
                              edgecolors='black', linewidths=0.3, vmin=vmin, vmax=vmax)
    
    ax2.set_xlabel('X (mm)', fontsize=10)
    ax2.set_ylabel('Y (mm)', fontsize=10)
    ax2.set_zlabel('Z (mm)', fontsize=10)
    title2 = f'Stabilized G-code\n{len(stab_pure_E)} micro-primes (replaced retractions), {len(stab_extrusion_XY)} extrusion, {len(stab_travel)} travel'
    ax2.set_title(title2, fontsize=11)
    if len(stab_extrusion_XY) > 0:
        cbar2 = plt.colorbar(scatter2, ax=ax2, label='Extrusion Rate', shrink=0.6)
        cbar2.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_3d_toolpath.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_3d_toolpath.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_3d_toolpath.png/pdf")
    
    # === Plot 2: Extrusion rate comparison ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Extrusion values histogram
    ax = axes[0, 0]
    if orig_e.any():
        ax.hist(orig_e[orig_e != 0], bins=50, alpha=0.7, label='Original', color='blue', edgecolor='black')
    if stab_e.any():
        ax.hist(stab_e[stab_e != 0], bins=50, alpha=0.7, label='Stabilized', color='green', edgecolor='black')
    ax.set_xlabel('Extrusion Value (E)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Extrusion Values Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Extrusion rate histogram
    ax = axes[0, 1]
    if orig_flags.any():
        ax.hist(orig_rates[orig_flags], bins=50, alpha=0.7, label='Original', color='blue', edgecolor='black')
    if stab_flags.any():
        ax.hist(stab_rates[stab_flags], bins=50, alpha=0.7, label='Stabilized', color='green', edgecolor='black')
    ax.set_xlabel('Extrusion Rate (E/mm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Extrusion Rate Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative extrusion over time (approximate)
    ax = axes[1, 0]
    orig_cumulative_e = np.cumsum(np.abs(orig_e))
    stab_cumulative_e = np.cumsum(np.abs(stab_e))
    ax.plot(orig_cumulative_e, label='Original', linewidth=2, alpha=0.8)
    ax.plot(stab_cumulative_e, label='Stabilized', linewidth=2, alpha=0.8)
    ax.set_xlabel('Move Number', fontsize=11)
    ax.set_ylabel('Cumulative |E|', fontsize=11)
    ax.set_title('Cumulative Extrusion Over Moves', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Statistics comparison
    ax = axes[1, 1]
    ax.axis('off')
    
    # Build statistics text with better categorization
    orig_extrusion_rate_text = 'No extrusion'
    if len(orig_extrusion_XY) > 0:
        orig_extrusion_rate_text = f'{orig_rates[orig_extrusion_XY].min():.4f} - {orig_rates[orig_extrusion_XY].max():.4f}'
    
    stab_extrusion_rate_text = 'No extrusion'
    if len(stab_extrusion_XY) > 0:
        stab_extrusion_rate_text = f'{stab_rates[stab_extrusion_XY].min():.4f} - {stab_rates[stab_extrusion_XY].max():.4f}'
    
    stab_pure_e_text = ''
    if len(stab_pure_E) > 0:
        stab_pure_e_text = f'\n  Pure E moves: {len(stab_pure_E)} (E={stab_e[stab_pure_E].mean():.3f} avg)'
    
    orig_retract_text = ''
    if len(orig_retractions) > 0:
        orig_retract_text = f'\n  Retraction E: {orig_e[orig_retractions].mean():.3f} avg'
    
    stats_text = f"""
Statistics Comparison:

Original G-code:
  Total moves: {len(orig_coords)}
  Extrusion with XY: {len(orig_extrusion_XY)}
  Pure E moves: {len(orig_pure_E)}
  Retractions: {len(orig_retractions)}{orig_retract_text}
  Travel moves: {len(orig_travel)}
  Positive E: {(orig_e > 0).sum()}
  Negative E: {(orig_e < 0).sum()}
  Zero E: {(orig_e == 0).sum()}
  Total |E|: {np.abs(orig_e).sum():.2f}
  Extrusion rate (XY moves): {orig_extrusion_rate_text}

Stabilized G-code:
  Total moves: {len(stab_coords)}
  Extrusion with XY: {len(stab_extrusion_XY)}
  Pure E moves: {len(stab_pure_E)}
  Retractions: {len(stab_retractions)} (removed)
  Travel moves: {len(stab_travel)}
  Positive E: {(stab_e > 0).sum()}
  Negative E: {(stab_e < 0).sum()}
  Zero E: {(stab_e == 0).sum()}
  Total |E|: {np.abs(stab_e).sum():.2f}
  Extrusion rate (XY moves): {stab_extrusion_rate_text}{stab_pure_e_text}

Changes:
  Moves added: {len(stab_coords) - len(orig_coords)}
  Extrusion moves: {stab_flags.sum()} (was {orig_flags.sum()})
  Retractions removed: {len(orig_retractions)}
  Micro-primes added: {len(stab_pure_E)}
"""
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
            verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_statistics.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_statistics.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_statistics.png/pdf")
    
    # === Plot 3: Retraction vs Micro-prime Comparison ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 3a: Retraction locations in original (2D top view)
    ax = axes[0, 0]
    if len(orig_retractions) > 0:
        ax.scatter(orig_coords[orig_retractions, 0], orig_coords[orig_retractions, 1],
                  c='red', s=50, alpha=0.7, marker='x', linewidths=2, label='Retractions')
        # Draw retraction paths
        for i in range(len(orig_coords) - 1):
            if orig_retract[i] or orig_retract[i+1]:
                ax.plot([orig_coords[i, 0], orig_coords[i+1, 0]],
                       [orig_coords[i, 1], orig_coords[i+1, 1]],
                       'r--', alpha=0.3, linewidth=1)
    if orig_flags.any():
        ax.scatter(orig_coords[orig_flags, 0], orig_coords[orig_flags, 1],
                  c='blue', s=10, alpha=0.3, label='Extrusion moves')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title(f'Original: Retraction Locations (Top View)\n{len(orig_retractions)} retractions', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 3b: Micro-prime locations in stabilized (2D top view)
    ax = axes[0, 1]
    if len(stab_pure_E) > 0:
        ax.scatter(stab_coords[stab_pure_E, 0], stab_coords[stab_pure_E, 1],
                  c='green', s=50, alpha=0.7, marker='o', linewidths=1.5, 
                  edgecolors='darkgreen', label='Micro-primes')
        # Draw micro-prime paths
        for i in range(len(stab_coords) - 1):
            if i in stab_pure_E_set or (i+1) in stab_pure_E_set:
                ax.plot([stab_coords[i, 0], stab_coords[i+1, 0]],
                       [stab_coords[i, 1], stab_coords[i+1, 1]],
                       'g-', alpha=0.3, linewidth=1)
    if stab_flags.any():
        ax.scatter(stab_coords[stab_flags, 0], stab_coords[stab_flags, 1],
                  c='blue', s=10, alpha=0.3, label='Extrusion moves')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title(f'Stabilized: Micro-prime Locations (Top View)\n{len(stab_pure_E)} micro-primes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 3c: E value comparison (retractions vs micro-primes)
    ax = axes[1, 0]
    if len(orig_retractions) > 0:
        ax.hist(orig_e[orig_retractions], bins=30, alpha=0.7, color='red', 
               edgecolor='darkred', label=f'Retractions ({len(orig_retractions)})')
    if len(stab_pure_E) > 0:
        ax.hist(stab_e[stab_pure_E], bins=30, alpha=0.7, color='green',
               edgecolor='darkgreen', label=f'Micro-primes ({len(stab_pure_E)})')
    ax.set_xlabel('E Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Retraction vs Micro-prime E Values', fontsize=12)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3d: Spatial distribution comparison
    ax = axes[1, 1]
    if len(orig_retractions) > 0:
        ax.scatter(orig_coords[orig_retractions, 0], orig_coords[orig_retractions, 1],
                  c='red', s=30, alpha=0.6, marker='x', linewidths=2, label='Original retractions')
    if len(stab_pure_E) > 0:
        ax.scatter(stab_coords[stab_pure_E, 0], stab_coords[stab_pure_E, 1],
                  c='green', s=30, alpha=0.6, marker='o', linewidths=1.5,
                  edgecolors='darkgreen', label='Stabilized micro-primes')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title('Spatial Distribution: Retractions → Micro-primes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_retractions.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_retractions.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_retractions.png/pdf")
    
    # === Plot 4: Feed Rate Comparison ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 4a: Feed rate distribution
    ax = axes[0, 0]
    orig_f_nonzero = orig_f[orig_f > 0]
    stab_f_nonzero = stab_f[stab_f > 0]
    if len(orig_f_nonzero) > 0:
        ax.hist(orig_f_nonzero, bins=50, alpha=0.7, color='blue', edgecolor='black',
               label=f'Original ({len(orig_f_nonzero)} moves)', density=True)
    if len(stab_f_nonzero) > 0:
        ax.hist(stab_f_nonzero, bins=50, alpha=0.7, color='green', edgecolor='black',
               label=f'Stabilized ({len(stab_f_nonzero)} moves)', density=True)
    ax.set_xlabel('Feed Rate (mm/min)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Feed Rate Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4b: Feed rate over moves
    ax = axes[0, 1]
    if len(orig_f_nonzero) > 0:
        orig_f_indices = np.where(orig_f > 0)[0]
        ax.plot(orig_f_indices, orig_f_nonzero, 'b-', alpha=0.5, linewidth=1, label='Original')
    if len(stab_f_nonzero) > 0:
        stab_f_indices = np.where(stab_f > 0)[0]
        ax.plot(stab_f_indices, stab_f_nonzero, 'g-', alpha=0.5, linewidth=1, label='Stabilized')
    ax.set_xlabel('Move Number', fontsize=11)
    ax.set_ylabel('Feed Rate (mm/min)', fontsize=11)
    ax.set_title('Feed Rate Over Moves', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4c: Feed rate statistics
    ax = axes[1, 0]
    ax.axis('off')
    feed_stats = []
    if len(orig_f_nonzero) > 0:
        feed_stats.append("Original Feed Rates:")
        feed_stats.append(f"  Mean: {orig_f_nonzero.mean():.1f} mm/min")
        feed_stats.append(f"  Median: {np.median(orig_f_nonzero):.1f} mm/min")
        feed_stats.append(f"  Min: {orig_f_nonzero.min():.1f} mm/min")
        feed_stats.append(f"  Max: {orig_f_nonzero.max():.1f} mm/min")
        feed_stats.append(f"  Std: {orig_f_nonzero.std():.1f} mm/min")
        feed_stats.append("")
    if len(stab_f_nonzero) > 0:
        feed_stats.append("Stabilized Feed Rates:")
        feed_stats.append(f"  Mean: {stab_f_nonzero.mean():.1f} mm/min")
        feed_stats.append(f"  Median: {np.median(stab_f_nonzero):.1f} mm/min")
        feed_stats.append(f"  Min: {stab_f_nonzero.min():.1f} mm/min")
        feed_stats.append(f"  Max: {stab_f_nonzero.max():.1f} mm/min")
        feed_stats.append(f"  Std: {stab_f_nonzero.std():.1f} mm/min")
    if len(feed_stats) > 0:
        ax.text(0.1, 0.5, '\n'.join(feed_stats), fontsize=11, family='monospace',
               verticalalignment='center', transform=ax.transAxes)
    
    # Plot 4d: Feed rate changes (if we can detect scaling)
    ax = axes[1, 1]
    # Compare feed rates at similar positions (approximate)
    if len(orig_f_nonzero) > 0 and len(stab_f_nonzero) > 0:
        # Sample comparison (simplified)
        orig_sample = orig_f_nonzero[:min(1000, len(orig_f_nonzero))]
        stab_sample = stab_f_nonzero[:min(1000, len(stab_f_nonzero))]
        ax.scatter(orig_sample, stab_sample[:len(orig_sample)], alpha=0.5, s=10)
        # Add diagonal line
        max_val = max(orig_sample.max(), stab_sample[:len(orig_sample)].max())
        ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='No change')
        ax.set_xlabel('Original Feed Rate (mm/min)', fontsize=11)
        ax.set_ylabel('Stabilized Feed Rate (mm/min)', fontsize=11)
        ax.set_title('Feed Rate Comparison (Sample)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Insufficient feed rate data', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_feedrates.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_feedrates.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_feedrates.png/pdf")
    
    # === Plot 5: Geometry Preservation Analysis ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 5a: Toolpath overlay (2D top view)
    ax = axes[0, 0]
    # Original toolpath
    if orig_flags.any():
        orig_extrusion_coords = orig_coords[orig_flags]
        ax.plot(orig_extrusion_coords[:, 0], orig_extrusion_coords[:, 1],
               'b-', alpha=0.4, linewidth=1, label='Original extrusion')
    # Stabilized toolpath
    if stab_flags.any():
        stab_extrusion_coords = stab_coords[stab_flags]
        ax.plot(stab_extrusion_coords[:, 0], stab_extrusion_coords[:, 1],
               'g-', alpha=0.4, linewidth=1, label='Stabilized extrusion')
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_title('Toolpath Overlay (Top View)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Plot 5b: Z-height comparison
    ax = axes[0, 1]
    ax.plot(orig_coords[:, 2], 'b-', alpha=0.6, linewidth=1, label='Original Z')
    ax.plot(stab_coords[:len(orig_coords), 2], 'g-', alpha=0.6, linewidth=1, label='Stabilized Z')
    ax.set_xlabel('Move Number', fontsize=11)
    ax.set_ylabel('Z Height (mm)', fontsize=11)
    ax.set_title('Z-Height Comparison', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5c: Distance traveled per move
    ax = axes[1, 0]
    orig_distances = np.zeros(len(orig_coords))
    orig_distances[1:] = np.linalg.norm(np.diff(orig_coords, axis=0), axis=1)
    stab_distances = np.zeros(len(stab_coords))
    stab_distances[1:] = np.linalg.norm(np.diff(stab_coords, axis=0), axis=1)
    
    ax.plot(np.cumsum(orig_distances), 'b-', alpha=0.7, linewidth=2, label='Original')
    ax.plot(np.cumsum(stab_distances), 'g-', alpha=0.7, linewidth=2, label='Stabilized')
    ax.set_xlabel('Move Number', fontsize=11)
    ax.set_ylabel('Cumulative Distance (mm)', fontsize=11)
    ax.set_title('Cumulative Distance Traveled', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5d: Move distance distribution
    ax = axes[1, 1]
    orig_dist_nonzero = orig_distances[orig_distances > 1e-6]
    stab_dist_nonzero = stab_distances[stab_distances > 1e-6]
    if len(orig_dist_nonzero) > 0:
        ax.hist(orig_dist_nonzero, bins=50, alpha=0.7, color='blue', edgecolor='black',
               label=f'Original ({len(orig_dist_nonzero)} moves)', density=True)
    if len(stab_dist_nonzero) > 0:
        ax.hist(stab_dist_nonzero, bins=50, alpha=0.7, color='green', edgecolor='black',
               label=f'Stabilized ({len(stab_dist_nonzero)} moves)', density=True)
    ax.set_xlabel('Move Distance (mm)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Move Distance Distribution', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(50, max(orig_dist_nonzero.max() if len(orig_dist_nonzero) > 0 else 0,
                               stab_dist_nonzero.max() if len(stab_dist_nonzero) > 0 else 0)))
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_geometry.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_geometry.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_geometry.png/pdf")
    
    # === Plot 6: Effectiveness Metrics Summary ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 6a: Move type comparison (bar chart)
    ax = axes[0, 0]
    move_types = ['Extrusion\n(XY)', 'Pure E\n(Micro-prime)', 'Retractions', 'Travel']
    orig_counts = [len(orig_extrusion_XY), len(orig_pure_E), len(orig_retractions), len(orig_travel)]
    stab_counts = [len(stab_extrusion_XY), len(stab_pure_E), len(stab_retractions), len(stab_travel)]
    
    x = np.arange(len(move_types))
    width = 0.35
    ax.bar(x - width/2, orig_counts, width, label='Original', color='blue', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, stab_counts, width, label='Stabilized', color='green', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Move Type', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Move Type Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(move_types)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6b: Extrusion continuity (no gaps)
    ax = axes[0, 1]
    # Count consecutive extrusion moves
    def count_consecutive_extrusions(flags):
        if len(flags) == 0:
            return []
        runs = []
        in_run = False
        run_length = 0
        for flag in flags:
            if flag:
                if not in_run:
                    in_run = True
                    run_length = 1
                else:
                    run_length += 1
            else:
                if in_run:
                    runs.append(run_length)
                    in_run = False
                    run_length = 0
        if in_run:
            runs.append(run_length)
        return runs
    
    orig_runs = count_consecutive_extrusions(orig_flags)
    stab_runs = count_consecutive_extrusions(stab_flags)
    
    if len(orig_runs) > 0:
        ax.hist(orig_runs, bins=min(30, max(orig_runs)+1), alpha=0.7, color='blue',
               edgecolor='black', label=f'Original (mean={np.mean(orig_runs):.1f})', density=True)
    if len(stab_runs) > 0:
        ax.hist(stab_runs, bins=min(30, max(stab_runs)+1), alpha=0.7, color='green',
               edgecolor='black', label=f'Stabilized (mean={np.mean(stab_runs):.1f})', density=True)
    ax.set_xlabel('Consecutive Extrusion Moves', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Extrusion Continuity (Longer = Better)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6c: Effectiveness summary metrics
    ax = axes[1, 0]
    ax.axis('off')
    
    # Calculate effectiveness metrics
    retraction_removal_rate = (len(orig_retractions) - len(stab_retractions)) / max(len(orig_retractions), 1) * 100
    geometry_preserved = abs(len(stab_coords) - len(orig_coords)) / max(len(orig_coords), 1) * 100
    total_distance_orig = np.sum(orig_distances)
    total_distance_stab = np.sum(stab_distances)
    distance_change = (total_distance_stab - total_distance_orig) / max(total_distance_orig, 1) * 100
    
    metrics_text = f"""
Stabilizer Effectiveness Metrics:

Retraction Suppression:
  Original retractions: {len(orig_retractions)}
  Stabilized retractions: {len(stab_retractions)}
  Removal rate: {retraction_removal_rate:.1f}%

Micro-prime Replacement:
  Micro-primes added: {len(stab_pure_E)}
  Avg micro-prime E: {stab_e[stab_pure_E].mean():.3f if len(stab_pure_E) > 0 else 0:.3f}

Geometry Preservation:
  Original moves: {len(orig_coords)}
  Stabilized moves: {len(stab_coords)}
  Move count change: {len(stab_coords) - len(orig_coords):+d} ({geometry_preserved:.1f}%)
  Total distance: {total_distance_orig:.1f} mm → {total_distance_stab:.1f} mm ({distance_change:+.1f}%)

Extrusion Continuity:
  Original avg consecutive: {np.mean(orig_runs):.1f if len(orig_runs) > 0 else 0:.1f}
  Stabilized avg consecutive: {np.mean(stab_runs):.1f if len(stab_runs) > 0 else 0:.1f}
  Improvement: {((np.mean(stab_runs) - np.mean(orig_runs)) / max(np.mean(orig_runs), 1) * 100):+.1f}%

Overall Assessment:
  ✓ Retractions eliminated: {'Yes' if len(stab_retractions) == 0 else 'Partial'}
  ✓ Geometry preserved: {'Yes' if abs(geometry_preserved) < 5 else 'Modified'}
  ✓ Extrusion continuity: {'Improved' if np.mean(stab_runs) > np.mean(orig_runs) else 'Similar'}
"""
    ax.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    # Plot 6d: Before/After comparison visualization
    ax = axes[1, 1]
    categories = ['Retractions', 'Micro-primes', 'Extrusion\nMoves', 'Travel\nMoves']
    orig_values = [len(orig_retractions), 0, len(orig_extrusion_XY), len(orig_travel)]
    stab_values = [len(stab_retractions), len(stab_pure_E), len(stab_extrusion_XY), len(stab_travel)]
    
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax.bar(x - width/2, orig_values, width, label='Original', color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, stab_values, width, label='Stabilized', color='green', alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Move Category', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Before/After Stabilization Comparison', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_effectiveness.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "comparison_effectiveness.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: comparison_effectiveness.png/pdf")
    
    print("\n=== Comparison complete ===")
    print(f"Generated {6} comparison plots:")
    print("  1. comparison_3d_toolpath.png - 3D toolpath visualization")
    print("  2. comparison_statistics.png - Extrusion statistics")
    print("  3. comparison_retractions.png - Retraction vs micro-prime analysis")
    print("  4. comparison_feedrates.png - Feed rate comparison")
    print("  5. comparison_geometry.png - Geometry preservation analysis")
    print("  6. comparison_effectiveness.png - Stabilizer effectiveness metrics")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 compare_gcode.py <original.gcode> <stabilized.gcode>")
        print("Example: python3 compare_gcode.py test.gcode results/stabilized.gcode")
        sys.exit(1)
    
    original_file = Path(sys.argv[1])
    stabilized_file = Path(sys.argv[2])
    
    if not original_file.exists():
        print(f"ERROR: Original file not found: {original_file}")
        sys.exit(1)
    if not stabilized_file.exists():
        print(f"ERROR: Stabilized file not found: {stabilized_file}")
        sys.exit(1)
    
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(original_file, stabilized_file, output_dir)

