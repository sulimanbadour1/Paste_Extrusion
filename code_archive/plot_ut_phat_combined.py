#!/usr/bin/env python3
"""
Combined 2×2 Figure: Extrusion Rate u(t) and Pressure Estimate p̂(t)
Shows the relationship: command excitation u(t) → pressure-window behavior p̂(t)

Top row: Extrusion rate u(t) - baseline (left) and stabilized (right)
Bottom row: Pressure estimate p̂(t) - baseline (left) and stabilized (right)

Usage:
    python3 plot_ut_phat_combined.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/ut_phat_combined.png
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib
# Use interactive backend if available, otherwise fall back to Agg
try:
    import matplotlib.pyplot as plt
    # Try to use interactive backend
    matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on what's available
except:
    matplotlib.use('Agg')  # Non-interactive backend fallback
    import matplotlib.pyplot as plt

# Professional color palette
COLORS = {
    'baseline': '#E63946',        # Vibrant red
    'stabilized': '#2A9D8F',      # Teal green
    'admissible': '#F1C40F',       # Gold
    'yield': '#E67E22',            # Dark orange
    'max': '#C0392B',              # Dark red
}

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


def compute_u_timeline(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute u(t) timeline from G-code.
    Returns: (time_array, u_array) where u is extrusion rate proxy.
    Includes both positive (extrusion) and negative (retraction) E moves.
    For moves with XY movement: u = ΔE / Δt where Δt = Δs / v
    For pure E moves: u = ΔE / Δt_est where Δt_est is estimated from feed rate
    """
    times = [0.0]
    u_values = [0.0]
    
    x_prev, y_prev, e_prev = 0.0, 0.0, 0.0
    e_cumulative = 0.0
    f_prev = 600.0  # Default feed rate
    t_curr = 0.0
    is_relative_e = True
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        if 'M83' in stripped.upper():
            is_relative_e = True
            continue
        elif 'M82' in stripped.upper():
            is_relative_e = False
            continue
        
        if stripped.startswith('G1') or stripped.startswith('G0'):
            parsed = parse_gcode_line(line)
            
            x_curr = parsed['X'] if parsed['X'] is not None else x_prev
            y_curr = parsed['Y'] if parsed['Y'] is not None else y_prev
            f_curr = parsed['F'] if parsed['F'] is not None else f_prev
            
            # Handle E value
            de = 0.0
            if parsed['E'] is not None:
                e_val = parsed['E']
                if is_relative_e:
                    de = e_val
                    e_cumulative += e_val
                else:
                    de = e_val - e_cumulative
                    e_cumulative = e_val
            
            # Compute Δs (XY distance)
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            ds = np.sqrt(dx**2 + dy**2)
            
            # Compute time increment
            dt = 0.0
            if ds > 1e-6:
                # Move with XY distance: use XY speed
                if f_curr > 0:
                    v = f_curr / 60.0  # mm/s
                    dt = ds / v  # seconds
            elif abs(de) > 1e-6:
                # Pure E move (no XY): estimate dt more accurately
                # For E-only moves, we need to estimate time based on extrusion rate
                # Typical extrusion rate: E per second depends on nozzle size and material
                # Conservative estimate: assume E moves take time proportional to E amount
                # Use feed rate as proxy for extrusion speed (mm/min -> mm/s)
                if f_curr > 0:
                    # For pure E moves, estimate time based on extrusion volume
                    # Assume extrusion happens at a rate related to feed rate
                    # More accurate: use a typical extrusion rate (e.g., 1-5 mm/s for paste)
                    # For micro-primes (small E), use a reasonable time estimate
                    if abs(de) < 1.0:  # Small E (micro-prime)
                        # Micro-primes are quick: estimate 0.1-0.5 seconds
                        dt = max(0.05, abs(de) / 2.0)  # Conservative: 2 mm/s extrusion rate
                    else:  # Larger E (priming, purge)
                        # Larger extrusions take more time
                        v_e = f_curr / 60.0  # mm/s (use feed as proxy)
                        dt = abs(de) / max(v_e, 1.0)
                else:
                    dt = 0.1  # Default small time step
            else:
                # No E and no XY movement: skip
                dt = 0.0
            
            # Include moves with any E change (positive or negative)
            # This captures both extrusions and retractions
            if abs(de) > 1e-6 and dt > 1e-6:
                t_curr += dt
                
                # Compute u(t) = ΔE / Δt (can be positive or negative)
                u = de / dt
                
                times.append(t_curr)
                u_values.append(u)
            
            # Update position tracking
            if x_curr != x_prev or y_curr != y_prev:
                x_prev, y_prev = x_curr, y_curr
            if f_curr > 0:
                f_prev = f_curr
    
    return np.array(times), np.array(u_values)


def compute_pressure_timeline(times: np.ndarray, u_values: np.ndarray,
                               alpha: float = 1.0, tau_r: float = 6.0,
                               p_y: float = 5.0, p_max: float = 14.0) -> np.ndarray:
    """
    Compute pressure estimate p̂(t) from u(t) timeline.
    Model: p_{k+1} = p_k + Δt_k * (α * u_k - p_k / τ_r)
    Uses sub-stepping for stability when dt is large.
    """
    p_hat = np.zeros_like(times)
    p_hat[0] = 0.0  # Initial pressure
    
    # Maximum time step for integration (for stability)
    max_dt = 0.1  # seconds
    
    for i in range(1, len(times)):
        dt_total = times[i] - times[i-1]
        u_k = u_values[i]
        p_k = p_hat[i-1]
        
        # Use sub-stepping if dt is too large (for numerical stability)
        if dt_total > max_dt:
            n_steps = max(1, int(np.ceil(dt_total / max_dt)))
            dt_step = dt_total / n_steps
            p_current = p_k
            for _ in range(n_steps):
                # Discrete pressure model with sub-stepping
                p_current = p_current + dt_step * (alpha * u_k - p_current / tau_r)
                # Ensure non-negative pressure
                p_current = max(0.0, p_current)
            p_hat[i] = p_current
        else:
            # Single step integration
            p_hat[i] = p_k + dt_total * (alpha * u_k - p_k / tau_r)
            # Ensure non-negative pressure
            p_hat[i] = max(0.0, p_hat[i])
    
    return p_hat


# ============================================================================
# Combined 2×2 Figure
# ============================================================================

def create_combined_figure(baseline_lines: List[str], stabilized_lines: List[str],
                          output_path: Path, dpi: int = 300,
                          alpha: float = 1.0, tau_r: float = 6.0,
                          p_y: float = 5.0, p_max: float = 14.0):
    """
    Create a 2×2 figure showing u(t) and p̂(t) for baseline and stabilized.
    """
    print("Computing u(t) timelines...", flush=True)
    
    # Compute u(t) for both
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    print("Computing p̂(t) timelines...", flush=True)
    
    # Compute p̂(t) for both
    p_baseline = compute_pressure_timeline(times_baseline, u_baseline, alpha, tau_r, p_y, p_max)
    p_stabilized = compute_pressure_timeline(times_stabilized, u_stabilized, alpha, tau_r, p_y, p_max)
    
    # Debug: Verify computed values
    print(f"Baseline p̂(t): range [{np.min(p_baseline):.2f}, {np.max(p_baseline):.2f}] kPa, mean {np.mean(p_baseline):.2f} kPa", flush=True)
    print(f"Stabilized p̂(t): range [{np.min(p_stabilized):.2f}, {np.max(p_stabilized):.2f}] kPa, mean {np.mean(p_stabilized):.2f} kPa", flush=True)
    print(f"Stabilized p̂(t) non-zero: {np.sum(p_stabilized > 1e-6)}/{len(p_stabilized)} points", flush=True)
    
    # Set up figure with professional IEEE paper styling - ALL FONTS BOLD
    plt.rcParams.update({
        'font.size': 11,
        'font.weight': 'bold',  # Make all fonts bold by default
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'lines.linewidth': 2.0,
    })
    
    # Create 1×2 subplot figure (side by side) with both baseline and stabilized overlaid
    # Make figure wider to accommodate legends outside
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    fig.suptitle('Command Excitation u(t) → Pressure-Window Behavior p̂(t)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Determine time range for consistent x-axis
    # Use the MINIMUM of both so both plots show the same time window for direct comparison
    t_max_baseline = np.max(times_baseline) if len(times_baseline) > 0 else 0
    t_max_stabilized = np.max(times_stabilized) if len(times_stabilized) > 0 else 0
    t_max = min(t_max_baseline, t_max_stabilized)  # Use min so both end at same time
    if t_max == 0:
        t_max = max(t_max_baseline, t_max_stabilized, 100)  # Fallback if one is empty
    
    print(f"Time ranges: Baseline [0, {t_max_baseline:.1f}]s, Stabilized [0, {t_max_stabilized:.1f}]s", flush=True)
    print(f"Using synchronized x-axis: [0, {t_max:.1f}]s for direct comparison (same time window)", flush=True)
    
    # Determine y-axis ranges
    u_min = 0
    u_max = 0
    if len(u_baseline) > 0:
        u_min = min(u_min, np.min(u_baseline))
        u_max = max(u_max, np.max(u_baseline))
    if len(u_stabilized) > 0:
        u_min = min(u_min, np.min(u_stabilized))
        u_max = max(u_max, np.max(u_stabilized))
    u_range = u_max - u_min
    if u_range < 1e-6:
        u_max = 10.0
        u_min = -5.0
    else:
        u_max += u_range * 0.1
        u_min -= u_range * 0.1
    
    p_min = 0
    p_max_val = 0
    if len(p_baseline) > 0:
        p_min = min(p_min, np.min(p_baseline))
        p_max_val = max(p_max_val, np.max(p_baseline))
    if len(p_stabilized) > 0:
        p_min = min(p_min, np.min(p_stabilized))
        p_max_val = max(p_max_val, np.max(p_stabilized))
    p_max_val = max(p_max_val, p_max * 1.1)  # Ensure window is visible
    
    # ========================================================================
    # Left: u(t) Comparison (Baseline + Stabilized Overlaid)
    # ========================================================================
    ax1 = axes[0]
    
    # Plot baseline u(t)
    if len(times_baseline) > 1 and len(u_baseline) > 1:
        # Downsample if too many points for performance
        if len(times_baseline) > 10000:
            step_b = len(times_baseline) // 10000
            ax1.plot(times_baseline[::step_b], u_baseline[::step_b], 
                   color=COLORS['baseline'], linewidth=2.5, alpha=0.9, 
                   label='Baseline', zorder=3)
        else:
            ax1.plot(times_baseline, u_baseline, 
                   color=COLORS['baseline'], linewidth=2.5, alpha=0.9, 
                   label='Baseline', zorder=3)
    
    # Plot stabilized u(t)
    if len(times_stabilized) > 1 and len(u_stabilized) > 1:
        # Downsample if too many points for performance
        if len(times_stabilized) > 10000:
            step_s = len(times_stabilized) // 10000
            ax1.plot(times_stabilized[::step_s], u_stabilized[::step_s], 
                   color=COLORS['stabilized'], linewidth=2.5, alpha=0.9, 
                   label='Stabilized', zorder=4)
        else:
            ax1.plot(times_stabilized, u_stabilized, 
                   color=COLORS['stabilized'], linewidth=2.5, alpha=0.9, 
                   label='Stabilized', zorder=4)
    
    # Add zero line for reference
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, zorder=1)
    
    ax1.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Extrusion Rate u(t) (mm/s)', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Extrusion Rate u(t)', fontsize=14, fontweight='bold', pad=12)
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=1)
    ax1.set_axisbelow(True)
    ax1.set_xlim([0, t_max])
    ax1.set_ylim([u_min, u_max])
    
    # Make tick labels bold
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # Enhanced legend - placed outside the figure
    legend1 = ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, 
                         fancybox=False, shadow=False, framealpha=0.98, edgecolor='black', 
                         fontsize=12, handlelength=2.5, handletextpad=0.5)
    # Make legend text bold
    for text in legend1.get_texts():
        text.set_fontweight('bold')
    
    # ========================================================================
    # Right: p̂(t) Comparison (Baseline + Stabilized Overlaid)
    # ========================================================================
    ax2 = axes[1]
    
    # Plot admissible window FIRST (background)
    ax2.fill_between([0, t_max], p_y, p_max, alpha=0.25, color=COLORS['admissible'], 
                   label='Admissible Window', zorder=1)
    ax2.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.5, 
               label=f'p_y = {p_y:.1f} kPa', zorder=2)
    ax2.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.5, 
               label=f'p_max = {p_max:.1f} kPa', zorder=2)
    
    # Plot baseline p̂(t)
    if len(times_baseline) > 0 and len(p_baseline) > 0:
        if len(times_baseline) > 10000:
            step_b = len(times_baseline) // 10000
            ax2.plot(times_baseline[::step_b], p_baseline[::step_b], 
                   color=COLORS['baseline'], linewidth=2.5, alpha=0.9, 
                   label='Baseline', zorder=4)
        else:
            ax2.plot(times_baseline, p_baseline, 
                   color=COLORS['baseline'], linewidth=2.5, alpha=0.9, 
                   label='Baseline', zorder=4)
    
    # Plot stabilized p̂(t)
    if len(times_stabilized) > 0 and len(p_stabilized) > 0:
        # Verify we have valid data
        valid_mask = (times_stabilized <= t_max) & np.isfinite(p_stabilized) & (p_stabilized >= 0)
        if np.sum(valid_mask) > 0:
            times_s_plot = times_stabilized[valid_mask]
            p_s_plot = p_stabilized[valid_mask]
            
            if len(times_s_plot) > 10000:
                step_s = len(times_s_plot) // 10000
                ax2.plot(times_s_plot[::step_s], p_s_plot[::step_s], 
                       color=COLORS['stabilized'], linewidth=2.5, alpha=0.9, 
                       label='Stabilized', zorder=5)
            else:
                ax2.plot(times_s_plot, p_s_plot, 
                       color=COLORS['stabilized'], linewidth=2.5, alpha=0.9, 
                       label='Stabilized', zorder=5)
            
            print(f"Plotted stabilized: {len(times_s_plot)} points, p̂ range [{np.min(p_s_plot):.2f}, {np.max(p_s_plot):.2f}] kPa", flush=True)
        else:
            print(f"WARNING: No valid stabilized data to plot within time window [0, {t_max:.2f}]s", flush=True)
    
    ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Estimated Pressure p̂(t) (kPa)', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Estimated Pressure p̂(t)', fontsize=14, fontweight='bold', pad=12)
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=1)
    ax2.set_axisbelow(True)
    ax2.set_xlim([0, t_max])
    ax2.set_ylim([0, p_max_val])
    
    # Make tick labels bold
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    # Enhanced legend - placed outside the figure
    legend2 = ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=True, 
                         fancybox=False, shadow=False, framealpha=0.98, edgecolor='black', 
                         fontsize=11, handlelength=2.5, handletextpad=0.5, ncol=1)
    # Make legend text bold
    for text in legend2.get_texts():
        text.set_fontweight('bold')
    
    # ========================================================================
    # Add explanatory note at bottom
    # ========================================================================
    time_note = f'Note: Both plots show synchronized time window [0, {t_max:.0f}s]. '
    time_note += f'Baseline total duration: {t_max_baseline:.0f}s, Stabilized: {t_max_stabilized:.0f}s. '
    time_note += 'Stabilized is faster due to retraction elimination (dwells + micro-primes vs travel moves).'
    
    fig.text(0.5, 0.02, 
             'Stabilization shapes u(t) to keep p̂(t) within admissible window [p_y, p_max]. ' + time_note,
             ha='center', fontsize=10, fontweight='bold', style='italic', color='#555555',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f8f8', alpha=0.9, 
                      edgecolor='#cccccc', linewidth=1.2))
    
    # ========================================================================
    # Display Figure (no auto-save)
    # ========================================================================
    # Adjust layout to accommodate legends outside
    plt.tight_layout(rect=[0, 0.04, 0.92, 0.96])  # Leave space for suptitle, note, and legends
    plt.show()  # Display figure on screen
    print(f"[OK] Figure displayed. Save manually if needed.", flush=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create combined 2×2 figure: u(t) and p̂(t) for baseline and stabilized",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Save to file
  python3 plot_ut_phat_combined.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/ut_phat_combined.png
  
  # Custom pressure parameters
  python3 plot_ut_phat_combined.py --baseline test.gcode --stabilized results/stabilized.gcode --output results/ut_phat_combined.png --p-y 5.0 --p-max 14.0
        """
    )
    
    parser.add_argument('--baseline', '--baseline-gcode', dest='baseline_gcode', required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized', '--stabilized-gcode', dest='stabilized_gcode', required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--output', '-o', dest='output', default='results/ut_phat_combined.png',
                       help='Output path for saved figure (default: results/ut_phat_combined.png)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure (default: 300)')
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Pressure model parameter alpha (default: 1.0, matches stabilizer)')
    parser.add_argument('--tau-r', type=float, default=6.0,
                       help='Pressure model parameter tau_r (default: 6.0)')
    parser.add_argument('--p-y', type=float, default=5.0,
                       help='Yield pressure p_y (default: 5.0)')
    parser.add_argument('--p-max', type=float, default=14.0,
                       help='Maximum pressure p_max (default: 14.0)')
    
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
    output_path = Path(args.output)
    create_combined_figure(baseline_lines, stabilized_lines, output_path, args.dpi,
                          args.alpha, args.tau_r, args.p_y, args.p_max)
    
    print("[OK] Done", flush=True)
    plt.show()


if __name__ == '__main__':
    main()

