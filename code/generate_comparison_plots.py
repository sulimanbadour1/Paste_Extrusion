#!/usr/bin/env python3
"""
Generate Three Key Comparison Plots for Paper
1. Extrusion Rate Comparison (baseline + stabilized in one plot)
2. Pressure Estimate Comparison (baseline + stabilized + window lines)
3. Electrical Validation Combined (bar for open-circuit + boxplot for resistance)
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Professional color palette
COLORS = {
    'baseline': '#E63946',        # Vibrant red
    'stabilized': '#2A9D8F',      # Teal green
    'admissible': '#F1C40F',      # Gold
    'yield': '#E67E22',           # Dark orange
    'max': '#C0392B',             # Dark red
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
    Filter: only moves with ΔE > 0 and Δs > 0.
    """
    times = [0.0]
    u_values = [0.0]
    
    x_prev, y_prev, e_prev = 0.0, 0.0, 0.0
    e_cumulative = 0.0
    f_prev = 0.0
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
            
            # Filter: only moves with ΔE > 0 and Δs > 0
            if de > 1e-6 and ds > 1e-6:
                # Compute Δt = Δs / v, where v = F / 60 (mm/s)
                if f_curr > 0:
                    v = f_curr / 60.0  # mm/s
                    dt = ds / v  # seconds
                    t_curr += dt
                    
                    # Compute u(t) = ΔE / Δt
                    u = de / dt if dt > 0 else 0.0
                    
                    times.append(t_curr)
                    u_values.append(u)
            
            x_prev, y_prev = x_curr, y_curr
            if f_curr > 0:
                f_prev = f_curr
    
    return np.array(times), np.array(u_values)


def compute_pressure_timeline(times: np.ndarray, u_values: np.ndarray,
                               alpha: float = 8.0, tau_r: float = 6.0,
                               p_y: float = 5.0, p_max: float = 14.0) -> np.ndarray:
    """
    Compute pressure estimate p̂(t) from u(t) timeline.
    Model: p_{k+1} = p_k + Δt_k * (α * u_k - p_k / τ_r)
    """
    p_hat = np.zeros_like(times)
    p_hat[0] = 0.0  # Initial pressure
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        u_k = u_values[i]
        p_k = p_hat[i-1]
        
        # Discrete pressure model
        p_hat[i] = p_k + dt * (alpha * u_k - p_k / tau_r)
        
        # Ensure non-negative pressure
        p_hat[i] = max(0.0, p_hat[i])
    
    return p_hat


# ============================================================================
# Plot 1: Extrusion Rate Comparison
# ============================================================================

def plot_extrusion_rate_comparison(baseline_lines: List[str], stabilized_lines: List[str],
                                    output_path: Path, dpi: int = 300):
    """
    Plot extrusion rate comparison: baseline + stabilized in one plot.
    """
    print("Computing extrusion rate timelines...", flush=True)
    
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Set up figure with professional IEEE paper styling
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
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
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot both timelines with professional styling
    if len(times_baseline) > 0 and len(u_baseline) > 0:
        # Downsample if too many points for performance
        if len(times_baseline) > 10000:
            step = len(times_baseline) // 10000
            ax.plot(times_baseline[::step], u_baseline[::step], 
                   color=COLORS['baseline'], linewidth=2.0, alpha=0.85, label='Baseline', zorder=2)
        else:
            ax.plot(times_baseline, u_baseline, 
                   color=COLORS['baseline'], linewidth=2.0, alpha=0.85, label='Baseline', zorder=2)
    
    if len(times_stabilized) > 0 and len(u_stabilized) > 0:
        # Downsample if too many points for performance
        if len(times_stabilized) > 10000:
            step = len(times_stabilized) // 10000
            ax.plot(times_stabilized[::step], u_stabilized[::step], 
                   color=COLORS['stabilized'], linewidth=2.0, alpha=0.85, label='Stabilized', zorder=2)
        else:
            ax.plot(times_stabilized, u_stabilized, 
                   color=COLORS['stabilized'], linewidth=2.0, alpha=0.85, label='Stabilized', zorder=2)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Extrusion Rate u(t) (mm/s)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
              framealpha=0.95, edgecolor='black', fontsize=11)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Saved: {output_path}", flush=True)


# ============================================================================
# Plot 2: Pressure Estimate Comparison
# ============================================================================

def plot_pressure_estimate_comparison(baseline_lines: List[str], stabilized_lines: List[str],
                                       output_path: Path, dpi: int = 300,
                                       alpha: float = 8.0, tau_r: float = 6.0,
                                       p_y: float = 5.0, p_max: float = 14.0):
    """
    Plot pressure estimate comparison: baseline + stabilized + window lines.
    """
    print("Computing pressure estimates...", flush=True)
    
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    p_baseline = compute_pressure_timeline(times_baseline, u_baseline, alpha, tau_r, p_y, p_max)
    p_stabilized = compute_pressure_timeline(times_stabilized, u_stabilized, alpha, tau_r, p_y, p_max)
    
    # Set up figure with professional IEEE paper styling
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
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
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Determine time range for window
    t_max = 0
    if len(times_baseline) > 0:
        t_max = max(t_max, np.max(times_baseline))
    if len(times_stabilized) > 0:
        t_max = max(t_max, np.max(times_stabilized))
    if t_max == 0:
        t_max = 100  # Default
    
    # Plot admissible window FIRST (background)
    ax.fill_between([0, t_max], p_y, p_max, alpha=0.2, color=COLORS['admissible'], 
                   label='Admissible Window', zorder=1)
    ax.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.0, 
              label=f'p_y = {p_y:.1f}', zorder=3)
    ax.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.0, 
              label=f'p_max = {p_max:.1f}', zorder=3)
    
    # Plot baseline pressure
    if len(times_baseline) > 0 and len(p_baseline) > 0:
        if len(times_baseline) > 10000:
            step = len(times_baseline) // 10000
            ax.plot(times_baseline[::step], p_baseline[::step], 
                   color=COLORS['baseline'], linewidth=2.0, alpha=0.85, label='Baseline', zorder=4)
        else:
            ax.plot(times_baseline, p_baseline, 
                   color=COLORS['baseline'], linewidth=2.0, alpha=0.85, label='Baseline', zorder=4)
    
    # Plot stabilized pressure
    if len(times_stabilized) > 0 and len(p_stabilized) > 0:
        if len(times_stabilized) > 10000:
            step = len(times_stabilized) // 10000
            ax.plot(times_stabilized[::step], p_stabilized[::step], 
                   color=COLORS['stabilized'], linewidth=2.0, alpha=0.85, label='Stabilized', zorder=4)
        else:
            ax.plot(times_stabilized, p_stabilized, 
                   color=COLORS['stabilized'], linewidth=2.0, alpha=0.85, label='Stabilized', zorder=4)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Estimated Pressure p̂(t) (kPa)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False, 
              framealpha=0.95, edgecolor='black', ncol=1, fontsize=10)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Saved: {output_path}", flush=True)


# ============================================================================
# Plot 3: Electrical Validation Combined
# ============================================================================

def plot_electrical_validation_combined(electrical_df: pd.DataFrame, output_path: Path, dpi: int = 300):
    """
    Plot electrical validation: bar for open-circuit + boxplot for resistance, same image.
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("ERROR: No electrical data available", flush=True)
        return
    
    if 'condition' not in electrical_df.columns:
        print("ERROR: electrical_traces.csv must contain 'condition' column", flush=True)
        return
    
    # Set up figure with professional IEEE paper styling
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 0.95,
        'legend.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
    })
    
    fig = plt.figure(figsize=(7.0, 3.5))
    
    # Map condition names (handle both 'full' and 'stabilized')
    condition_map = {'baseline': 'Baseline', 'full': 'Stabilized', 'stabilized': 'Stabilized'}
    electrical_df['condition_display'] = electrical_df['condition'].map(condition_map).fillna(electrical_df['condition'])
    
    conditions = ['Baseline', 'Stabilized']
    colors = {'Baseline': COLORS['baseline'], 'Stabilized': COLORS['stabilized']}
    
    # Left subplot: Open-circuit rate (bar chart)
    ax1 = fig.add_subplot(121)
    
    open_circuit_rates = []
    n_trials = []
    
    for cond in conditions:
        cond_data = electrical_df[electrical_df['condition_display'] == cond]
        if len(cond_data) > 0:
            if 'open_circuit' in cond_data.columns:
                open_count = cond_data['open_circuit'].sum()
                total = len(cond_data)
                rate = (open_count / total) * 100 if total > 0 else 0
                open_circuit_rates.append(rate)
                n_trials.append(total)
            else:
                open_circuit_rates.append(0)
                n_trials.append(0)
        else:
            open_circuit_rates.append(0)
            n_trials.append(0)
    
    x = np.arange(len(conditions))
    bars = ax1.bar(x, open_circuit_rates, 
                   color=[colors.get(c, 'gray') for c in conditions],
                   alpha=0.9, edgecolor='black', linewidth=1.2, width=0.6)
    
    # Add value labels
    for bar, rate, n in zip(bars, open_circuit_rates, n_trials):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Open-Circuit Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, fontsize=11)
    ax1.set_ylim([0, max(open_circuit_rates) + 15 if len(open_circuit_rates) > 0 else 100])
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Right subplot: Resistance boxplot
    ax2 = fig.add_subplot(122)
    
    # Filter to non-open circuits only
    if 'open_circuit' in electrical_df.columns and 'resistance_ohm' in electrical_df.columns:
        df_ok = electrical_df[electrical_df['open_circuit'] == 0].copy()
        df_ok['resistance_ohm'] = pd.to_numeric(df_ok['resistance_ohm'], errors='coerce')
        df_ok = df_ok.dropna(subset=['resistance_ohm'])
        
        if len(df_ok) > 0:
            data = []
            labels = []
            for cond in conditions:
                cond_data = df_ok[df_ok['condition_display'] == cond]
                if len(cond_data) > 0:
                    data.append(cond_data['resistance_ohm'].values)
                    labels.append(cond)
            
            if len(data) > 0:
                bp = ax2.boxplot(data, tick_labels=labels, patch_artist=True, 
                                widths=0.6, showmeans=True, meanline=False)
                
                # Color the boxes professionally
                for patch, cond in zip(bp['boxes'], conditions):
                    patch.set_facecolor(colors.get(cond, 'gray'))
                    patch.set_alpha(0.8)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.2)
                
                # Style other elements
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    if element in bp:
                        plt.setp(bp[element], color='black', linewidth=1.2)
                # Means in different style
                if 'means' in bp:
                    plt.setp(bp['means'], marker='D', markerfacecolor='white', 
                            markeredgecolor='black', markersize=6, markeredgewidth=1.2)
                
                ax2.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Condition', fontsize=12, fontweight='bold')
                ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
                ax2.set_axisbelow(True)
            else:
                ax2.text(0.5, 0.5, 'No resistance data\n(non-open circuits)', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=11)
                ax2.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Condition', fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No non-open circuits', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=11)
            ax2.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Condition', fontsize=12, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'Missing columns:\nopen_circuit or resistance_ohm', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=11)
        ax2.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Condition', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"[OK] Saved: {output_path}", flush=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate three key comparison plots for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all three plots
  python3 generate_comparison_plots.py --baseline test.gcode --stabilized results/stabilized.gcode --data-dir code/data --output-dir results/figures
  
  # Generate with custom pressure parameters
  python3 generate_comparison_plots.py --baseline test.gcode --stabilized results/stabilized.gcode --data-dir code/data --output-dir results/figures --p-y 5.0 --p-max 14.0
        """
    )
    
    parser.add_argument('--baseline', '--baseline-gcode', dest='baseline_gcode', required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized', '--stabilized-gcode', dest='stabilized_gcode', required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--data-dir', dest='data_dir', default=None,
                       help='Directory containing CSV data files (default: code/data)')
    parser.add_argument('--output-dir', dest='output_dir', default='results/figures',
                       help='Output directory for figures (default: results/figures)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figures (default: 300)')
    parser.add_argument('--alpha', type=float, default=8.0,
                       help='Pressure model parameter alpha (default: 8.0)')
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
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / 'data'
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read G-code files
    print(f"Reading baseline G-code: {baseline_path}", flush=True)
    with open(baseline_path, 'r', encoding='utf-8', errors='replace') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_path}", flush=True)
    with open(stabilized_path, 'r', encoding='utf-8', errors='replace') as f:
        stabilized_lines = f.readlines()
    
    print(f"Loaded {len(baseline_lines)} baseline lines, {len(stabilized_lines)} stabilized lines", flush=True)
    
    # Read electrical data
    electrical_path = data_dir / 'electrical_traces.csv'
    electrical_df = None
    if electrical_path.exists():
        print(f"Reading electrical data: {electrical_path}", flush=True)
        electrical_df = pd.read_csv(electrical_path)
        print(f"Loaded {len(electrical_df)} electrical trace records", flush=True)
    else:
        print(f"WARNING: {electrical_path} not found, will skip electrical validation plot", flush=True)
    
    # Generate Plot 1: Extrusion Rate Comparison
    print("\n" + "="*70, flush=True)
    print("Generating Plot 1: Extrusion Rate Comparison", flush=True)
    print("="*70, flush=True)
    plot_extrusion_rate_comparison(
        baseline_lines, stabilized_lines,
        output_dir / 'extrusion_rate_comparison.png',
        args.dpi
    )
    
    # Generate Plot 2: Pressure Estimate Comparison
    print("\n" + "="*70, flush=True)
    print("Generating Plot 2: Pressure Estimate Comparison", flush=True)
    print("="*70, flush=True)
    plot_pressure_estimate_comparison(
        baseline_lines, stabilized_lines,
        output_dir / 'pressure_estimate_comparison.png',
        args.dpi,
        args.alpha, args.tau_r, args.p_y, args.p_max
    )
    
    # Generate Plot 3: Electrical Validation Combined
    if electrical_df is not None:
        print("\n" + "="*70, flush=True)
        print("Generating Plot 3: Electrical Validation Combined", flush=True)
        print("="*70, flush=True)
        plot_electrical_validation_combined(
            electrical_df,
            output_dir / 'electrical_validation_combined.png',
            args.dpi
        )
    else:
        print("\nSkipping Plot 3: Electrical Validation (no data)", flush=True)
    
    print("\n" + "="*70, flush=True)
    print("[OK] All plots generated successfully!", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print("="*70, flush=True)


if __name__ == '__main__':
    main()

