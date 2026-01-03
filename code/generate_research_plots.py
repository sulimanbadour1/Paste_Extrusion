#!/usr/bin/env python3
"""
Generate publication-ready figures for IEEE paper on paste extrusion stabilization.
All figures are formatted for single-column IEEE format and match LaTeX references exactly.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# IEEE single-column width: 3.5 inches (88.9mm)
IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.0

# Set style for publication-quality figures
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    try:
        plt.style.use('seaborn-paper')
    except:
        plt.style.use('default')

# Publication-quality settings
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 8
matplotlib.rcParams['figure.titlesize'] = 11
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.2

# Directories
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
CODE_DIR = SCRIPT_DIR
RESULTS_DIR = CODE_DIR / "results"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

def save_fig(name: str, fig=None, dpi=300):
    """Save figure in PDF (LaTeX) and PNG (preview) formats."""
    if fig is None:
        fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", dpi=dpi, format='pdf')
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=dpi, format='png')
    print(f"  ✓ Saved {name}.pdf and {name}.png")

# ============================================================================
# 1. EXTRUSION ONSET AND FLOW DURATION
# ============================================================================

def plot_extrusion_onset_boxplot(print_df: pd.DataFrame):
    """Figure: Extrusion onset time distribution (boxplot with mean marker)."""
    order = ["baseline", "partial", "full"]
    colors = {"baseline": "#d62728", "partial": "#ff7f0e", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    data = [print_df.loc[print_df["condition"] == c, "onset_s"].values for c in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True,
                    meanline=True, showfliers=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], [colors[c] for c in order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=0.8)
    
    ax.set_ylabel("Extrusion Onset Time (s)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    save_fig("extrusion_onset_boxplot", fig)

def plot_flow_duration_boxplot(print_df: pd.DataFrame):
    """Figure: Continuous flow duration distribution (boxplot with mean marker)."""
    order = ["baseline", "partial", "full"]
    colors = {"baseline": "#d62728", "partial": "#ff7f0e", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    data = [print_df.loc[print_df["condition"] == c, "flow_duration_s"].values for c in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, showmeans=True,
                    meanline=True, showfliers=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], [colors[c] for c in order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(0.8)
    
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=0.8)
    
    ax.set_ylabel("Continuous Flow Duration (s)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    save_fig("flow_duration_boxplot", fig)

# ============================================================================
# 2. SUCCESS RATES
# ============================================================================

def plot_first_layer_success_rate(print_df: pd.DataFrame):
    """Figure: First-layer success rate by configuration."""
    order = ["baseline", "partial", "full"]
    colors = {"baseline": "#d62728", "partial": "#ff7f0e", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    g = print_df.groupby("condition")["first_layer_success"].mean().reindex(order)
    bars = ax.bar(g.index, g.values * 100, 
                   color=[colors[c] for c in g.index], alpha=0.8, 
                   edgecolor='black', linewidth=0.8, width=0.6)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig("first_layer_success_rate", fig)

def plot_completion_rate(print_df: pd.DataFrame):
    """Figure: Print completion rate by configuration."""
    order = ["baseline", "partial", "full"]
    colors = {"baseline": "#d62728", "partial": "#ff7f0e", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    g = print_df.groupby("condition")["completed"].mean().reindex(order)
    bars = ax.bar(g.index, g.values * 100,
                   color=[colors[c] for c in g.index], alpha=0.8, 
                   edgecolor='black', linewidth=0.8, width=0.6)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Completion Rate (%)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig("completion_rate", fig)

# ============================================================================
# 3. CLOG FREQUENCY
# ============================================================================

def plot_clogs_per_print(print_df: pd.DataFrame):
    """Figure: Clogs per print (mean ± SD) across configurations."""
    order = ["baseline", "partial", "full"]
    colors = {"baseline": "#d62728", "partial": "#ff7f0e", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    means = print_df.groupby("condition")["clogs"].mean().reindex(order)
    stds = print_df.groupby("condition")["clogs"].std(ddof=1).reindex(order)
    
    bars = ax.bar(means.index, means.values, yerr=stds.values, capsize=5,
                  color=[colors[c] for c in means.index], alpha=0.8, 
                  edgecolor='black', linewidth=0.8, width=0.6,
                  error_kw={'linewidth': 1.0, 'capthick': 1.0})
    
    ax.set_ylabel("Clogs per Print (mean ± SD)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means.values, stds.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.05,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig("clogs_per_print", fig)

# ============================================================================
# 4. PRESSURE SIMULATION
# ============================================================================

def plot_pressure_simulation_baseline():
    """Figure: Simulated pressure trajectory under baseline stop/start extrusion."""
    dt = 0.05
    t = np.arange(0, 60 + dt, dt)
    
    alpha = 1.0
    tau_r = 6.0
    p_y = 5.0
    p_max = 14.0
    
    # Baseline u(t): frequent start/stop
    u_base = np.zeros_like(t)
    for k in range(len(t)):
        phase = t[k] % 3.0
        u_base[k] = 1.8 if phase < 2.0 else 0.0
    
    def integrate(u):
        p = np.zeros_like(t)
        for k in range(1, len(t)):
            p_dot = alpha * u[k-1] - (1.0 / tau_r) * p[k-1]
            p[k] = p[k-1] + dt * p_dot
        return p
    
    p_base = integrate(u_base)
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    ax.plot(t, p_base, 'b-', linewidth=1.5, label="$p(t)$")
    ax.axhline(p_y, linestyle="--", color="orange", linewidth=1.5, label=f"$p_y = {p_y:.0f}$")
    ax.axhline(p_max, linestyle="--", color="red", linewidth=1.5, label=f"$p_{{\\max}} = {p_max:.0f}$")
    ax.fill_between(t, p_y, p_max, alpha=0.15, color='green', label='Admissible window')
    
    ax.set_xlabel("Time (s)", fontweight='bold')
    ax.set_ylabel("Pressure (arb. units)", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    save_fig("pressure_simulation_baseline", fig)

def plot_pressure_simulation_stabilized():
    """Figure: Simulated pressure trajectory under stabilized execution."""
    dt = 0.05
    t = np.arange(0, 60 + dt, dt)
    
    alpha = 1.0
    tau_r = 6.0
    p_y = 5.0
    p_max = 14.0
    
    # Stabilized u(t): ramp + fewer discontinuities
    u_stab = np.zeros_like(t)
    for k in range(len(t)):
        if t[k] < 5:
            u_stab[k] = 0.35 * t[k]  # ramp
        else:
            u_stab[k] = 1.4 + 0.2 * np.sin(0.2 * t[k])
    
    def integrate(u):
        p = np.zeros_like(t)
        for k in range(1, len(t)):
            p_dot = alpha * u[k-1] - (1.0 / tau_r) * p[k-1]
            p[k] = p[k-1] + dt * p_dot
        return p
    
    p_stab = integrate(u_stab)
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    ax.plot(t, p_stab, 'g-', linewidth=1.5, label="$p(t)$")
    ax.axhline(p_y, linestyle="--", color="orange", linewidth=1.5, label=f"$p_y = {p_y:.0f}$")
    ax.axhline(p_max, linestyle="--", color="red", linewidth=1.5, label=f"$p_{{\\max}} = {p_max:.0f}$")
    ax.fill_between(t, p_y, p_max, alpha=0.15, color='green', label='Admissible window')
    
    ax.set_xlabel("Time (s)", fontweight='bold')
    ax.set_ylabel("Pressure (arb. units)", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    save_fig("pressure_simulation_stabilized", fig)

# ============================================================================
# 5. ESTIMATED PRESSURE TRACE
# ============================================================================

def plot_phat_trace():
    """Figure: Estimated pressure state p_hat(t) and window bounds."""
    csv_path = RESULTS_DIR / "run_log.csv"
    if not csv_path.exists():
        print(f"  ⚠ Warning: {csv_path} not found. Skipping p_hat trace.")
        return
    
    df = pd.read_csv(csv_path)
    df["p_hat"] = pd.to_numeric(df["p_hat"], errors="coerce")
    df = df.dropna(subset=["p_hat"])
    
    if len(df) == 0:
        print(f"  ⚠ Warning: No valid p_hat data in {csv_path}")
        return
    
    t = df["t_s"].values
    p = df["p_hat"].values
    
    p_y = 5.0
    p_max = 14.0
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    ax.plot(t, p, 'b-', linewidth=1.0, alpha=0.8, label="$\\hat{p}(t)$")
    ax.axhline(p_y, linestyle="--", color="orange", linewidth=1.5, label=f"$p_y = {p_y:.0f}$")
    ax.axhline(p_max, linestyle="--", color="red", linewidth=1.5, label=f"$p_{{\\max}} = {p_max:.0f}$")
    ax.fill_between(t, p_y, p_max, alpha=0.15, color='green', label='Admissible window')
    
    ax.set_xlabel("Time (s)", fontweight='bold')
    ax.set_ylabel("Estimated Pressure (arb. units)", fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=False)
    
    save_fig("phat_trace", fig)

# ============================================================================
# 6. ELECTRICAL TRACE RESULTS
# ============================================================================

def plot_open_circuit_rate(elec_df: pd.DataFrame):
    """Figure: Open-circuit rate for printed conductive traces."""
    order = ["baseline", "full"]
    colors = {"baseline": "#d62728", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    g = elec_df.groupby("condition")["open_circuit"].mean().reindex(order)
    bars = ax.bar(g.index, g.values * 100,
                  color=[colors[c] for c in g.index], alpha=0.8, 
                  edgecolor='black', linewidth=0.8, width=0.6)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel("Open-Circuit Rate (%)", fontweight='bold')
    ax.set_xlabel("Configuration", fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    save_fig("open_circuit_rate", fig)

def plot_resistance_boxplot(elec_df: pd.DataFrame):
    """Figure: Resistance distribution of non-open conductive traces."""
    order = ["baseline", "full"]
    colors = {"baseline": "#d62728", "full": "#2ca02c"}
    
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75))
    
    df_ok = elec_df[elec_df["open_circuit"] == 0].copy()
    df_ok["resistance_ohm"] = pd.to_numeric(df_ok["resistance_ohm"], errors="coerce")
    
    data = [df_ok.loc[df_ok["condition"] == c, "resistance_ohm"].dropna().values 
            for c in order if len(df_ok[df_ok["condition"] == c]) > 0]
    labels = [c for c in order if len(df_ok[df_ok["condition"] == c]) > 0]
    
    if len(data) > 0:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True,
                        meanline=True, showfliers=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], [colors[c] for c in labels]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(0.8)
        
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=0.8)
        
        ax.set_ylabel("Resistance ($\\Omega$)", fontweight='bold')
        ax.set_xlabel("Configuration", fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_axisbelow(True)
    
    save_fig("resistance_boxplot", fig)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Generate all figures for IEEE paper."""
    
    print("="*70)
    print("Generating IEEE Paper Figures")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    print_trials = pd.read_csv(DATA_DIR / "print_trials.csv")
    electrical = pd.read_csv(DATA_DIR / "electrical_traces.csv")
    
    print(f"   ✓ Print trials: {len(print_trials)} records")
    print(f"   ✓ Electrical traces: {len(electrical)} records")
    
    # Generate figures matching LaTeX references
    print("\n2. Generating extrusion metrics...")
    plot_extrusion_onset_boxplot(print_trials)
    plot_flow_duration_boxplot(print_trials)
    
    print("\n3. Generating success rates...")
    plot_first_layer_success_rate(print_trials)
    plot_completion_rate(print_trials)
    
    print("\n4. Generating clog frequency...")
    plot_clogs_per_print(print_trials)
    
    print("\n5. Generating pressure simulations...")
    plot_pressure_simulation_baseline()
    plot_pressure_simulation_stabilized()
    
    print("\n6. Generating pressure trace...")
    plot_phat_trace()
    
    print("\n7. Generating electrical results...")
    plot_open_circuit_rate(electrical)
    plot_resistance_boxplot(electrical)
    
    print("\n" + "="*70)
    print(f"All figures saved to: {FIG_DIR}/")
    print("="*70)
    print("\nGenerated figures (matching LaTeX references):")
    print("  - extrusion_onset_boxplot.pdf")
    print("  - flow_duration_boxplot.pdf")
    print("  - first_layer_success_rate.pdf")
    print("  - completion_rate.pdf")
    print("  - clogs_per_print.pdf")
    print("  - pressure_simulation_baseline.pdf")
    print("  - pressure_simulation_stabilized.pdf")
    print("  - phat_trace.pdf")
    print("  - open_circuit_rate.pdf")
    print("  - resistance_boxplot.pdf")
    print("\nAll figures formatted for IEEE single-column width (3.5 inches).")

if __name__ == "__main__":
    main()
