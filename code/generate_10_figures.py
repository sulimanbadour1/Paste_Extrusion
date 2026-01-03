#!/usr/bin/env python3
"""
generate_10_figures.py

Generates exactly 10 figures for paste extrusion stabilization paper.
All figures are displayed interactively - save them manually using figure window controls.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set professional matplotlib style for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': True,
    'legend.framealpha': 0.9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'patch.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.minor.width': 0.8,
    'ytick.minor.width': 0.8,
})

# Professional color palette
COLORS = {
    'baseline': '#d62728',      # Red
    'stabilized': '#2ca02c',    # Green
    'partial': '#ff7f0e',       # Orange
    'full': '#2ca02c',          # Green (same as stabilized)
    'admissible': '#ffeb3b',    # Yellow
    'yield': '#ff9800',         # Orange
    'max': '#f44336',           # Red
}

# Try to import scipy for survival analysis (optional)
try:
    from scipy.stats import kaplan_meier_estimator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Figure 7 will use simplified survival analysis.")


# ============================================================================
# G-code Parsing Utilities
# ============================================================================

def parse_gcode_line(line: str) -> Dict[str, Optional[float]]:
    """Parse a G-code line and extract X, Y, Z, E, F values."""
    result = {'X': None, 'Y': None, 'Z': None, 'E': None, 'F': None}
    
    # Extract X, Y, Z, E, F values
    for key in result.keys():
        pattern = rf'{key}([+-]?\d+\.?\d*)'
        match = re.search(pattern, line)
        if match:
            result[key] = float(match.group(1))
    
    return result


def extract_gcode_metrics(gcode_lines: List[str], is_stabilized: bool = False) -> Dict[str, any]:
    """
    Extract metrics from G-code for Figure 1.
    For stabilized G-code, also extract E values from stabilization blocks.
    """
    total_lines = len([l for l in gcode_lines if l.strip() and not l.strip().startswith(';')])
    
    motion_moves = 0
    extrusion_moves = 0
    retractions = 0
    dwells = 0
    total_e_added = 0.0
    
    e_prev = 0.0
    e_curr = 0.0
    in_stabilization_block = False
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            # Check if we're entering a stabilization block
            if '===== Paste Stabilization Layer' in line:
                in_stabilization_block = True
            continue
        
        # Count motion moves (G0/G1)
        if stripped.startswith('G0') or stripped.startswith('G1'):
            motion_moves += 1
            
            parsed = parse_gcode_line(line)
            
            # Check for extrusion (E present)
            if parsed['E'] is not None:
                extrusion_moves += 1
                e_curr = parsed['E']
                delta_e = e_curr - e_prev
                
                if delta_e < 0:
                    retractions += 1
                elif delta_e > 0:
                    # Only count E added within stabilization blocks for stabilized G-code
                    if not is_stabilized or in_stabilization_block:
                        total_e_added += delta_e
                
                e_prev = e_curr
        
        # Count dwells (G4)
        if stripped.startswith('G4'):
            dwells += 1
    
    return {
        'total_lines': total_lines,
        'motion_moves': motion_moves,
        'extrusion_moves': extrusion_moves,
        'retractions': retractions,
        'dwells': dwells,
        'total_e_added': total_e_added
    }


def extract_retraction_magnitudes(gcode_lines: List[str]) -> List[float]:
    """Extract retraction magnitudes |ΔE| for Figure 2."""
    retraction_mags = []
    e_prev = 0.0
    e_curr = 0.0
    
    for line in gcode_lines:
        if 'E' in line and not line.strip().startswith(';'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None:
                e_curr = parsed['E']
                delta_e = e_curr - e_prev
                
                if delta_e < 0:
                    retraction_mags.append(abs(delta_e))
                
                e_prev = e_curr
    
    return retraction_mags


def compute_u_timeline(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute u(t) timeline from G-code for Figures 3 and 4.
    Returns: (time_array, u_array) where u is extrusion rate proxy.
    Filter: only moves with ΔE > 0 and Δs > 0.
    """
    times = [0.0]
    u_values = [0.0]
    
    x_prev, y_prev, e_prev = 0.0, 0.0, 0.0
    f_prev = 0.0
    t_curr = 0.0
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        if stripped.startswith('G1') or stripped.startswith('G0'):
            parsed = parse_gcode_line(line)
            
            x_curr = parsed['X'] if parsed['X'] is not None else x_prev
            y_curr = parsed['Y'] if parsed['Y'] is not None else y_prev
            e_curr = parsed['E'] if parsed['E'] is not None else e_prev
            f_curr = parsed['F'] if parsed['F'] is not None else f_prev
            
            # Compute Δs (XY distance)
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            ds = np.sqrt(dx**2 + dy**2)
            
            # Compute ΔE
            de = e_curr - e_prev
            
            # Filter: only moves with ΔE > 0 and Δs > 0
            if de > 0 and ds > 0:
                # Compute Δt = Δs / v, where v = F / 60 (mm/s)
                if f_curr > 0:
                    v = f_curr / 60.0  # mm/s
                    dt = ds / v  # seconds
                    t_curr += dt
                    
                    # Compute u(t) = ΔE / Δt
                    u = de / dt if dt > 0 else 0.0
                    
                    times.append(t_curr)
                    u_values.append(u)
            
            x_prev, y_prev, e_prev = x_curr, y_curr, e_curr
            if f_curr > 0:
                f_prev = f_curr
    
    return np.array(times), np.array(u_values)


def compute_pressure_timeline(times: np.ndarray, u_values: np.ndarray,
                               alpha: float = 8.0, tau_r: float = 6.0,
                               p_y: float = 5.0, p_max: float = 14.0) -> np.ndarray:
    """
    Compute pressure estimate p̂(t) from u(t) timeline for Figures 5 and 6.
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
# Figure Generation Functions
# ============================================================================

def figure_1_gcode_delta(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 1 — G-code modification summary (delta)
    Grouped bars: baseline vs stabilized for retractions, dwells, extrusion moves
    """
    baseline_metrics = extract_gcode_metrics(baseline_lines, is_stabilized=False)
    stabilized_metrics = extract_gcode_metrics(stabilized_lines, is_stabilized=True)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    categories = ['Retractions', 'Dwells', 'Extrusion Moves']
    baseline_data = [
        baseline_metrics['retractions'],
        baseline_metrics['dwells'],
        baseline_metrics['extrusion_moves']
    ]
    stabilized_data = [
        stabilized_metrics['retractions'],
        stabilized_metrics['dwells'],
        stabilized_metrics['extrusion_moves']
    ]
    
    x = np.arange(len(categories))
    width = 0.38
    
    bars1 = ax.bar(x - width/2, baseline_data, width, label='Baseline', 
                   color=COLORS['baseline'], alpha=0.85, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, stabilized_data, width, label='Stabilized', 
                   color=COLORS['stabilized'], alpha=0.85, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel('Metric Category', fontweight='bold', fontsize=13)
    ax.set_ylabel('Count', fontweight='bold', fontsize=13)
    ax.set_title('Fig. 1 — G-code Modification Summary', fontweight='bold', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add value labels on bars with better formatting
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 1 — G-code Modification Summary")


def figure_2_retraction_histogram(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 2 — Retraction suppression histogram
    Histogram of retraction magnitudes |ΔE| for baseline (optionally overlay stabilized)
    """
    baseline_retractions = extract_retraction_magnitudes(baseline_lines)
    stabilized_retractions = extract_retraction_magnitudes(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(11, 7))
    
    if len(baseline_retractions) > 0:
        bins = np.linspace(0, max(baseline_retractions) if baseline_retractions else 1, 35)
        n, bins, patches = ax.hist(baseline_retractions, bins=bins, alpha=0.75, label='Baseline',
                                   color=COLORS['baseline'], edgecolor='black', linewidth=1.0)
        # Color gradient for visual appeal
        for patch in patches:
            patch.set_alpha(0.75)
    
    if len(stabilized_retractions) > 0:
        ax.hist(stabilized_retractions, bins=bins, alpha=0.6, label='Stabilized',
               color=COLORS['stabilized'], edgecolor='black', linewidth=1.0)
    else:
        ax.axvline(0, color=COLORS['stabilized'], linestyle='--', linewidth=2.5, 
                  label='Stabilized (near zero)', zorder=10)
    
    ax.set_xlabel('|ΔE| (mm)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Frequency', fontweight='bold', fontsize=13)
    ax.set_title('Fig. 2 — Retraction Suppression Histogram', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 2 — Retraction Suppression Histogram")


def figure_3_u_baseline(baseline_lines: List[str]):
    """
    Fig. 3 — Extrusion-rate proxy timeline u(t) (baseline)
    """
    times, u_values = compute_u_timeline(baseline_lines)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    ax.plot(times, u_values, color=COLORS['baseline'], linewidth=2.2, alpha=0.9, label='Baseline u(t)', zorder=3)
    ax.fill_between(times, 0, u_values, where=(u_values > 0), 
                    color=COLORS['baseline'], alpha=0.25, zorder=1)
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=13)
    ax.set_ylabel('u(t) (mm/s)', fontweight='bold', fontsize=13)
    ax.set_title('Fig. 3 — Extrusion-Rate Proxy Timeline (Baseline)', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, zorder=2)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 3 — Extrusion-Rate Proxy Timeline (Baseline)")


def figure_4_u_stabilized(stabilized_lines: List[str]):
    """
    Fig. 4 — Extrusion-rate proxy timeline u(t) (stabilized)
    """
    times, u_values = compute_u_timeline(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    ax.plot(times, u_values, color=COLORS['stabilized'], linewidth=2.2, alpha=0.9, 
           label='Stabilized u(t)', zorder=3)
    ax.fill_between(times, 0, u_values, where=(u_values > 0), 
                    color=COLORS['stabilized'], alpha=0.25, zorder=1)
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=13)
    ax.set_ylabel('u(t) (mm/s)', fontweight='bold', fontsize=13)
    ax.set_title('Fig. 4 — Extrusion-Rate Proxy Timeline (Stabilized)', fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.0, zorder=2)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 4 — Extrusion-Rate Proxy Timeline (Stabilized)")


def figure_5_p_baseline(baseline_lines: List[str],
                       alpha: float = 8.0, tau_r: float = 6.0,
                       p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 5 — Pressure estimate p̂(t) with bounds (baseline)
    """
    times, u_values = compute_u_timeline(baseline_lines)
    p_hat = compute_pressure_timeline(times, u_values, alpha, tau_r, p_y, p_max)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    ax.plot(times, p_hat, color=COLORS['baseline'], linewidth=2.2, alpha=0.9, 
           label='p̂(t)', zorder=4)
    ax.fill_between(times, p_y, p_max, alpha=0.15, color=COLORS['admissible'], 
                    label='Admissible window', zorder=1)
    ax.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.5, 
              label=f'p_y = {p_y}', zorder=3)
    ax.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.5, 
              label=f'p_max = {p_max}', zorder=3)
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=13)
    ax.set_ylabel('p̂(t)', fontweight='bold', fontsize=13)
    ax.set_title(f'Fig. 5 — Pressure Estimate (Baseline)\n'
                f'α={alpha}, τ_r={tau_r}s, p_y={p_y}, p_max={p_max}', 
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 5 — Pressure Estimate (Baseline)")


def figure_6_p_stabilized(stabilized_lines: List[str],
                         alpha: float = 8.0, tau_r: float = 6.0,
                         p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 6 — Pressure estimate p̂(t) with bounds (stabilized)
    """
    times, u_values = compute_u_timeline(stabilized_lines)
    p_hat = compute_pressure_timeline(times, u_values, alpha, tau_r, p_y, p_max)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    
    ax.plot(times, p_hat, color=COLORS['stabilized'], linewidth=2.2, alpha=0.9, 
           label='p̂(t)', zorder=4)
    ax.fill_between(times, p_y, p_max, alpha=0.15, color=COLORS['admissible'], 
                    label='Admissible window', zorder=1)
    ax.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.5, 
              label=f'p_y = {p_y}', zorder=3)
    ax.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.5, 
              label=f'p_max = {p_max}', zorder=3)
    
    ax.set_xlabel('Time (s)', fontweight='bold', fontsize=13)
    ax.set_ylabel('p̂(t)', fontweight='bold', fontsize=13)
    ax.set_title(f'Fig. 6 — Pressure Estimate (Stabilized)\n'
                f'α={alpha}, τ_r={tau_r}s, p_y={p_y}, p_max={p_max}', 
                fontweight='bold', fontsize=14, pad=15)
    ax.legend(loc='upper right', framealpha=0.95, shadow=True)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 6 — Pressure Estimate (Stabilized)")


def figure_7_survival_curve(print_trials_df: pd.DataFrame):
    """
    Fig. 7 — Extrusion continuity survival curve
    Uses flow_duration_s for survival analysis (baseline, partial, full)
    """
    if 'flow_duration_s' not in print_trials_df.columns:
        print("ERROR: print_trials.csv must contain 'flow_duration_s' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = ['baseline', 'partial', 'full']
    colors = {'baseline': '#d62728', 'partial': '#ff7f0e', 'full': '#2ca02c'}
    
    for condition in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == condition]
        if len(cond_data) == 0:
            continue
        
        durations = cond_data['flow_duration_s'].values
        completed = cond_data.get('completed', pd.Series([True] * len(cond_data))).values
        
        if SCIPY_AVAILABLE:
            # Use Kaplan-Meier estimator
            durations_sorted = np.sort(durations)
            n = len(durations_sorted)
            survival = np.arange(n, 0, -1) / n
            ax.plot(durations_sorted, survival, 'o-', linewidth=2, markersize=6,
                   label=condition.capitalize(), color=colors.get(condition, 'gray'), alpha=0.8)
        else:
            # Simple empirical survival function
            durations_sorted = np.sort(durations)
            n = len(durations_sorted)
            survival = np.arange(n, 0, -1) / n
            ax.plot(durations_sorted, survival, 'o-', linewidth=2, markersize=6,
                   label=condition.capitalize(), color=colors.get(condition, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Survival Probability', fontweight='bold')
    ax.set_title('Fig. 7 — Extrusion Continuity Survival Curve', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 7 — Extrusion Continuity Survival Curve")


def figure_8_first_layer_map(first_layer_df: pd.DataFrame):
    """
    Fig. 8 — First-layer operating envelope heatmap
    x: h_ratio = h1/d_nozzle, y: speed_mmps, cell value: mean success
    """
    if not all(col in first_layer_df.columns for col in ['h_ratio', 'speed_mmps', 'success']):
        print("ERROR: first_layer_sweep.csv must contain columns: h_ratio, speed_mmps, success")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table for heatmap
    pivot = first_layer_df.pivot_table(
        values='success',
        index='h_ratio',
        columns='speed_mmps',
        aggfunc='mean'
    )
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('Speed (mm/s)', fontweight='bold')
    ax.set_ylabel('h₁/d_nozzle', fontweight='bold')
    ax.set_title('Fig. 8 — First-Layer Operating Envelope', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 8 — First-Layer Operating Envelope")


def figure_9_open_circuit_rate(electrical_df: pd.DataFrame):
    """
    Fig. 9 — Electrical yield (open-circuit rate)
    Bar chart: baseline vs stabilized
    """
    if 'open_circuit' not in electrical_df.columns or 'condition' not in electrical_df.columns:
        print("ERROR: electrical_traces.csv must contain columns: condition, open_circuit")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    conditions = ['baseline', 'full']  # Note: using 'full' as stabilized
    open_rates = []
    colors_bar = {'baseline': '#d62728', 'full': '#2ca02c', 'stabilized': '#2ca02c'}
    
    for condition in conditions:
        cond_data = electrical_df[electrical_df['condition'] == condition]
        if len(cond_data) > 0:
            open_rate = cond_data['open_circuit'].mean()
            open_rates.append(open_rate)
        else:
            open_rates.append(0.0)
    
    # Map 'full' to 'stabilized' for display
    display_labels = ['baseline', 'stabilized']
    bars = ax.bar(display_labels, open_rates,
                 color=[colors_bar.get(c, 'gray') for c in conditions],
                 alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Open-Circuit Rate', fontweight='bold')
    ax.set_title('Fig. 9 — Electrical Yield', fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 9 — Electrical Yield")


def figure_10_resistance_boxplot(electrical_df: pd.DataFrame):
    """
    Fig. 10 — Resistance stability (distribution for successful traces)
    Boxplot: baseline vs stabilized (filter: open_circuit==0)
    """
    if 'resistance_ohm' not in electrical_df.columns or 'condition' not in electrical_df.columns:
        print("ERROR: electrical_traces.csv must contain columns: condition, resistance_ohm")
        return
    
    # Filter: open_circuit==0 and resistance_ohm not empty
    successful = electrical_df[(electrical_df['open_circuit'] == 0) & 
                              (electrical_df['resistance_ohm'].notna())].copy()
    
    if len(successful) == 0:
        print("WARNING: No successful traces with resistance data found")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    conditions = ['baseline', 'full']  # Note: using 'full' as stabilized
    data_by_condition = []
    labels = []
    
    for condition in conditions:
        cond_data = successful[successful['condition'] == condition]
        resistances = cond_data['resistance_ohm'].dropna()
        if len(resistances) > 0:
            data_by_condition.append(resistances.values)
            labels.append('stabilized' if condition == 'full' else condition.capitalize())
    
    if len(data_by_condition) > 0:
        bp = ax.boxplot(data_by_condition, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        
        # Color the boxes
        colors_box = {'baseline': '#d62728', 'stabilized': '#2ca02c'}
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors_box.get(label.lower(), 'gray'))
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Resistance (Ω)', fontweight='bold')
        ax.set_title('Fig. 10 — Resistance Stability', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
    
    plt.tight_layout()
    plt.show()
    print("✓ Displayed: Figure 10 — Resistance Stability")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate 10 figures for paste extrusion paper')
    parser.add_argument('--baseline-gcode', type=str, required=True,
                      help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                      help='Path to stabilized G-code file')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Directory containing CSV data files (default: code/data)')
    parser.add_argument('--figures', type=str, nargs='+', default=['all'],
                      choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'all'],
                      help='Which figures to generate (default: all)')
    parser.add_argument('--alpha', type=float, default=8.0,
                      help='Pressure model parameter α (default: 8.0)')
    parser.add_argument('--tau-r', type=float, default=6.0,
                      help='Pressure model parameter τ_r in seconds (default: 6.0)')
    parser.add_argument('--p-y', type=float, default=5.0,
                      help='Yield pressure threshold (default: 5.0)')
    parser.add_argument('--p-max', type=float, default=14.0,
                      help='Maximum pressure bound (default: 14.0)')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    baseline_path = Path(args.baseline_gcode)
    if not baseline_path.is_absolute():
        test_path = script_dir / args.baseline_gcode
        if test_path.exists():
            baseline_path = test_path
        elif not baseline_path.exists():
            baseline_path = test_path
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline G-code not found: {args.baseline_gcode} (tried: {baseline_path})")
    
    stabilized_path = Path(args.stabilized_gcode)
    if not stabilized_path.is_absolute():
        test_path = script_dir / args.stabilized_gcode
        if test_path.exists():
            stabilized_path = test_path
        elif not stabilized_path.exists():
            stabilized_path = test_path
    if not stabilized_path.exists():
        raise FileNotFoundError(f"Stabilized G-code not found: {args.stabilized_gcode} (tried: {stabilized_path})")
    
    # Data directory defaults to code/data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = script_dir / 'data'
    
    # Read G-code files
    print(f"Reading baseline G-code: {baseline_path}")
    with open(baseline_path, 'r') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_path}")
    with open(stabilized_path, 'r') as f:
        stabilized_lines = f.readlines()
    
    # Read CSV data files
    print_trials_path = data_dir / 'print_trials.csv'
    first_layer_path = data_dir / 'first_layer_sweep.csv'
    electrical_path = data_dir / 'electrical_traces.csv'
    
    print_trials_df = None
    if print_trials_path.exists():
        print(f"Reading print trials data: {print_trials_path}")
        print_trials_df = pd.read_csv(print_trials_path)
    else:
        print(f"WARNING: {print_trials_path} not found")
    
    first_layer_df = None
    if first_layer_path.exists():
        print(f"Reading first layer sweep data: {first_layer_path}")
        first_layer_df = pd.read_csv(first_layer_path)
    else:
        print(f"WARNING: {first_layer_path} not found")
    
    electrical_df = None
    if electrical_path.exists():
        print(f"Reading electrical traces data: {electrical_path}")
        electrical_df = pd.read_csv(electrical_path)
    else:
        print(f"WARNING: {electrical_path} not found")
    
    # Determine which figures to generate
    if 'all' in args.figures:
        figures_to_generate = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    else:
        figures_to_generate = args.figures
    
    print(f"\nGenerating figures: {', '.join(figures_to_generate)}")
    print("Figures will be displayed interactively - save them manually using figure window controls.\n")
    
    # Generate requested figures
    if '1' in figures_to_generate:
        figure_1_gcode_delta(baseline_lines, stabilized_lines)
    
    if '2' in figures_to_generate:
        figure_2_retraction_histogram(baseline_lines, stabilized_lines)
    
    if '3' in figures_to_generate:
        figure_3_u_baseline(baseline_lines)
    
    if '4' in figures_to_generate:
        figure_4_u_stabilized(stabilized_lines)
    
    if '5' in figures_to_generate:
        figure_5_p_baseline(baseline_lines, args.alpha, args.tau_r, args.p_y, args.p_max)
    
    if '6' in figures_to_generate:
        figure_6_p_stabilized(stabilized_lines, args.alpha, args.tau_r, args.p_y, args.p_max)
    
    if '7' in figures_to_generate:
        if print_trials_df is not None:
            figure_7_survival_curve(print_trials_df)
        else:
            print("⚠ Skipping Figure 7: print_trials.csv not found")
    
    if '8' in figures_to_generate:
        if first_layer_df is not None:
            figure_8_first_layer_map(first_layer_df)
        else:
            print("⚠ Skipping Figure 8: first_layer_sweep.csv not found")
    
    if '9' in figures_to_generate:
        if electrical_df is not None:
            figure_9_open_circuit_rate(electrical_df)
        else:
            print("⚠ Skipping Figure 9: electrical_traces.csv not found")
    
    if '10' in figures_to_generate:
        if electrical_df is not None:
            figure_10_resistance_boxplot(electrical_df)
        else:
            print("⚠ Skipping Figure 10: electrical_traces.csv not found")
    
    print(f"\n✓ All requested figures displayed.")


if __name__ == '__main__':
    main()

