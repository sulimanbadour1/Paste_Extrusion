#!/usr/bin/env python3
"""
generate_effectiveness_figures.py

Generates additional figures highlighting stabilizer effectiveness.
These figures complement the main paper figures and provide deeper insights
into how the stabilizer improves paste extrusion printing.

All figures use IEEE-compatible styling with bold text and professional formatting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib
import os
import sys

# Force interactive backend for GUI display
if sys.platform == 'darwin':
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'

try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('macosx')
        except:
            pass

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
import matplotlib.patches as mpatches

# Enable interactive mode
plt.ion()

# IEEE-compatible styling
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 0.98,
    'legend.edgecolor': 'black',
    'figure.titlesize': 12,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
})

# Professional color palette
COLORS = {
    'baseline': '#d62728',      # Red
    'stabilized': '#2ca02c',    # Green
    'partial': '#ff7f0e',       # Orange
    'full': '#2ca02c',          # Green (same as stabilized)
    'in_bounds': '#2ca02c',     # Green for compliance
    'out_bounds': '#d62728',    # Red for non-compliance
    'admissible': '#ffeb3b',    # Yellow
    'yield': '#ff9800',         # Orange
    'max': '#f44336',           # Red
    'action_low_prime': '#3498DB',    # Blue
    'action_dwell': '#9B59B6',       # Purple
    'action_suppress': '#E67E22',     # Dark orange
}

# ============================================================================
# G-code Parsing Utilities
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
    """Compute u(t) timeline from G-code."""
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
            
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            ds = np.sqrt(dx**2 + dy**2)
            de = e_curr - e_prev
            
            if de > 0 and ds > 0:
                if f_curr > 0:
                    v = f_curr / 60.0
                    dt = ds / v
                    t_curr += dt
                    u = de / dt if dt > 0 else 0.0
                    times.append(t_curr)
                    u_values.append(u)
            
            x_prev, y_prev, e_prev = x_curr, y_curr, e_curr
            if f_curr > 0:
                f_prev = f_curr
    
    return np.array(times), np.array(u_values)


def extract_layers(gcode_lines: List[str]) -> Dict[int, List[int]]:
    """Extract layer information from G-code. Returns dict mapping layer_num to line indices."""
    layers = {}
    current_layer = 0
    z_prev = 0.0
    z_threshold = 0.1  # mm threshold for layer change
    
    for i, line in enumerate(gcode_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        if stripped.startswith('G1') or stripped.startswith('G0'):
            parsed = parse_gcode_line(line)
            if parsed['Z'] is not None:
                z_curr = parsed['Z']
                if abs(z_curr - z_prev) > z_threshold:
                    current_layer += 1
                    z_prev = z_curr
                
                if current_layer not in layers:
                    layers[current_layer] = []
                layers[current_layer].append(i)
    
    return layers


# ============================================================================
# Effectiveness Figures
# ============================================================================

def figure_24_pressure_compliance_timeline(run_log_path: Path, p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 24 — Pressure Window Compliance Over Time
    Shows when pressure stays within admissible bounds (p_y to p_max)
    """
    if not run_log_path.exists():
        print(f"ERROR: run_log.csv not found at {run_log_path}")
        return
    
    df = pd.read_csv(run_log_path)
    
    if 'p_hat' not in df.columns or 't_s' not in df.columns:
        print("ERROR: run_log.csv must contain 'p_hat' and 't_s' columns")
        return
    
    p_hat = pd.to_numeric(df['p_hat'], errors='coerce').dropna().values
    t_s = pd.to_numeric(df['t_s'], errors='coerce').dropna().values
    
    if len(p_hat) == 0 or len(t_s) == 0:
        print("ERROR: No valid pressure data found")
        return
    
    # Ensure arrays are same length
    min_len = min(len(p_hat), len(t_s))
    p_hat = p_hat[:min_len]
    t_s = t_s[:min_len]
    
    # Calculate compliance
    in_bounds = (p_hat >= p_y) & (p_hat <= p_max)
    compliance_pct = (in_bounds.sum() / len(in_bounds)) * 100 if len(in_bounds) > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.0), sharex=True)
    fig.patch.set_facecolor('white')
    
    # Top plot: Pressure trace with bounds
    ax1.plot(t_s, p_hat, color='black', linewidth=1.5, alpha=0.7, label='p̂(t)', zorder=3)
    ax1.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.0, label=f'p_y = {p_y:.1f}', zorder=2)
    ax1.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.0, label=f'p_max = {p_max:.1f}', zorder=2)
    ax1.fill_between(t_s, p_y, p_max, alpha=0.2, color=COLORS['in_bounds'], label='Admissible Window', zorder=1)
    
    # Color-code by compliance
    ax1.fill_between(t_s, p_y, p_max, where=in_bounds, alpha=0.3, color=COLORS['in_bounds'], zorder=0)
    ax1.fill_between(t_s, 0, p_y, where=~in_bounds, alpha=0.2, color=COLORS['out_bounds'], zorder=0)
    ax1.fill_between(t_s, p_max, p_max*1.2, where=p_hat > p_max, alpha=0.2, color=COLORS['out_bounds'], zorder=0)
    
    ax1.set_ylabel('Pressure p̂(t) (kPa)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title(f'Pressure Compliance Timeline (Compliance: {compliance_pct:.1f}%)', 
                 fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(loc='best', fontsize=10, framealpha=0.98, edgecolor='black')
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    
    # Bold tick labels
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    # Bottom plot: Binary compliance indicator
    compliance_binary = in_bounds.astype(int)
    ax2.fill_between(t_s, 0, compliance_binary, alpha=0.6, color=COLORS['in_bounds'], 
                     label='Within Bounds', step='post', zorder=2)
    ax2.fill_between(t_s, compliance_binary, 1, alpha=0.6, color=COLORS['out_bounds'],
                     label='Out of Bounds', step='post', zorder=1)
    ax2.set_ylim([-0.1, 1.1])
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Compliance', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Out', 'In'], fontweight='bold', fontfamily='serif')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.98, edgecolor='black')
    ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 24 — Pressure Window Compliance Over Time")


def figure_25_action_frequency(run_log_path: Path):
    """
    Fig. 25 — Action Frequency Analysis
    Shows when and how often the stabilizer intervenes
    """
    if not run_log_path.exists():
        print(f"ERROR: run_log.csv not found at {run_log_path}")
        return
    
    df = pd.read_csv(run_log_path)
    
    if 'action' not in df.columns or 't_s' not in df.columns:
        print("ERROR: run_log.csv must contain 'action' and 't_s' columns")
        return
    
    t_s = pd.to_numeric(df['t_s'], errors='coerce').dropna().values
    actions = df['action'].values
    
    # Filter to same length
    min_len = min(len(t_s), len(actions))
    t_s = t_s[:min_len]
    actions = actions[:min_len]
    
    # Count actions
    action_counts = pd.Series(actions).value_counts()
    
    # Categorize actions
    action_categories = {
        'low_prime': [],
        'relax_dwell': [],
        'retract_suppressed': [],
        'other': []
    }
    
    for i, action in enumerate(actions):
        if pd.isna(action):
            continue
        action_str = str(action).lower()
        if 'low_prime' in action_str or 'prime' in action_str:
            action_categories['low_prime'].append(t_s[i])
        elif 'dwell' in action_str or 'relax' in action_str:
            action_categories['relax_dwell'].append(t_s[i])
        elif 'retract' in action_str or 'suppress' in action_str:
            action_categories['retract_suppressed'].append(t_s[i])
        else:
            action_categories['other'].append(t_s[i])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 5.5), sharex=True)
    fig.patch.set_facecolor('white')
    
    # Top plot: Timeline of actions
    if action_categories['low_prime']:
        ax1.scatter(action_categories['low_prime'], [1]*len(action_categories['low_prime']),
                   color=COLORS['action_low_prime'], label='Low Prime', s=30, alpha=0.7, zorder=3)
    if action_categories['relax_dwell']:
        ax1.scatter(action_categories['relax_dwell'], [2]*len(action_categories['relax_dwell']),
                   color=COLORS['action_dwell'], label='Relax Dwell', s=30, alpha=0.7, zorder=3)
    if action_categories['retract_suppressed']:
        ax1.scatter(action_categories['retract_suppressed'], [3]*len(action_categories['retract_suppressed']),
                   color=COLORS['action_suppress'], label='Retraction Suppressed', s=30, alpha=0.7, zorder=3)
    
    ax1.set_ylabel('Action Type', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Stabilizer Action Timeline', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_yticks([1, 2, 3])
    ax1.set_yticklabels(['Low Prime', 'Relax Dwell', 'Retract Suppressed'], fontweight='bold', fontfamily='serif')
    ax1.set_ylim([0.5, 3.5])
    ax1.legend(loc='best', fontsize=10, framealpha=0.98, edgecolor='black')
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    
    # Bottom plot: Action frequency histogram
    action_names = ['Low Prime', 'Relax Dwell', 'Retract Suppressed']
    action_values = [
        len(action_categories['low_prime']),
        len(action_categories['relax_dwell']),
        len(action_categories['retract_suppressed'])
    ]
    colors_list = [COLORS['action_low_prime'], COLORS['action_dwell'], COLORS['action_suppress']]
    
    bars = ax2.bar(action_names, action_values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_title('Action Frequency Summary', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    
    # Add value labels on bars
    for bar, val in zip(bars, action_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for label in ax2.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 25 — Action Frequency Analysis")


def figure_26_layer_by_layer_analysis(baseline_lines: List[str], stabilized_lines: List[str], 
                                      run_log_path: Optional[Path] = None):
    """
    Fig. 26 — Layer-by-Layer Analysis
    Shows how stabilization affects different layers
    """
    baseline_layers = extract_layers(baseline_lines)
    stabilized_layers = extract_layers(stabilized_lines)
    
    # Extract metrics per layer
    baseline_metrics = {}
    stabilized_metrics = {}
    
    for layer_num in baseline_layers.keys():
        layer_lines = [baseline_lines[i] for i in baseline_layers[layer_num]]
        retractions = sum(1 for line in layer_lines if 'E-' in line or (parse_gcode_line(line).get('E', 0) or 0) < 0)
        extrusions = sum(1 for line in layer_lines if parse_gcode_line(line).get('E', 0) or 0 > 0)
        baseline_metrics[layer_num] = {'retractions': retractions, 'extrusions': extrusions}
    
    for layer_num in stabilized_layers.keys():
        layer_lines = [stabilized_lines[i] for i in stabilized_layers[layer_num]]
        retractions = sum(1 for line in layer_lines if 'E-' in line or (parse_gcode_line(line).get('E', 0) or 0) < 0)
        extrusions = sum(1 for line in layer_lines if parse_gcode_line(line).get('E', 0) or 0 > 0)
        dwells = sum(1 for line in layer_lines if 'G4' in line or 'dwell' in line.lower())
        stabilized_metrics[layer_num] = {'retractions': retractions, 'extrusions': extrusions, 'dwells': dwells}
    
    # Get common layers
    all_layers = sorted(set(list(baseline_metrics.keys()) + list(stabilized_metrics.keys())))
    
    if len(all_layers) == 0:
        print("ERROR: No layers found in G-code")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.5))
    fig.patch.set_facecolor('white')
    fig.suptitle('Layer-by-Layer Analysis', fontsize=14, fontweight='bold', fontfamily='serif', y=0.98)
    
    # Extract data
    layer_nums = all_layers[:20]  # Limit to first 20 layers for clarity
    baseline_retractions = [baseline_metrics.get(l, {}).get('retractions', 0) for l in layer_nums]
    stabilized_retractions = [stabilized_metrics.get(l, {}).get('retractions', 0) for l in layer_nums]
    stabilized_dwells = [stabilized_metrics.get(l, {}).get('dwells', 0) for l in layer_nums]
    baseline_extrusions = [baseline_metrics.get(l, {}).get('extrusions', 0) for l in layer_nums]
    stabilized_extrusions = [stabilized_metrics.get(l, {}).get('extrusions', 0) for l in layer_nums]
    
    # Plot 1: Retractions per layer
    ax1 = axes[0, 0]
    width = 0.35
    x = np.arange(len(layer_nums))
    ax1.bar(x - width/2, baseline_retractions, width, label='Baseline', 
           color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.bar(x + width/2, stabilized_retractions, width, label='Stabilized',
           color=COLORS['stabilized'], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Layer Number', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Retractions', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.set_title('Retractions per Layer', fontsize=11, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=9, framealpha=0.98, edgecolor='black')
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_xticks(x[::max(1, len(x)//10)])
    ax1.set_xticklabels([layer_nums[i] for i in range(0, len(layer_nums), max(1, len(layer_nums)//10))], 
                        fontweight='bold', fontfamily='serif')
    
    # Plot 2: Dwells added per layer
    ax2 = axes[0, 1]
    ax2.bar(layer_nums, stabilized_dwells, color=COLORS['action_dwell'], alpha=0.8, 
           edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Layer Number', fontsize=11, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Dwells Added', fontsize=11, fontweight='bold', fontfamily='serif')
    ax2.set_title('Dwells Added per Layer', fontsize=11, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xticks(layer_nums[::max(1, len(layer_nums)//10)])
    ax2.set_xticklabels([layer_nums[i] for i in range(0, len(layer_nums), max(1, len(layer_nums)//10))],
                       fontweight='bold', fontfamily='serif')
    
    # Plot 3: Extrusion moves per layer
    ax3 = axes[1, 0]
    ax3.bar(x - width/2, baseline_extrusions, width, label='Baseline',
           color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.bar(x + width/2, stabilized_extrusions, width, label='Stabilized',
           color=COLORS['stabilized'], alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Layer Number', fontsize=11, fontweight='bold', fontfamily='serif')
    ax3.set_ylabel('Extrusion Moves', fontsize=11, fontweight='bold', fontfamily='serif')
    ax3.set_title('Extrusion Moves per Layer', fontsize=11, fontweight='bold', fontfamily='serif')
    ax3.legend(fontsize=9, framealpha=0.98, edgecolor='black')
    ax3.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax3.set_axisbelow(True)
    ax3.set_xticks(x[::max(1, len(x)//10)])
    ax3.set_xticklabels([layer_nums[i] for i in range(0, len(layer_nums), max(1, len(layer_nums)//10))],
                       fontweight='bold', fontfamily='serif')
    
    # Plot 4: Retraction elimination rate per layer
    ax4 = axes[1, 1]
    elimination_rates = []
    for l in layer_nums:
        base_ret = baseline_metrics.get(l, {}).get('retractions', 0)
        stab_ret = stabilized_metrics.get(l, {}).get('retractions', 0)
        if base_ret > 0:
            rate = ((base_ret - stab_ret) / base_ret) * 100
        else:
            rate = 0
        elimination_rates.append(rate)
    
    ax4.bar(layer_nums, elimination_rates, color=COLORS['stabilized'], alpha=0.8,
           edgecolor='black', linewidth=1.2)
    ax4.axhline(100, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax4.set_xlabel('Layer Number', fontsize=11, fontweight='bold', fontfamily='serif')
    ax4.set_ylabel('Elimination Rate (%)', fontsize=11, fontweight='bold', fontfamily='serif')
    ax4.set_title('Retraction Elimination Rate', fontsize=11, fontweight='bold', fontfamily='serif')
    ax4.set_ylim([0, 105])
    ax4.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax4.set_axisbelow(True)
    ax4.set_xticks(layer_nums[::max(1, len(layer_nums)//10)])
    ax4.set_xticklabels([layer_nums[i] for i in range(0, len(layer_nums), max(1, len(layer_nums)//10))],
                       fontweight='bold', fontfamily='serif')
    
    # Bold all tick labels
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)
    print("[OK] Displayed: Figure 26 — Layer-by-Layer Analysis")


def figure_27_failure_mode_analysis(print_trials_df: pd.DataFrame, electrical_df: pd.DataFrame):
    """
    Fig. 27 — Failure Mode Analysis
    Breakdown of failure types by condition
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 27")
        return
    
    conditions = ['baseline', 'partial', 'full']
    condition_labels = ['Baseline', 'Partial', 'Full']
    
    # Analyze failures
    failure_data = {}
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        total = len(cond_data)
        if total == 0:
            continue
        
        incomplete = (cond_data['completed'] == 0).sum()
        clogs = cond_data['clogs'].sum() if 'clogs' in cond_data.columns else 0
        first_layer_fail = (cond_data.get('first_layer_success', pd.Series([1]*len(cond_data))) == 0).sum()
        
        # Electrical failures
        elec_failures = 0
        if electrical_df is not None and len(electrical_df) > 0:
            elec_cond = electrical_df[electrical_df['condition'] == cond]
            if len(elec_cond) > 0:
                elec_failures = (elec_cond['open_circuit'] == 1).sum()
        
        failure_data[cond] = {
            'Incomplete Prints': incomplete,
            'Clogs': clogs,
            'First Layer Failures': first_layer_fail,
            'Electrical Failures': elec_failures,
            'Total Trials': total
        }
    
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor('white')
    
    # Prepare data for stacked bar chart
    categories = condition_labels
    failure_types = ['Incomplete Prints', 'Clogs', 'First Layer Failures', 'Electrical Failures']
    colors_failures = [COLORS['baseline'], COLORS['partial'], COLORS['yield'], COLORS['max']]
    
    bottom = np.zeros(len(categories))
    bars_list = []
    
    for i, failure_type in enumerate(failure_types):
        values = [failure_data.get(cond, {}).get(failure_type, 0) for cond in conditions]
        bars = ax.bar(categories, values, bottom=bottom, label=failure_type,
                     color=colors_failures[i], alpha=0.8, edgecolor='black', linewidth=1.2)
        bars_list.append(bars)
        bottom += values
    
    ax.set_ylabel('Failure Count', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_title('Failure Mode Analysis by Condition', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.legend(loc='upper left', fontsize=10, framealpha=0.98, edgecolor='black')
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Add total trial counts
    for i, cond in enumerate(conditions):
        total = failure_data.get(cond, {}).get('Total Trials', 0)
        if total > 0:
            ax.text(i, bottom[i] + 0.5, f'n={total}', ha='center', va='bottom',
                   fontsize=10, fontweight='bold', fontfamily='serif')
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 27 — Failure Mode Analysis")


def figure_28_feed_rate_scaling(run_log_path: Path):
    """
    Fig. 28 — Feed Rate Scaling Distribution
    Shows how often and by how much feed rates are scaled
    """
    if not run_log_path.exists():
        print(f"ERROR: run_log.csv not found at {run_log_path}")
        return
    
    df = pd.read_csv(run_log_path)
    
    if 'feed_scale' not in df.columns:
        print("ERROR: run_log.csv must contain 'feed_scale' column")
        return
    
    feed_scales = pd.to_numeric(df['feed_scale'], errors='coerce').dropna()
    feed_scales = feed_scales[feed_scales > 0]  # Remove invalid values
    
    if len(feed_scales) == 0:
        print("WARNING: No valid feed scale data found")
        return
    
    no_change = (feed_scales == 1.0).sum()
    scaled_up = (feed_scales > 1.0).sum()
    scaled_down = (feed_scales < 1.0).sum()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left plot: Histogram
    ax1.hist(feed_scales, bins=50, color=COLORS['stabilized'], alpha=0.7, 
            edgecolor='black', linewidth=1.0)
    ax1.axvline(1.0, color='black', linestyle='--', linewidth=2.0, label='No Change (1.0)')
    ax1.set_xlabel('Feed Scale Factor', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Feed Rate Scaling Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=10, framealpha=0.98, edgecolor='black')
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Right plot: Summary statistics
    categories = ['No Change\n(=1.0)', 'Scaled Up\n(>1.0)', 'Scaled Down\n(<1.0)']
    values = [no_change, scaled_up, scaled_down]
    colors_summary = ['gray', COLORS['stabilized'], COLORS['baseline']]
    
    bars = ax2.bar(categories, values, color=colors_summary, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_title('Scaling Summary', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add percentage labels
    total = sum(values)
    if total > 0:
        for i, (bar, val) in enumerate(zip(bars, values)):
            pct = (val / total) * 100
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{pct:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if pct > 20 else 'black')
    
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 28 — Feed Rate Scaling Distribution")


def figure_29_print_time_impact(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 29 — Print Time Impact
    Comparison of total print time before/after stabilization
    """
    def compute_print_time(gcode_lines: List[str]) -> float:
        """Compute total print time from G-code."""
        total_time = 0.0
        x_prev, y_prev = 0.0, 0.0
        f_prev = 0.0
        
        for line in gcode_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            
            if stripped.startswith('G1') or stripped.startswith('G0'):
                parsed = parse_gcode_line(line)
                x_curr = parsed['X'] if parsed['X'] is not None else x_prev
                y_curr = parsed['Y'] if parsed['Y'] is not None else y_prev
                f_curr = parsed['F'] if parsed['F'] is not None else f_prev
                
                dx = x_curr - x_prev
                dy = y_curr - y_prev
                ds = np.sqrt(dx**2 + dy**2)
                
                if f_curr > 0 and ds > 0:
                    v = f_curr / 60.0  # mm/s
                    dt = ds / v
                    total_time += dt
                
                # Check for dwell commands
                if 'G4' in stripped:
                    # Extract dwell time
                    s_match = re.search(r'S(\d+\.?\d*)', stripped)
                    if s_match:
                        dwell_time = float(s_match.group(1))
                        total_time += dwell_time
                
                x_prev, y_prev = x_curr, y_curr
                if f_curr > 0:
                    f_prev = f_curr
        
        return total_time
    
    baseline_time = compute_print_time(baseline_lines)
    stabilized_time = compute_print_time(stabilized_lines)
    time_increase = stabilized_time - baseline_time
    time_increase_pct = (time_increase / baseline_time * 100) if baseline_time > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left plot: Bar comparison
    categories = ['Baseline', 'Stabilized']
    times = [baseline_time, stabilized_time]
    colors_times = [COLORS['baseline'], COLORS['stabilized']]
    
    bars = ax1.bar(categories, times, color=colors_times, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Print Time (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Total Print Time Comparison', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right plot: Time increase breakdown
    # Estimate time from different sources
    dwell_time = sum(0.35 for line in stabilized_lines if 'G4' in line)  # Typical dwell duration
    micro_prime_time = sum(0.1 for line in stabilized_lines if 'micro-prime' in line.lower())  # Estimate
    
    breakdown_categories = ['Dwells', 'Micro-primes', 'Other']
    breakdown_times = [dwell_time, micro_prime_time, max(0, time_increase - dwell_time - micro_prime_time)]
    breakdown_colors = [COLORS['action_dwell'], COLORS['action_low_prime'], 'gray']
    
    bars2 = ax2.bar(breakdown_categories, breakdown_times, color=breakdown_colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time Added (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_title(f'Time Overhead Breakdown\n(Total: +{time_increase_pct:.1f}%)', 
                 fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels
    for bar, time_val in zip(bars2, breakdown_times):
        if time_val > 0:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print(f"[OK] Displayed: Figure 29 — Print Time Impact (Increase: {time_increase_pct:.1f}%)")


def figure_30_extrusion_variability(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 30 — Extrusion Rate Variability (Coefficient of Variation)
    Shows flow consistency improvement
    """
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Calculate coefficient of variation
    cv_baseline = (np.std(u_baseline) / np.mean(u_baseline) * 100) if np.mean(u_baseline) > 0 else 0
    cv_stabilized = (np.std(u_stabilized) / np.mean(u_stabilized) * 100) if np.mean(u_stabilized) > 0 else 0
    
    improvement = ((cv_baseline - cv_stabilized) / cv_baseline * 100) if cv_baseline > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left plot: CV comparison
    categories = ['Baseline', 'Stabilized']
    cv_values = [cv_baseline, cv_stabilized]
    colors_cv = [COLORS['baseline'], COLORS['stabilized']]
    
    bars = ax1.bar(categories, cv_values, color=colors_cv, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Extrusion Rate Variability', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Add value labels
    for bar, cv_val in zip(bars, cv_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{cv_val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right plot: Distribution comparison
    if len(u_baseline) > 0 and len(u_stabilized) > 0:
        # Downsample if too many points
        n_plot = min(1000, len(u_baseline), len(u_stabilized))
        u_b_plot = u_baseline[:n_plot] if len(u_baseline) > n_plot else u_baseline
        u_s_plot = u_stabilized[:n_plot] if len(u_stabilized) > n_plot else u_stabilized
        
        ax2.hist(u_b_plot, bins=30, alpha=0.6, label='Baseline', color=COLORS['baseline'],
                edgecolor='black', linewidth=0.8)
        ax2.hist(u_s_plot, bins=30, alpha=0.6, label='Stabilized', color=COLORS['stabilized'],
                edgecolor='black', linewidth=0.8)
        ax2.set_xlabel('Extrusion Rate u(t)', fontsize=12, fontweight='bold', fontfamily='serif')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold', fontfamily='serif')
        ax2.set_title('Extrusion Rate Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
        ax2.legend(fontsize=10, framealpha=0.98, edgecolor='black')
        ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
        ax2.set_axisbelow(True)
    
    # Add improvement annotation
    ax1.text(0.5, max(cv_values) * 0.9, f'Improvement: {improvement:.1f}%',
            ha='center', fontsize=11, fontweight='bold', fontfamily='serif',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='black'))
    
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print(f"[OK] Displayed: Figure 30 — Extrusion Rate Variability (CV Improvement: {improvement:.1f}%)")


def figure_31_microprime_vs_retraction(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 31 — Micro-prime vs Retraction Magnitude Comparison
    Shows replacement strategy effectiveness
    """
    # Extract retraction magnitudes from baseline
    retraction_magnitudes = []
    for line in baseline_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        if stripped.startswith('G1') or stripped.startswith('G0'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None and parsed['E'] < 0:
                retraction_magnitudes.append(abs(parsed['E']))
    
    # Extract micro-prime magnitudes from stabilized
    microprime_magnitudes = []
    for line in stabilized_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        if 'micro-prime' in stripped.lower() or ('E' in stripped and 'micro' in stripped.lower()):
            # Try to extract E value
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None and parsed['E'] > 0:
                microprime_magnitudes.append(parsed['E'])
    
    # Also look for micro-primes in comments
    for line in stabilized_lines:
        if 'micro-prime' in line.lower():
            e_match = re.search(r'E([0-9.]+)', line)
            if e_match:
                microprime_magnitudes.append(float(e_match.group(1)))
    
    if len(retraction_magnitudes) == 0 or len(microprime_magnitudes) == 0:
        print("WARNING: Insufficient data for micro-prime vs retraction comparison")
        return
    
    # Match counts (use minimum)
    min_count = min(len(retraction_magnitudes), len(microprime_magnitudes))
    retraction_magnitudes = sorted(retraction_magnitudes, reverse=True)[:min_count]
    microprime_magnitudes = sorted(microprime_magnitudes, reverse=True)[:min_count]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left plot: Scatter comparison
    ax1.scatter(retraction_magnitudes, microprime_magnitudes, alpha=0.6, s=50,
               color=COLORS['stabilized'], edgecolor='black', linewidth=0.5, zorder=3)
    
    # Add diagonal line (y=x)
    max_val = max(max(retraction_magnitudes), max(microprime_magnitudes))
    ax1.plot([0, max_val], [0, max_val], 'k--', linewidth=2.0, alpha=0.5, label='y=x', zorder=1)
    
    # Add trend line
    if len(retraction_magnitudes) > 1:
        z = np.polyfit(retraction_magnitudes, microprime_magnitudes, 1)
        p = np.poly1d(z)
        ax1.plot(retraction_magnitudes, p(retraction_magnitudes), 'r-', linewidth=2.0,
                alpha=0.7, label=f'Trend (slope={z[0]:.2f})', zorder=2)
    
    ax1.set_xlabel('Retraction Magnitude |ΔE|', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Micro-prime Magnitude ΔE', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Micro-prime vs Retraction Magnitude', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=10, framealpha=0.98, edgecolor='black')
    ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_aspect('equal', adjustable='box')
    
    # Right plot: Distribution comparison
    ax2.hist(retraction_magnitudes, bins=20, alpha=0.6, label='Retractions', 
            color=COLORS['baseline'], edgecolor='black', linewidth=0.8)
    ax2.hist(microprime_magnitudes, bins=20, alpha=0.6, label='Micro-primes',
            color=COLORS['stabilized'], edgecolor='black', linewidth=0.8)
    ax2.set_xlabel('Magnitude |ΔE|', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_title('Magnitude Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.legend(fontsize=10, framealpha=0.98, edgecolor='black')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Calculate correlation
    if len(retraction_magnitudes) > 1:
        correlation = np.corrcoef(retraction_magnitudes, microprime_magnitudes)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=ax1.transAxes, fontsize=11, fontweight='bold', fontfamily='serif',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 31 — Micro-prime vs Retraction Magnitude Comparison")


def figure_32_cumulative_extrusion(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 32 — Cumulative Extrusion Comparison
    Shows total material usage and flow continuity
    """
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Compute cumulative extrusion
    def compute_cumulative_e(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        times = [0.0]
        cumulative_e = [0.0]
        e_cumulative = 0.0
        
        for line in gcode_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            
            if stripped.startswith('G1') or stripped.startswith('G0'):
                parsed = parse_gcode_line(line)
                if parsed['E'] is not None and parsed['E'] > 0:
                    e_cumulative += parsed['E']
                    # Estimate time (simplified)
                    if len(times) > 0:
                        times.append(times[-1] + 0.1)  # Approximate
                    else:
                        times.append(0.1)
                    cumulative_e.append(e_cumulative)
        
        return np.array(times), np.array(cumulative_e)
    
    t_base, e_base = compute_cumulative_e(baseline_lines)
    t_stab, e_stab = compute_cumulative_e(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor('white')
    
    if len(t_base) > 0 and len(e_base) > 0:
        ax.plot(t_base, e_base, color=COLORS['baseline'], linewidth=2.5, 
               label='Baseline', alpha=0.8, zorder=3)
    if len(t_stab) > 0 and len(e_stab) > 0:
        ax.plot(t_stab, e_stab, color=COLORS['stabilized'], linewidth=2.5,
               label='Stabilized', alpha=0.8, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Cumulative Extrusion E', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_title('Cumulative Extrusion Over Time', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.98, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    # Add final values
    if len(e_base) > 0 and len(e_stab) > 0:
        final_base = e_base[-1]
        final_stab = e_stab[-1]
        ax.text(0.98, 0.02, f'Baseline Total: {final_base:.1f}\nStabilized Total: {final_stab:.1f}',
               transform=ax.transAxes, fontsize=10, fontweight='bold', fontfamily='serif',
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 32 — Cumulative Extrusion Comparison")


def figure_33_resistance_vs_length(electrical_df: pd.DataFrame):
    """
    Fig. 33 — Resistance vs Print Length
    Shows if resistance scales linearly (good) or has issues (bad)
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("WARNING: No electrical data available for Figure 33")
        return
    
    if 'resistance_ohm' not in electrical_df.columns or 'length_mm' not in electrical_df.columns:
        print("ERROR: electrical_traces.csv must contain 'resistance_ohm' and 'length_mm' columns")
        return
    
    # Filter to successful traces only
    successful = electrical_df[(electrical_df['open_circuit'] == 0) & 
                               (electrical_df['resistance_ohm'].notna())].copy()
    
    if len(successful) == 0:
        print("WARNING: No successful electrical traces found")
        return
    
    conditions = ['baseline', 'partial', 'full']
    condition_labels = ['Baseline', 'Partial', 'Full']
    colors_map = {'baseline': COLORS['baseline'], 'partial': COLORS['partial'], 'full': COLORS['stabilized']}
    
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor('white')
    
    for cond, label, color in zip(conditions, condition_labels, [COLORS['baseline'], COLORS['partial'], COLORS['stabilized']]):
        cond_data = successful[successful['condition'] == cond]
        if len(cond_data) > 0:
            lengths = cond_data['length_mm'].values
            resistances = cond_data['resistance_ohm'].values
            
            ax.scatter(lengths, resistances, alpha=0.7, s=60, label=label, color=color,
                      edgecolor='black', linewidth=0.8, zorder=3)
            
            # Add trend line
            if len(lengths) > 1:
                z = np.polyfit(lengths, resistances, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(lengths.min(), lengths.max(), 100)
                ax.plot(x_trend, p(x_trend), '--', linewidth=2.0, alpha=0.6, color=color, zorder=2)
    
    ax.set_xlabel('Trace Length (mm)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_title('Resistance vs Print Length', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.legend(loc='best', fontsize=11, framealpha=0.98, edgecolor='black')
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 33 — Resistance vs Print Length")


def figure_34_stabilization_overhead(run_log_path: Path, stabilized_lines: List[str]):
    """
    Fig. 34 — Stabilization Overhead Analysis
    Breakdown of added time from different stabilization actions
    """
    if not run_log_path.exists():
        print(f"ERROR: run_log.csv not found at {run_log_path}")
        return
    
    df = pd.read_csv(run_log_path)
    
    # Count actions and estimate time
    actions = df['action'].values if 'action' in df.columns else []
    
    dwell_count = sum(1 for a in actions if 'dwell' in str(a).lower() or 'relax' in str(a).lower())
    prime_count = sum(1 for a in actions if 'prime' in str(a).lower())
    suppress_count = sum(1 for a in actions if 'suppress' in str(a).lower() or 'retract' in str(a).lower())
    
    # Estimate time per action (typical values)
    dwell_time_per = 0.35  # seconds
    prime_time_per = 0.1   # seconds
    suppress_time_per = 0.15  # seconds (micro-prime time)
    
    total_dwell_time = dwell_count * dwell_time_per
    total_prime_time = prime_count * prime_time_per
    total_suppress_time = suppress_count * suppress_time_per
    total_overhead = total_dwell_time + total_prime_time + total_suppress_time
    
    # Also count from G-code
    gcode_dwells = sum(1 for line in stabilized_lines if 'G4' in line)
    gcode_dwell_time = gcode_dwells * dwell_time_per
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left plot: Time breakdown
    categories = ['Dwells', 'Low Primes', 'Retraction\nSuppression']
    times = [total_dwell_time, total_prime_time, total_suppress_time]
    colors_overhead = [COLORS['action_dwell'], COLORS['action_low_prime'], COLORS['action_suppress']]
    
    bars = ax1.bar(categories, times, color=colors_overhead, alpha=0.8,
                  edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time Added (s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax1.set_title('Stabilization Overhead Breakdown', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        if time_val > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Right plot: Action counts
    action_counts = [dwell_count, prime_count, suppress_count]
    bars2 = ax2.bar(categories, action_counts, color=colors_overhead, alpha=0.8,
                   edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Action Count', fontsize=12, fontweight='bold', fontfamily='serif')
    ax2.set_title('Stabilization Actions Count', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    
    # Add value labels
    for bar, count in zip(bars2, action_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add total overhead annotation
    ax1.text(0.5, max(times) * 0.9, f'Total Overhead: {total_overhead:.1f}s',
            ha='center', fontsize=11, fontweight='bold', fontfamily='serif',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, edgecolor='black'))
    
    for ax in [ax1, ax2]:
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
    
    plt.tight_layout()
    plt.show(block=True)
    print(f"[OK] Displayed: Figure 34 — Stabilization Overhead Analysis (Total: {total_overhead:.1f}s)")


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate effectiveness figures highlighting stabilizer impact')
    parser.add_argument('--baseline-gcode', type=str, required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--run-log', type=str, default='results/run_log.csv',
                       help='Path to run_log.csv file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing CSV data files (default: code/input)')
    parser.add_argument('--figures', type=str, nargs='+', default=['all'],
                       choices=['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 'all'],
                       help='Which figures to generate (default: all)')
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
    
    stabilized_path = Path(args.stabilized_gcode)
    if not stabilized_path.is_absolute():
        test_path = script_dir / args.stabilized_gcode
        if test_path.exists():
            stabilized_path = test_path
    
    run_log_path = Path(args.run_log)
    if not run_log_path.is_absolute():
        test_path = script_dir / args.run_log
        if test_path.exists():
            run_log_path = test_path
    
    # Data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        input_dir = script_dir / 'input'
        if input_dir.exists():
            data_dir = input_dir
        else:
            data_dir = script_dir / 'data'
    
    # Read G-code files
    print(f"Reading baseline G-code: {baseline_path}", flush=True)
    with open(baseline_path, 'r') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_path}", flush=True)
    with open(stabilized_path, 'r') as f:
        stabilized_lines = f.readlines()
    
    # Read CSV data files
    print_trials_path = data_dir / 'print_trials.csv'
    electrical_path = data_dir / 'electrical_traces.csv'
    
    print_trials_df = None
    if print_trials_path.exists():
        print(f"Reading print trials data: {print_trials_path}")
        print_trials_df = pd.read_csv(print_trials_path)
    
    electrical_df = None
    if electrical_path.exists():
        print(f"Reading electrical traces data: {electrical_path}")
        electrical_df = pd.read_csv(electrical_path)
    
    # Determine which figures to generate
    if 'all' in args.figures:
        figures_to_generate = ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34']
    else:
        figures_to_generate = args.figures
    
    print(f"\nGenerating effectiveness figures: {', '.join(figures_to_generate)}", flush=True)
    print("Figures will be displayed interactively - save them manually using figure window controls.\n", flush=True)
    
    # Generate requested figures
    if '24' in figures_to_generate:
        print("Generating Figure 24 (Pressure Compliance Timeline)...", flush=True)
        figure_24_pressure_compliance_timeline(run_log_path, args.p_y, args.p_max)
    
    if '25' in figures_to_generate:
        print("Generating Figure 25 (Action Frequency Analysis)...", flush=True)
        figure_25_action_frequency(run_log_path)
    
    if '26' in figures_to_generate:
        print("Generating Figure 26 (Layer-by-Layer Analysis)...", flush=True)
        figure_26_layer_by_layer_analysis(baseline_lines, stabilized_lines, run_log_path)
    
    if '27' in figures_to_generate:
        print("Generating Figure 27 (Failure Mode Analysis)...", flush=True)
        figure_27_failure_mode_analysis(print_trials_df, electrical_df)
    
    if '28' in figures_to_generate:
        print("Generating Figure 28 (Feed Rate Scaling Distribution)...", flush=True)
        figure_28_feed_rate_scaling(run_log_path)
    
    if '29' in figures_to_generate:
        print("Generating Figure 29 (Print Time Impact)...", flush=True)
        figure_29_print_time_impact(baseline_lines, stabilized_lines)
    
    if '30' in figures_to_generate:
        print("Generating Figure 30 (Extrusion Rate Variability)...", flush=True)
        figure_30_extrusion_variability(baseline_lines, stabilized_lines)
    
    if '31' in figures_to_generate:
        print("Generating Figure 31 (Micro-prime vs Retraction Comparison)...", flush=True)
        figure_31_microprime_vs_retraction(baseline_lines, stabilized_lines)
    
    if '32' in figures_to_generate:
        print("Generating Figure 32 (Cumulative Extrusion Comparison)...", flush=True)
        figure_32_cumulative_extrusion(baseline_lines, stabilized_lines)
    
    if '33' in figures_to_generate:
        print("Generating Figure 33 (Resistance vs Print Length)...", flush=True)
        figure_33_resistance_vs_length(electrical_df)
    
    if '34' in figures_to_generate:
        print("Generating Figure 34 (Stabilization Overhead Analysis)...", flush=True)
        figure_34_stabilization_overhead(run_log_path, stabilized_lines)
    
    print(f"\n[OK] All requested effectiveness figures displayed.", flush=True)


if __name__ == '__main__':
    main()
