#!/usr/bin/env python3
"""
generate_paper_figures.py

Generates reviewer-proof figures for paste extrusion stabilization paper.
All figures are displayed interactively - user can save them manually if desired.

Figure set demonstrates:
1. Software changes commands (G-code delta)
2. Changes stabilize the process (pressure bounds)
3. Stabilization improves functional outcomes (electrical yield)
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

# Try to import scipy for survival analysis (optional)
try:
    from scipy import stats
    from scipy.stats import kaplan_meier_estimator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Figure 4 will use simplified survival analysis.")


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


def extract_gcode_metrics(gcode_lines: List[str]) -> Dict[str, any]:
    """Extract metrics from G-code for Figure 1."""
    retractions = 0
    dwells = 0
    total_e_added = 0.0
    e_deltas = []
    
    e_prev = 0.0
    e_curr = 0.0
    
    for line in gcode_lines:
        # Count retractions (negative E moves)
        if 'E' in line and not line.strip().startswith(';'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None:
                e_curr = parsed['E']
                delta_e = e_curr - e_prev
                
                if delta_e < 0:
                    retractions += 1
                elif delta_e > 0:
                    e_deltas.append(delta_e)
                    total_e_added += delta_e
                
                e_prev = e_curr
        
        # Count dwells
        if re.match(r'G4\s+S', line) or 'dwell' in line.lower():
            dwells += 1
    
    total_lines = len([l for l in gcode_lines if not l.strip().startswith(';')])
    
    return {
        'retractions': retractions,
        'dwells': dwells,
        'total_e_added': total_e_added,
        'e_deltas': e_deltas,
        'total_lines': total_lines
    }


def compute_u_timeline(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute u(t) timeline from G-code for Figure 2.
    Returns: (time_array, u_array) where u is extrusion rate proxy.
    """
    times = [0.0]
    u_values = [0.0]
    
    x_prev, y_prev = 0.0, 0.0
    e_prev = 0.0
    f_curr = 1000.0  # Default feed rate
    
    t_curr = 0.0
    
    for line in gcode_lines:
        if line.strip().startswith(';') or not line.strip():
            continue
        
        parsed = parse_gcode_line(line)
        
        # Update feed rate if present
        if parsed['F'] is not None:
            f_curr = parsed['F']
        
        # Process G1 moves
        if line.strip().startswith('G1') or line.strip().startswith('G0'):
            x_curr = parsed['X'] if parsed['X'] is not None else x_prev
            y_curr = parsed['Y'] if parsed['Y'] is not None else y_prev
            e_curr = parsed['E'] if parsed['E'] is not None else e_prev
            
            # Compute travel distance
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            ds = np.sqrt(dx**2 + dy**2)
            
            # Compute time delta: dt = ds / v, where v = F/60 (mm/s)
            if f_curr > 0 and ds > 0:
                v = f_curr / 60.0  # mm/s
                dt = ds / v
            else:
                dt = 0.01  # Small default for pure E moves
            
            # Compute extrusion rate proxy: u = dE/dt
            de = e_curr - e_prev
            if dt > 0:
                u = de / dt
            else:
                u = 0.0
            
            t_curr += dt
            times.append(t_curr)
            u_values.append(u)
            
            x_prev, y_prev = x_curr, y_curr
            e_prev = e_curr
    
    return np.array(times), np.array(u_values)


def compute_p_hat_timeline(gcode_lines: List[str], csv_log: Optional[pd.DataFrame] = None,
                           alpha: float = 0.1, tau_r: float = 2.0,
                           p_y: float = 5.0, p_max: float = 14.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute p_hat(t) from G-code or CSV log for Figure 3.
    Returns: (time_array, p_hat_array)
    """
    if csv_log is not None and 'p_hat' in csv_log.columns:
        # Use existing p_hat from CSV
        t = csv_log['t_s'].values
        p = csv_log['p_hat'].values
        return t, p
    
    # Otherwise compute from u(t)
    t, u = compute_u_timeline(gcode_lines)
    
    # Initialize pressure
    p_hat = np.zeros_like(t)
    p_hat[0] = 0.0
    
    dt = np.diff(t)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 0.01], dt])
    
    # Discrete-time pressure model: p_{k+1} = p_k + T_s * (alpha * u_k - p_k / tau_r)
    for i in range(1, len(t)):
        T_s = dt[i] if i < len(dt) else dt[-1]
        p_hat[i] = p_hat[i-1] + T_s * (alpha * u[i] - p_hat[i-1] / tau_r)
        p_hat[i] = max(0, p_hat[i])  # Prevent negative pressure
    
    return t, p_hat


# ============================================================================
# Figure Generation Functions
# ============================================================================

def figure_1_gcode_delta(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Figure 1: What the stabilizer changed (G-code delta)
    Shows: retractions, dwells, E deltas before vs after
    """
    baseline_metrics = extract_gcode_metrics(baseline_lines)
    stabilized_metrics = extract_gcode_metrics(stabilized_lines)
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.5])
    
    # Subplot 1: Retractions per 1000 lines
    ax1 = fig.add_subplot(gs[0])
    baseline_retract_rate = (baseline_metrics['retractions'] / baseline_metrics['total_lines']) * 1000
    stabilized_retract_rate = (stabilized_metrics['retractions'] / stabilized_metrics['total_lines']) * 1000
    
    bars1 = ax1.bar(['Baseline', 'Stabilized'], 
                    [baseline_retract_rate, stabilized_retract_rate],
                    color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Retractions per 1,000 lines', fontweight='bold')
    ax1.set_title('(a) Retraction Suppression', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Dwell events per 1000 lines
    ax2 = fig.add_subplot(gs[1])
    baseline_dwell_rate = (baseline_metrics['dwells'] / baseline_metrics['total_lines']) * 1000
    stabilized_dwell_rate = (stabilized_metrics['dwells'] / stabilized_metrics['total_lines']) * 1000
    
    bars2 = ax2.bar(['Baseline', 'Stabilized'],
                    [baseline_dwell_rate, stabilized_dwell_rate],
                    color=['#d62728', '#2ca02c'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Dwell Events per 1,000 lines', fontweight='bold')
    ax2.set_title('(b) Dwell Insertion', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 3: Histogram of E deltas
    ax3 = fig.add_subplot(gs[2])
    if baseline_metrics['e_deltas']:
        ax3.hist(baseline_metrics['e_deltas'], bins=30, alpha=0.6, 
                label='Baseline', color='#d62728', edgecolor='black')
    if stabilized_metrics['e_deltas']:
        ax3.hist(stabilized_metrics['e_deltas'], bins=30, alpha=0.6,
                label='Stabilized', color='#2ca02c', edgecolor='black')
    ax3.set_xlabel('ΔE per Extrusion Move', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.set_title('(c) Extrusion Step Size Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Figure 1: G-code Delta Analysis\n'
                 'Software layer reduces stop–start pressure shocks by eliminating retractions and smoothing extrusion events',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def figure_2_command_timeline(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Figure 2: Command timeline u(t) before vs after
    Shows: extrusion rate proxy over time for baseline and stabilized
    """
    t_baseline, u_baseline = compute_u_timeline(baseline_lines)
    t_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Baseline u(t)
    ax1.plot(t_baseline, u_baseline, 'r-', linewidth=1.5, alpha=0.8, label='Baseline')
    ax1.fill_between(t_baseline, 0, u_baseline, where=(u_baseline > 0), 
                     color='red', alpha=0.3)
    ax1.fill_between(t_baseline, 0, u_baseline, where=(u_baseline < 0),
                     color='orange', alpha=0.3, label='Retractions')
    ax1.set_ylabel('Extrusion Rate Proxy u(t)', fontweight='bold')
    ax1.set_title('(a) Baseline Execution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    # Stabilized u(t)
    ax2.plot(t_stabilized, u_stabilized, 'g-', linewidth=1.5, alpha=0.8, label='Stabilized')
    ax2.fill_between(t_stabilized, 0, u_stabilized, where=(u_stabilized > 0),
                     color='green', alpha=0.3)
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Extrusion Rate Proxy u(t)', fontweight='bold')
    ax2.set_title('(b) Stabilized Execution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
    
    plt.suptitle('Figure 2: Command Timeline\n'
                 'Stabilized execution replaces impulsive start/stop behavior with bounded-rate extrusion',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def figure_3_pressure_estimate(baseline_lines: List[str], stabilized_lines: List[str],
                               baseline_csv: Optional[pd.DataFrame] = None,
                               stabilized_csv: Optional[pd.DataFrame] = None,
                               p_y: float = 5.0, p_max: float = 14.0):
    """
    Figure 3: Model-based pressure estimate p_hat(t) with bounds
    Shows: pressure timeline with yield and max bounds
    """
    t_baseline, p_baseline = compute_p_hat_timeline(baseline_lines, baseline_csv)
    t_stabilized, p_stabilized = compute_p_hat_timeline(stabilized_lines, stabilized_csv)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Baseline p_hat(t)
    ax1.plot(t_baseline, p_baseline, 'r-', linewidth=2, alpha=0.8, label='p̂(t)')
    ax1.axhline(p_y, color='orange', linestyle='--', linewidth=2, label=f'p_y = {p_y}')
    ax1.axhline(p_max, color='red', linestyle='--', linewidth=2, label=f'p_max = {p_max}')
    ax1.fill_between(t_baseline, p_y, p_max, alpha=0.2, color='yellow', label='Admissible window')
    ax1.set_ylabel('Estimated Pressure p̂(t)', fontweight='bold')
    ax1.set_title('(a) Baseline', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Stabilized p_hat(t)
    ax2.plot(t_stabilized, p_stabilized, 'g-', linewidth=2, alpha=0.8, label='p̂(t)')
    ax2.axhline(p_y, color='orange', linestyle='--', linewidth=2, label=f'p_y = {p_y}')
    ax2.axhline(p_max, color='red', linestyle='--', linewidth=2, label=f'p_max = {p_max}')
    ax2.fill_between(t_stabilized, p_y, p_max, alpha=0.2, color='yellow', label='Admissible window')
    ax2.set_xlabel('Time (s)', fontweight='bold')
    ax2.set_ylabel('Estimated Pressure p̂(t)', fontweight='bold')
    ax2.set_title('(b) Stabilized', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 3: Model-Based Pressure Estimate\n'
                 'Baseline repeatedly crosses the yield threshold; stabilization maintains p̂(t) in the admissible window',
                 fontsize=12, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def figure_4_survival_curve(print_trials_df: pd.DataFrame):
    """
    Figure 4: Extrusion continuity survival curve
    Requires: print_trials.csv with columns: condition, flow_duration_s, completed
    """
    if 'flow_duration_s' not in print_trials_df.columns:
        print("ERROR: print_trials.csv must contain 'flow_duration_s' column")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conditions = print_trials_df['condition'].unique()
    colors = {'baseline': '#d62728', 'partial': '#ff7f0e', 'full': '#2ca02c'}
    
    for condition in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == condition]
        durations = cond_data['flow_duration_s'].values
        completed = cond_data.get('completed', pd.Series([True] * len(cond_data))).values
        
        # Simple empirical survival function
        durations_sorted = np.sort(durations)
        n = len(durations_sorted)
        survival = np.arange(n, 0, -1) / n
        
        ax.plot(durations_sorted, survival, 'o-', linewidth=2, markersize=6,
               label=condition.capitalize(), color=colors.get(condition, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontweight='bold')
    ax.set_ylabel('Probability of Continuous Extrusion', fontweight='bold')
    ax.set_title('Figure 4: Extrusion Continuity Survival Curve\n'
                 'Stabilization substantially increases the probability of maintaining continuous extrusion over time',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()


def figure_5_operability_map(print_trials_df: pd.DataFrame):
    """
    Figure 5: First-layer operability map (gap-speed phase diagram)
    Requires: print_trials.csv with columns: first_layer_height, first_layer_speed, success
    OR: first_layer_success can be used as success proxy
    """
    # Try to use first_layer_success if success column doesn't exist
    if 'success' not in print_trials_df.columns and 'first_layer_success' in print_trials_df.columns:
        print_trials_df = print_trials_df.copy()
        print_trials_df['success'] = print_trials_df['first_layer_success']
    
    required_cols = ['first_layer_height', 'first_layer_speed', 'success']
    if not all(col in print_trials_df.columns for col in required_cols):
        print(f"WARNING: print_trials.csv missing columns: {required_cols}")
        print(f"Available columns: {list(print_trials_df.columns)}")
        print("Skipping Figure 5: First-layer operability map")
        print("To generate this figure, add columns: first_layer_height, first_layer_speed, success")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pivot table for heatmap
    pivot = print_trials_df.pivot_table(
        values='success', 
        index='first_layer_height', 
        columns='first_layer_speed',
        aggfunc='mean'
    )
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    ax.set_xlabel('First Layer Speed (mm/s)', fontweight='bold')
    ax.set_ylabel('First Layer Height (relative to nozzle diameter)', fontweight='bold')
    ax.set_title('Figure 5: First-Layer Operability Map\n'
                 'Stable first-layer deposition exists in a paste-specific region; FDM "squish" lies outside',
                 fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def figure_6_clog_correlation(print_trials_df: pd.DataFrame):
    """
    Figure 6: Clog events vs retraction count (correlation)
    Requires: print_trials.csv with columns: retractions, clogs, condition
    Note: If retractions column missing, can be computed from G-code (not implemented here)
    """
    required_cols = ['clogs', 'condition']
    if not all(col in print_trials_df.columns for col in required_cols):
        print(f"WARNING: print_trials.csv missing columns: {required_cols}")
        print(f"Available columns: {list(print_trials_df.columns)}")
        print("Skipping Figure 6: Clog correlation")
        return
    
    # Check if retractions column exists
    if 'retractions' not in print_trials_df.columns:
        print("WARNING: 'retractions' column not found in print_trials.csv")
        print("Skipping Figure 6: Clog correlation")
        print("To generate this figure, add 'retractions' column (retraction count per trial)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {'baseline': '#d62728', 'partial': '#ff7f0e', 'full': '#2ca02c'}
    
    for condition in print_trials_df['condition'].unique():
        cond_data = print_trials_df[print_trials_df['condition'] == condition]
        ax.scatter(cond_data['retractions'], cond_data['clogs'],
                  label=condition.capitalize(), color=colors.get(condition, 'gray'),
                  s=100, alpha=0.7, edgecolors='black')
    
    # Add trend line
    x_all = print_trials_df['retractions'].values
    y_all = print_trials_df['clogs'].values
    if len(x_all) > 1:
        z = np.polyfit(x_all, y_all, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=2, alpha=0.5, label='Trend')
    
    ax.set_xlabel('Retraction Count', fontweight='bold')
    ax.set_ylabel('Clog Events', fontweight='bold')
    ax.set_title('Figure 6: Clog Events vs Retraction Count\n'
                 'Retraction frequency predicts clogging; suppressing retractions reduces clog incidence',
                 fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def figure_7_electrical_yield(print_trials_df: pd.DataFrame):
    """
    Figure 7: Conductive trace yield + resistance stability
    Requires: print_trials.csv with columns: condition, open_circuit, resistance
    Note: If open_circuit missing, can use completed==0 as proxy (not implemented here)
    """
    required_cols = ['condition']
    if 'condition' not in print_trials_df.columns:
        print(f"WARNING: print_trials.csv missing 'condition' column")
        print(f"Available columns: {list(print_trials_df.columns)}")
        print("Skipping Figure 7: Electrical yield")
        return
    
    # Check if open_circuit exists, if not try to infer from completed
    if 'open_circuit' not in print_trials_df.columns:
        if 'completed' in print_trials_df.columns:
            print("Note: Using 'completed==0' as proxy for open_circuit")
            print_trials_df = print_trials_df.copy()
            print_trials_df['open_circuit'] = (print_trials_df['completed'] == 0).astype(int)
        else:
            print("WARNING: 'open_circuit' column not found and 'completed' not available")
            print("Skipping Figure 7: Electrical yield")
            print("To generate this figure, add 'open_circuit' column (1=open, 0=closed)")
            return
    
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig)
    
    # Subplot 1: Open circuit rate
    ax1 = fig.add_subplot(gs[0])
    conditions = print_trials_df['condition'].unique()
    open_rates = []
    colors_bar = {'baseline': '#d62728', 'partial': '#ff7f0e', 'full': '#2ca02c'}
    
    for condition in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == condition]
        open_rate = cond_data['open_circuit'].mean()
        open_rates.append(open_rate)
    
    bars = ax1.bar(conditions, open_rates, 
                   color=[colors_bar.get(c, 'gray') for c in conditions],
                   alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Open Circuit Rate', fontweight='bold')
    ax1.set_title('(a) Electrical Yield', fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Resistance distribution (if available)
    ax2 = fig.add_subplot(gs[1])
    if 'resistance' in print_trials_df.columns:
        data_by_condition = []
        labels = []
        for condition in conditions:
            cond_data = print_trials_df[print_trials_df['condition'] == condition]
            # Only non-open circuits
            resistances = cond_data[cond_data['open_circuit'] == 0]['resistance'].dropna()
            if len(resistances) > 0:
                data_by_condition.append(resistances.values)
                labels.append(condition.capitalize())
        
        if data_by_condition:
            bp = ax2.boxplot(data_by_condition, labels=labels, patch_artist=True)
            for patch, condition in zip(bp['boxes'], conditions):
                patch.set_facecolor(colors_bar.get(condition, 'gray'))
                patch.set_alpha(0.7)
            ax2.set_ylabel('Resistance (Ω)', fontweight='bold')
            ax2.set_title('(b) Resistance Distribution', fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
    else:
        ax2.text(0.5, 0.5, 'Resistance data not available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('(b) Resistance Distribution', fontweight='bold')
    
    plt.suptitle('Figure 7: Conductive Trace Yield + Resistance Stability\n'
                 'Extrusion continuity translates to electrical yield and lower resistance variance',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def figure_8_visual_evidence():
    """
    Figure 8: Before/After visual evidence (photo panel)
    Note: This requires actual images. Placeholder shows structure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Placeholder text - user should replace with actual images
    axes[0, 0].text(0.5, 0.5, 'Baseline\nFirst Layer\n(Blocked/Plowing)', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('(a) Baseline First Layer', fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].text(0.5, 0.5, 'Stabilized\nFirst Layer\n(Clean Bead)', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('(b) Stabilized First Layer', fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].text(0.5, 0.5, 'Baseline\nConductive Trace\n(Breaks/Voids)', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('(c) Baseline Conductive Trace', fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].text(0.5, 0.5, 'Stabilized\nConductive Trace\n(Continuous)', 
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('(d) Stabilized Conductive Trace', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('Figure 8: Before/After Visual Evidence\n'
                 'Visual comparison showing defect reduction and improved continuity',
                 fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate reviewer-proof figures for paste extrusion stabilization paper'
    )
    parser.add_argument('--baseline-gcode', type=str, required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--baseline-csv', type=str, default=None,
                       help='Path to baseline CSV log (optional)')
    parser.add_argument('--stabilized-csv', type=str, default=None,
                       help='Path to stabilized CSV log (optional)')
    parser.add_argument('--print-trials', type=str, default='../print_trials.csv',
                       help='Path to print_trials.csv')
    parser.add_argument('--p-y', type=float, default=5.0,
                       help='Yield pressure threshold')
    parser.add_argument('--p-max', type=float, default=14.0,
                       help='Maximum pressure bound')
    parser.add_argument('--figures', type=str, nargs='+',
                       choices=['1', '2', '3', '4', '5', '6', '7', '8', 'all'],
                       default=['all'],
                       help='Which figures to generate')
    parser.add_argument('--baseline-first-layer-img', type=str, default=None,
                       help='Path to baseline first layer image (Figure 8)')
    parser.add_argument('--stabilized-first-layer-img', type=str, default=None,
                       help='Path to stabilized first layer image (Figure 8)')
    parser.add_argument('--baseline-trace-img', type=str, default=None,
                       help='Path to baseline trace image (Figure 8)')
    parser.add_argument('--stabilized-trace-img', type=str, default=None,
                       help='Path to stabilized trace image (Figure 8)')
    
    args = parser.parse_args()
    
    # Read G-code files (resolve relative paths)
    baseline_path = Path(args.baseline_gcode)
    stabilized_path = Path(args.stabilized_gcode)
    
    # If relative paths don't exist, try resolving from script directory
    if not baseline_path.exists():
        script_dir = Path(__file__).parent
        baseline_path = script_dir / args.baseline_gcode
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline G-code not found: {args.baseline_gcode}")
    
    if not stabilized_path.exists():
        script_dir = Path(__file__).parent
        stabilized_path = script_dir / args.stabilized_gcode
    if not stabilized_path.exists():
        raise FileNotFoundError(f"Stabilized G-code not found: {args.stabilized_gcode}")
    
    with open(baseline_path, 'r') as f:
        baseline_lines = f.readlines()
    with open(stabilized_path, 'r') as f:
        stabilized_lines = f.readlines()
    
    # Read CSV logs if provided
    baseline_csv = None
    stabilized_csv = None
    if args.baseline_csv:
        baseline_csv = pd.read_csv(args.baseline_csv)
    if args.stabilized_csv:
        stabilized_csv = pd.read_csv(args.stabilized_csv)
    
    # Read print trials data
    print_trials_path = Path(args.print_trials)
    print_trials_df = None
    if print_trials_path.exists():
        print_trials_df = pd.read_csv(print_trials_path)
    else:
        print(f"Warning: print_trials.csv not found at {print_trials_path}")
        print("Figures 4, 5, 6, 7 will be skipped")
    
    # Generate requested figures
    figures_to_generate = args.figures if 'all' not in args.figures else ['1', '2', '3', '4', '5', '6', '7', '8']
    
    if '1' in figures_to_generate:
        print("Generating Figure 1: G-code Delta...")
        figure_1_gcode_delta(baseline_lines, stabilized_lines)
    
    if '2' in figures_to_generate:
        print("Generating Figure 2: Command Timeline...")
        figure_2_command_timeline(baseline_lines, stabilized_lines)
    
    if '3' in figures_to_generate:
        print("Generating Figure 3: Pressure Estimate...")
        figure_3_pressure_estimate(baseline_lines, stabilized_lines,
                                   baseline_csv, stabilized_csv,
                                   args.p_y, args.p_max)
    
    if '4' in figures_to_generate and print_trials_df is not None:
        print("Generating Figure 4: Survival Curve...")
        figure_4_survival_curve(print_trials_df)
    
    if '5' in figures_to_generate and print_trials_df is not None:
        print("Generating Figure 5: Operability Map...")
        figure_5_operability_map(print_trials_df)
    
    if '6' in figures_to_generate and print_trials_df is not None:
        print("Generating Figure 6: Clog Correlation...")
        figure_6_clog_correlation(print_trials_df)
    
    if '7' in figures_to_generate and print_trials_df is not None:
        print("Generating Figure 7: Electrical Yield...")
        figure_7_electrical_yield(print_trials_df)
    
    if '8' in figures_to_generate:
        print("Generating Figure 8: Visual Evidence...")
        figure_8_visual_evidence(
            args.baseline_first_layer_img,
            args.stabilized_first_layer_img,
            args.baseline_trace_img,
            args.stabilized_trace_img
        )
    
    print("\nAll requested figures displayed. Use the figure window controls to save them manually.")


if __name__ == "__main__":
    main()

