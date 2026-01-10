#!/usr/bin/env python3
"""
generate_effectiveness_figures.py

Generates additional figures highlighting stabilizer effectiveness for the paper.
Focuses on metrics that demonstrate the impact and value of the stabilization approach.

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
})

# Professional color palette
COLORS = {
    'baseline': '#d62728',      # Red
    'stabilized': '#2ca02c',    # Green
    'partial': '#ff7f0e',       # Orange
    'full': '#2ca02c',          # Green (same as stabilized)
    'in_bounds': '#90EE90',      # Light green
    'out_bounds': '#FFB6C1',     # Light red
    'low_prime': '#4169E1',      # Royal blue
    'relax_dwell': '#FF6347',    # Tomato
    'retract_suppressed': '#9370DB',  # Medium purple
}

# ============================================================================
# Utility Functions
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


def load_run_log(csv_path: Path) -> pd.DataFrame:
    """Load and parse run_log.csv."""
    if not csv_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        # Convert numeric columns
        if 'p_hat' in df.columns:
            df['p_hat'] = pd.to_numeric(df['p_hat'], errors='coerce')
        if 'feed_scale' in df.columns:
            df['feed_scale'] = pd.to_numeric(df['feed_scale'], errors='coerce')
        if 't_s' in df.columns:
            df['t_s'] = pd.to_numeric(df['t_s'], errors='coerce')
        return df
    except Exception as e:
        print(f"Error loading run_log.csv: {e}", flush=True)
        return pd.DataFrame()


# ============================================================================
# Figure 24: Pressure Window Compliance Over Time
# ============================================================================

def figure_24_pressure_compliance_timeline(csv_path: Path, p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 24 — Pressure Window Compliance Over Time
    Shows when pressure stays within admissible bounds (p_y to p_max)
    """
    df = load_run_log(csv_path)
    if len(df) == 0 or 'p_hat' not in df.columns:
        print("WARNING: No pressure data available for Figure 24", flush=True)
        return
    
    times = df['t_s'].values
    p_hat = df['p_hat'].values
    
    # Determine compliance
    in_bounds = (p_hat >= p_y) & (p_hat <= p_max)
    below_yield = p_hat < p_y
    above_max = p_hat > p_max
    
    compliance_pct = (in_bounds.sum() / len(p_hat)) * 100 if len(p_hat) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    fig.patch.set_facecolor('white')
    
    # Plot pressure trace
    ax.plot(times, p_hat, color='black', linewidth=1.5, alpha=0.7, label='p̂(t)', zorder=3)
    
    # Fill regions
    ax.fill_between(times, p_y, p_max, alpha=0.2, color=COLORS['in_bounds'], 
                   label=f'Admissible Window ({compliance_pct:.1f}% compliance)', zorder=1)
    ax.fill_between(times, 0, p_y, alpha=0.15, color=COLORS['out_bounds'], 
                   label='Below Yield', zorder=1)
    ax.fill_between(times, p_max, p_max + 5, alpha=0.15, color=COLORS['out_bounds'], 
                   label='Above Max', zorder=1)
    
    # Reference lines
    ax.axhline(p_y, color='blue', linestyle='--', linewidth=2.0, label=f'p_y = {p_y:.1f}', zorder=2)
    ax.axhline(p_max, color='red', linestyle='--', linewidth=2.0, label=f'p_max = {p_max:.1f}', zorder=2)
    
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Pressure p̂(t)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title(f'Pressure Window Compliance ({compliance_pct:.1f}% in bounds)', 
                fontsize=13, fontweight='bold', fontfamily='serif')
    
    # Bold tick labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    legend = ax.legend(loc='best', fontsize=11, framealpha=0.98, edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontfamily('serif')
    
    ax.grid(True, alpha=0.5, linestyle='--', linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 24 — Pressure Window Compliance Over Time", flush=True)


# ============================================================================
# Figure 25: Action Frequency Analysis
# ============================================================================

def figure_25_action_frequency(csv_path: Path):
    """
    Fig. 25 — Stabilizer Action Frequency Analysis
    Shows when and how often the stabilizer intervenes
    """
    df = load_run_log(csv_path)
    if len(df) == 0 or 'action' not in df.columns:
        print("WARNING: No action data available for Figure 25", flush=True)
        return
    
    # Count actions
    action_counts = df['action'].value_counts()
    
    # Filter to stabilization actions
    stabilization_actions = ['low_prime', 'relax_dwell', 'retract_suppressed', 'emit']
    action_data = {action: action_counts.get(action, 0) for action in stabilization_actions}
    
    # Create timeline of actions
    times = df['t_s'].values if 't_s' in df.columns else np.arange(len(df))
    actions = df['action'].values
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.0), sharex=True)
    fig.patch.set_facecolor('white')
    
    # Top: Bar chart of action counts
    actions_list = list(action_data.keys())
    counts_list = [action_data[a] for a in actions_list]
    colors_list = [COLORS.get(a, 'gray') for a in actions_list]
    
    bars = ax1.bar(actions_list, counts_list, color=colors_list, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Action Count', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_title('Stabilizer Action Frequency', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    for bar, count in zip(bars, counts_list):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Bottom: Timeline of actions
    action_colors_map = {
        'low_prime': COLORS['low_prime'],
        'relax_dwell': COLORS['relax_dwell'],
        'retract_suppressed': COLORS['retract_suppressed'],
        'emit': COLORS['stabilized']
    }
    
    for action_type in stabilization_actions:
        mask = actions == action_type
        if mask.sum() > 0:
            ax2.scatter(times[mask], np.ones(mask.sum()) * stabilization_actions.index(action_type),
                       color=action_colors_map.get(action_type, 'gray'), 
                       alpha=0.6, s=20, label=action_type.replace('_', ' ').title(), zorder=3)
    
    ax2.set_xlabel('Time (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Action Type', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_title('Action Timeline', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_yticks(range(len(stabilization_actions)))
    ax2.set_yticklabels([a.replace('_', ' ').title() for a in stabilization_actions])
    ax2.grid(True, alpha=0.5, linestyle='--', linewidth=1.0, zorder=0)
    
    # Bold labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 25 — Action Frequency Analysis", flush=True)


# ============================================================================
# Figure 26: Feed Rate Scaling Distribution
# ============================================================================

def figure_26_feed_scaling_distribution(csv_path: Path):
    """
    Fig. 26 — Feed Rate Scaling Distribution
    Shows how often feed rates are scaled and by how much
    """
    df = load_run_log(csv_path)
    if len(df) == 0 or 'feed_scale' not in df.columns:
        print("WARNING: No feed scaling data available for Figure 26", flush=True)
        return
    
    feed_scales = df['feed_scale'].dropna()
    feed_scales = feed_scales[feed_scales > 0]  # Remove invalid values
    
    if len(feed_scales) == 0:
        print("WARNING: No valid feed scaling data", flush=True)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left: Histogram
    ax1.hist(feed_scales, bins=30, color=COLORS['stabilized'], alpha=0.7, 
            edgecolor='black', linewidth=1.2)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2.0, label='No Scaling (1.0)')
    ax1.axvline(np.mean(feed_scales), color='blue', linestyle='--', linewidth=2.0, 
               label=f'Mean: {np.mean(feed_scales):.2f}')
    ax1.set_xlabel('Feed Scale Factor', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_title('Feed Rate Scaling Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Right: Statistics
    stats_text = f"""
    Total Moves: {len(feed_scales):,}
    Scaled Moves: {(feed_scales != 1.0).sum():,}
    Scaling Rate: {(feed_scales != 1.0).sum() / len(feed_scales) * 100:.1f}%
    
    Mean: {np.mean(feed_scales):.3f}
    Median: {np.median(feed_scales):.3f}
    Std Dev: {np.std(feed_scales):.3f}
    Min: {np.min(feed_scales):.3f}
    Max: {np.max(feed_scales):.3f}
    """
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=12,
            fontweight='bold', fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='black'))
    ax2.axis('off')
    
    # Bold labels
    ax1.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 26 — Feed Rate Scaling Distribution", flush=True)


# ============================================================================
# Figure 27: Layer-by-Layer Analysis
# ============================================================================

def figure_27_layer_analysis(baseline_lines: List[str], stabilized_lines: List[str], 
                             csv_path: Path, p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 27 — Layer-by-Layer Analysis
    Shows how stabilization metrics vary across layers
    """
    df = load_run_log(csv_path)
    
    # Extract layer information from G-code
    def extract_layers(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Z coordinates and corresponding move indices."""
        z_coords = []
        move_indices = []
        z_curr = 0.0
        move_idx = 0
        
        for line in gcode_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith(';'):
                continue
            
            if stripped.startswith('G1') or stripped.startswith('G0'):
                parsed = parse_gcode_line(line)
                if parsed['Z'] is not None:
                    z_curr = parsed['Z']
                z_coords.append(z_curr)
                move_indices.append(move_idx)
                move_idx += 1
        
        return np.array(z_coords), np.array(move_indices)
    
    z_baseline, _ = extract_layers(baseline_lines)
    z_stabilized, _ = extract_layers(stabilized_lines)
    
    # Group by layer (assuming layer height ~0.3mm)
    layer_height = 0.3
    layers_baseline = np.floor(z_baseline / layer_height).astype(int)
    layers_stabilized = np.floor(z_stabilized / layer_height).astype(int)
    
    # Count retractions per layer (baseline)
    retractions_per_layer = {}
    for i, line in enumerate(baseline_lines):
        if i < len(layers_baseline):
            layer = layers_baseline[i]
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None and parsed['E'] < -1e-6:
                retractions_per_layer[layer] = retractions_per_layer.get(layer, 0) + 1
    
    # Count actions per layer (stabilized)
    if len(df) > 0 and 'action' in df.columns:
        actions_per_layer = {}
        for idx, row in df.iterrows():
            if idx < len(layers_stabilized):
                layer = layers_stabilized[min(idx, len(layers_stabilized)-1)]
                action = row['action']
                if action in ['low_prime', 'relax_dwell', 'retract_suppressed']:
                    if layer not in actions_per_layer:
                        actions_per_layer[layer] = {}
                    actions_per_layer[layer][action] = actions_per_layer[layer].get(action, 0) + 1
    
    # Plot
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    fig.patch.set_facecolor('white')
    
    layers = sorted(set(list(retractions_per_layer.keys()) + list(actions_per_layer.keys())))
    
    if len(layers) > 0:
        retraction_counts = [retractions_per_layer.get(l, 0) for l in layers]
        action_counts = [sum(actions_per_layer.get(l, {}).values()) for l in layers]
        
        x = np.arange(len(layers))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, retraction_counts, width, label='Retractions (Baseline)',
                      color=COLORS['baseline'], alpha=0.8, edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, action_counts, width, label='Stabilizer Actions',
                      color=COLORS['stabilized'], alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xlabel('Layer Number', fontsize=13, fontweight='bold', fontfamily='serif')
        ax.set_ylabel('Count', fontsize=13, fontweight='bold', fontfamily='serif')
        ax.set_title('Layer-by-Layer Stabilization Analysis', fontsize=13, fontweight='bold', fontfamily='serif')
        ax.set_xticks(x)
        ax.set_xticklabels([f'L{l}' for l in layers])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 27 — Layer-by-Layer Analysis", flush=True)


# ============================================================================
# Figure 28: Print Time Impact
# ============================================================================

def figure_28_print_time_impact(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 28 — Print Time Impact
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
                
                if ds > 0 and f_curr > 0:
                    v = f_curr / 60.0  # mm/s
                    dt = ds / v
                    total_time += dt
                
                # Check for dwells (G4)
                if 'G4' in stripped:
                    match = re.search(r'S(\d+\.?\d*)', stripped)
                    if match:
                        total_time += float(match.group(1))
                
                x_prev, y_prev = x_curr, y_curr
                if f_curr > 0:
                    f_prev = f_curr
        
        return total_time
    
    time_baseline = compute_print_time(baseline_lines)
    time_stabilized = compute_print_time(stabilized_lines)
    time_increase = time_stabilized - time_baseline
    time_increase_pct = (time_increase / time_baseline * 100) if time_baseline > 0 else 0
    
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    fig.patch.set_facecolor('white')
    
    categories = ['Baseline', 'Stabilized']
    times = [time_baseline, time_stabilized]
    colors = [COLORS['baseline'], COLORS['stabilized']]
    
    bars = ax.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, time_val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               f'{time_val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add percentage increase annotation
    ax.text(0.5, 0.95, f'Time Increase: +{time_increase:.1f}s ({time_increase_pct:.1f}%)',
           transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, edgecolor='black'))
    
    ax.set_ylabel('Total Print Time (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Print Time Impact of Stabilization', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 28 — Print Time Impact", flush=True)


# ============================================================================
# Figure 29: Pressure Recovery Time
# ============================================================================

def figure_29_pressure_recovery(csv_path: Path, p_y: float = 5.0):
    """
    Fig. 29 — Pressure Recovery Time
    Time to recover pressure after low_prime interventions
    """
    df = load_run_log(csv_path)
    if len(df) == 0 or 'p_hat' not in df.columns or 'action' not in df.columns:
        print("WARNING: No pressure/action data available for Figure 29", flush=True)
        return
    
    times = df['t_s'].values
    p_hat = df['p_hat'].values
    actions = df['action'].values
    
    # Find low_prime events and measure recovery time
    recovery_times = []
    low_prime_indices = np.where(actions == 'low_prime')[0]
    
    for idx in low_prime_indices:
        if idx < len(p_hat) - 1:
            p_start = p_hat[idx]
            # Find when pressure recovers above p_y
            for i in range(idx + 1, min(idx + 100, len(p_hat))):  # Look ahead up to 100 steps
                if p_hat[i] >= p_y:
                    recovery_time = times[i] - times[idx]
                    recovery_times.append(recovery_time)
                    break
    
    if len(recovery_times) == 0:
        print("WARNING: No pressure recovery events found", flush=True)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left: Histogram
    ax1.hist(recovery_times, bins=20, color=COLORS['stabilized'], alpha=0.7,
            edgecolor='black', linewidth=1.2)
    ax1.axvline(np.mean(recovery_times), color='red', linestyle='--', linewidth=2.0,
               label=f'Mean: {np.mean(recovery_times):.2f}s')
    ax1.set_xlabel('Recovery Time (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_title('Pressure Recovery Time Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Right: Statistics
    stats_text = f"""
    Recovery Events: {len(recovery_times)}
    
    Mean: {np.mean(recovery_times):.3f} s
    Median: {np.median(recovery_times):.3f} s
    Std Dev: {np.std(recovery_times):.3f} s
    Min: {np.min(recovery_times):.3f} s
    Max: {np.max(recovery_times):.3f} s
    
    < 1s: {(np.array(recovery_times) < 1.0).sum()} ({100*(np.array(recovery_times) < 1.0).sum()/len(recovery_times):.1f}%)
    < 2s: {(np.array(recovery_times) < 2.0).sum()} ({100*(np.array(recovery_times) < 2.0).sum()/len(recovery_times):.1f}%)
    """
    
    ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=11,
            fontweight='bold', fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, edgecolor='black'))
    ax2.axis('off')
    
    # Bold labels
    ax1.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax1.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 29 — Pressure Recovery Time", flush=True)


# ============================================================================
# Figure 30: Extrusion Rate Variability (CV)
# ============================================================================

def figure_30_extrusion_variability(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 30 — Extrusion Rate Variability (Coefficient of Variation)
    Shows flow consistency improvement
    """
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Compute coefficient of variation
    cv_baseline = np.std(u_baseline) / np.mean(u_baseline) if np.mean(u_baseline) > 0 else 0
    cv_stabilized = np.std(u_stabilized) / np.mean(u_stabilized) if np.mean(u_stabilized) > 0 else 0
    improvement = ((cv_baseline - cv_stabilized) / cv_baseline * 100) if cv_baseline > 0 else 0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left: Boxplot comparison
    data = [u_baseline[:min(1000, len(u_baseline))], 
           u_stabilized[:min(1000, len(u_stabilized))]]
    bp = ax1.boxplot(data, labels=['Baseline', 'Stabilized'], patch_artist=True,
                     widths=0.6, showmeans=True)
    
    bp['boxes'][0].set_facecolor(COLORS['baseline'])
    bp['boxes'][1].set_facecolor(COLORS['stabilized'])
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)
    
    ax1.set_ylabel('Extrusion Rate u(t)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_title('Extrusion Rate Distribution', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Right: CV comparison
    categories = ['Baseline', 'Stabilized']
    cv_values = [cv_baseline, cv_stabilized]
    colors = [COLORS['baseline'], COLORS['stabilized']]
    
    bars = ax2.bar(categories, cv_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, cv_val in zip(bars, cv_values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{cv_val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.text(0.5, 0.95, f'Variability Reduction: {improvement:.1f}%',
            transform=ax2.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, edgecolor='black'))
    
    ax2.set_ylabel('Coefficient of Variation', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_title('Flow Consistency Improvement', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 30 — Extrusion Rate Variability", flush=True)


# ============================================================================
# Figure 31: Failure Mode Analysis
# ============================================================================

def figure_31_failure_modes(print_trials_df: pd.DataFrame, electrical_df: pd.DataFrame):
    """
    Fig. 31 — Failure Mode Analysis
    Breakdown of failure types by condition
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 31", flush=True)
        return
    
    conditions = ['baseline', 'partial', 'full']
    condition_labels = ['Baseline', 'Partial', 'Full']
    
    # Analyze failures
    failure_data = {}
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_data) > 0:
            total = len(cond_data)
            clogs = cond_data['clogs'].sum() if 'clogs' in cond_data.columns else 0
            incomplete = (cond_data['completed'] == 0).sum() if 'completed' in cond_data.columns else 0
            
            # Electrical failures
            elec_failures = 0
            if electrical_df is not None and len(electrical_df) > 0:
                elec_cond = electrical_df[electrical_df['condition'] == cond]
                if len(elec_cond) > 0:
                    elec_failures = (elec_cond['open_circuit'] == 1).sum()
            
            failure_data[cond] = {
                'Clogs': clogs,
                'Incomplete Prints': incomplete,
                'Electrical Failures': elec_failures,
                'Total Trials': total
            }
    
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor('white')
    
    x = np.arange(len(conditions))
    width = 0.25
    
    failure_types = ['Clogs', 'Incomplete Prints', 'Electrical Failures']
    colors_failures = ['#FF6B6B', '#4ECDC4', '#FFE66D']
    
    bottom = np.zeros(len(conditions))
    for i, (fail_type, color) in enumerate(zip(failure_types, colors_failures)):
        values = [failure_data.get(cond, {}).get(fail_type, 0) for cond in conditions]
        bars = ax.bar(x, values, width, bottom=bottom, label=fail_type,
                     color=color, alpha=0.8, edgecolor='black', linewidth=1.2)
        bottom += values
    
    ax.set_xlabel('Condition', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Failure Count', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Failure Mode Analysis by Condition', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 31 — Failure Mode Analysis", flush=True)


# ============================================================================
# Figure 32: Micro-prime vs Retraction Magnitude
# ============================================================================

def figure_32_microprime_comparison(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 32 — Micro-prime vs Retraction Magnitude Comparison
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
            if parsed['E'] is not None and parsed['E'] < -1e-6:
                retraction_magnitudes.append(abs(parsed['E']))
    
    # Extract micro-prime magnitudes from stabilized
    microprime_magnitudes = []
    for line in stabilized_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        if stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            # Look for micro-primes (small positive E after comments about micro-prime)
            if parsed['E'] is not None and parsed['E'] > 1e-6 and parsed['E'] < 2.0:
                # Check if this is likely a micro-prime
                if 'micro-prime' in line.lower() or (parsed['E'] > 0.4 and parsed['E'] < 1.0):
                    microprime_magnitudes.append(parsed['E'])
    
    if len(retraction_magnitudes) == 0 or len(microprime_magnitudes) == 0:
        print("WARNING: Insufficient data for micro-prime comparison", flush=True)
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 4.0))
    fig.patch.set_facecolor('white')
    
    # Left: Scatter plot
    # Match counts for comparison
    min_len = min(len(retraction_magnitudes), len(microprime_magnitudes))
    ret_mags = np.array(retraction_magnitudes[:min_len])
    mp_mags = np.array(microprime_magnitudes[:min_len])
    
    ax1.scatter(ret_mags, mp_mags, alpha=0.6, s=50, color=COLORS['stabilized'],
               edgecolors='black', linewidths=0.5)
    
    # Add diagonal line (perfect replacement)
    max_val = max(np.max(ret_mags), np.max(mp_mags))
    ax1.plot([0, max_val], [0, max_val], 'r--', linewidth=2.0, label='Perfect Replacement')
    
    # Add trend line
    if len(ret_mags) > 1:
        z = np.polyfit(ret_mags, mp_mags, 1)
        p = np.poly1d(z)
        ax1.plot(ret_mags, p(ret_mags), 'b-', linewidth=2.0, alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    ax1.set_xlabel('Retraction Magnitude |ΔE| (Baseline)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_ylabel('Micro-prime Magnitude ΔE (Stabilized)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.set_title('Micro-prime Replacement Strategy', fontsize=13, fontweight='bold', fontfamily='serif')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Right: Histogram comparison
    ax2.hist(ret_mags, bins=20, alpha=0.6, label='Retractions', color=COLORS['baseline'],
            edgecolor='black', linewidth=1.2)
    ax2.hist(mp_mags, bins=20, alpha=0.6, label='Micro-primes', color=COLORS['stabilized'],
            edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Magnitude |ΔE|', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_ylabel('Frequency', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.set_title('Magnitude Distribution Comparison', fontsize=13, fontweight='bold', fontfamily='serif')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 32 — Micro-prime vs Retraction Magnitude", flush=True)


# ============================================================================
# Figure 33: Cumulative Extrusion Comparison
# ============================================================================

def figure_33_cumulative_extrusion(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 33 — Cumulative Extrusion Comparison
    Shows total material usage and flow continuity
    """
    def compute_cumulative_extrusion(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute cumulative E over time."""
        times = [0.0]
        cumulative_e = [0.0]
        e_cumulative = 0.0
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
                
                if de > 0:  # Positive extrusion
                    e_cumulative += de
                
                if ds > 0 and f_curr > 0:
                    v = f_curr / 60.0
                    dt = ds / v
                    t_curr += dt
                    times.append(t_curr)
                    cumulative_e.append(e_cumulative)
                
                x_prev, y_prev, e_prev = x_curr, y_curr, e_curr
                if f_curr > 0:
                    f_prev = f_curr
        
        return np.array(times), np.array(cumulative_e)
    
    t_baseline, e_baseline = compute_cumulative_extrusion(baseline_lines)
    t_stabilized, e_stabilized = compute_cumulative_extrusion(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    fig.patch.set_facecolor('white')
    
    ax.plot(t_baseline, e_baseline, color=COLORS['baseline'], linewidth=2.5,
           label='Baseline', alpha=0.8, zorder=3)
    ax.plot(t_stabilized, e_stabilized, color=COLORS['stabilized'], linewidth=2.5,
           label='Stabilized', alpha=0.8, zorder=3)
    
    ax.set_xlabel('Time (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Cumulative Extrusion E', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Cumulative Extrusion Comparison', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.5, linestyle='--', linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 33 — Cumulative Extrusion Comparison", flush=True)


# ============================================================================
# Figure 34: Resistance vs Print Length
# ============================================================================

def figure_34_resistance_vs_length(electrical_df: pd.DataFrame):
    """
    Fig. 34 — Resistance vs Print Length
    Shows if resistance scales linearly (good) or has issues
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("WARNING: No electrical data available for Figure 34", flush=True)
        return
    
    # Filter to successful traces only
    successful = electrical_df[(electrical_df['open_circuit'] == 0) & 
                               (electrical_df['resistance_ohm'].notna())].copy()
    
    if len(successful) == 0:
        print("WARNING: No successful electrical traces", flush=True)
        return
    
    conditions = ['baseline', 'partial', 'full']
    condition_labels = ['Baseline', 'Partial', 'Full']
    colors_map = {'baseline': COLORS['baseline'], 'partial': COLORS['partial'], 'full': COLORS['stabilized']}
    
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    fig.patch.set_facecolor('white')
    
    for cond, label, color in zip(conditions, condition_labels, colors_map.values()):
        cond_data = successful[successful['condition'] == cond]
        if len(cond_data) > 0:
            lengths = cond_data['length_mm'].values
            resistances = cond_data['resistance_ohm'].values
            
            ax.scatter(lengths, resistances, color=color, alpha=0.7, s=60,
                      edgecolors='black', linewidths=0.5, label=label, zorder=3)
            
            # Add trend line
            if len(lengths) > 1:
                z = np.polyfit(lengths, resistances, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(lengths.min(), lengths.max(), 100)
                ax.plot(x_trend, p(x_trend), '--', color=color, linewidth=2.0, alpha=0.7,
                       label=f'{label} trend: {z[0]:.1f}Ω/mm')
    
    ax.set_xlabel('Trace Length (mm)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_ylabel('Resistance (Ω)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Resistance vs Print Length', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.5, linestyle='--', linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 34 — Resistance vs Print Length", flush=True)


# ============================================================================
# Figure 35: Stabilization Overhead Analysis
# ============================================================================

def figure_35_overhead_analysis(baseline_lines: List[str], stabilized_lines: List[str], csv_path: Path):
    """
    Fig. 35 — Stabilization Overhead Analysis
    Breakdown of added time from dwells, micro-primes, feed scaling
    """
    df = load_run_log(csv_path)
    
    # Count actions and estimate time
    if len(df) > 0 and 'action' in df.columns:
        action_counts = df['action'].value_counts()
        dwell_time = action_counts.get('relax_dwell', 0) * 0.3  # Assume 0.3s per dwell
        low_prime_time = action_counts.get('low_prime', 0) * 0.1  # Assume 0.1s per prime
        retract_suppressed_time = action_counts.get('retract_suppressed', 0) * 0.45  # Dwell + micro-prime
    else:
        dwell_time = low_prime_time = retract_suppressed_time = 0
    
    # Estimate feed scaling overhead (minimal, but count scaled moves)
    if len(df) > 0 and 'feed_scale' in df.columns:
        feed_scaled_moves = (df['feed_scale'] != 1.0).sum()
        feed_scaling_overhead = feed_scaled_moves * 0.01  # Minimal overhead per scaled move
    else:
        feed_scaling_overhead = 0
    
    total_overhead = dwell_time + low_prime_time + retract_suppressed_time + feed_scaling_overhead
    
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    fig.patch.set_facecolor('white')
    
    categories = ['Dwells', 'Low Primes', 'Retraction\nSuppression', 'Feed Scaling']
    times = [dwell_time, low_prime_time, retract_suppressed_time, feed_scaling_overhead]
    colors = [COLORS['relax_dwell'], COLORS['low_prime'], COLORS['retract_suppressed'], '#87CEEB']
    
    bars = ax.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, time_val in zip(bars, times):
        if time_val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{time_val:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.text(0.5, 0.95, f'Total Overhead: {total_overhead:.2f}s',
           transform=ax.transAxes, ha='center', va='top', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3, edgecolor='black'))
    
    ax.set_ylabel('Time Overhead (s)', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.set_title('Stabilization Overhead Breakdown', fontsize=13, fontweight='bold', fontfamily='serif')
    ax.grid(axis='y', alpha=0.5, linestyle='--', linewidth=1.0)
    
    # Bold labels
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show(block=True)
    print("[OK] Displayed: Figure 35 — Stabilization Overhead Analysis", flush=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate effectiveness figures for paste extrusion paper')
    parser.add_argument('--baseline-gcode', type=str, required=True,
                       help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                       help='Path to stabilized G-code file')
    parser.add_argument('--run-log', type=str, default='results/run_log.csv',
                       help='Path to run_log.csv file')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory containing CSV data files')
    parser.add_argument('--figures', type=str, nargs='+', default=['all'],
                       choices=['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', 'all'],
                       help='Which figures to generate')
    parser.add_argument('--p-y', type=float, default=5.0,
                       help='Yield pressure threshold')
    parser.add_argument('--p-max', type=float, default=14.0,
                       help='Maximum pressure bound')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    baseline_path = Path(args.baseline_gcode)
    if not baseline_path.is_absolute():
        baseline_path = script_dir / args.baseline_gcode
    
    stabilized_path = Path(args.stabilized_gcode)
    if not stabilized_path.is_absolute():
        stabilized_path = script_dir / args.stabilized_gcode
    
    run_log_path = Path(args.run_log)
    if not run_log_path.is_absolute():
        run_log_path = script_dir / args.run_log
    
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
    
    # Read CSV data
    print_trials_df = None
    print_trials_path = data_dir / 'print_trials.csv'
    if print_trials_path.exists():
        print_trials_df = pd.read_csv(print_trials_path)
    
    electrical_df = None
    electrical_path = data_dir / 'electrical_traces.csv'
    if electrical_path.exists():
        electrical_df = pd.read_csv(electrical_path)
    
    # Determine which figures to generate
    if 'all' in args.figures:
        figures_to_generate = ['24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35']
    else:
        figures_to_generate = args.figures
    
    print(f"\nGenerating effectiveness figures: {', '.join(figures_to_generate)}", flush=True)
    print("Figures will be displayed interactively - save them manually using figure window controls.\n", flush=True)
    
    # Generate figures
    if '24' in figures_to_generate:
        print("Generating Figure 24 (Pressure Window Compliance)...", flush=True)
        figure_24_pressure_compliance_timeline(run_log_path, args.p_y, args.p_max)
    
    if '25' in figures_to_generate:
        print("Generating Figure 25 (Action Frequency Analysis)...", flush=True)
        figure_25_action_frequency(run_log_path)
    
    if '26' in figures_to_generate:
        print("Generating Figure 26 (Feed Rate Scaling Distribution)...", flush=True)
        figure_26_feed_scaling_distribution(run_log_path)
    
    if '27' in figures_to_generate:
        print("Generating Figure 27 (Layer-by-Layer Analysis)...", flush=True)
        figure_27_layer_analysis(baseline_lines, stabilized_lines, run_log_path, args.p_y, args.p_max)
    
    if '28' in figures_to_generate:
        print("Generating Figure 28 (Print Time Impact)...", flush=True)
        figure_28_print_time_impact(baseline_lines, stabilized_lines)
    
    if '29' in figures_to_generate:
        print("Generating Figure 29 (Pressure Recovery Time)...", flush=True)
        figure_29_pressure_recovery(run_log_path, args.p_y)
    
    if '30' in figures_to_generate:
        print("Generating Figure 30 (Extrusion Rate Variability)...", flush=True)
        figure_30_extrusion_variability(baseline_lines, stabilized_lines)
    
    if '31' in figures_to_generate:
        print("Generating Figure 31 (Failure Mode Analysis)...", flush=True)
        figure_31_failure_modes(print_trials_df, electrical_df)
    
    if '32' in figures_to_generate:
        print("Generating Figure 32 (Micro-prime vs Retraction Magnitude)...", flush=True)
        figure_32_microprime_comparison(baseline_lines, stabilized_lines)
    
    if '33' in figures_to_generate:
        print("Generating Figure 33 (Cumulative Extrusion Comparison)...", flush=True)
        figure_33_cumulative_extrusion(baseline_lines, stabilized_lines)
    
    if '34' in figures_to_generate:
        print("Generating Figure 34 (Resistance vs Print Length)...", flush=True)
        figure_34_resistance_vs_length(electrical_df)
    
    if '35' in figures_to_generate:
        print("Generating Figure 35 (Stabilization Overhead Analysis)...", flush=True)
        figure_35_overhead_analysis(baseline_lines, stabilized_lines, run_log_path)
    
    print(f"\n[OK] All requested effectiveness figures displayed.", flush=True)


if __name__ == '__main__':
    main()
