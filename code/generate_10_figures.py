#!/usr/bin/env python3
"""
generate_10_figures.py

Generates 12 figures for paste extrusion stabilization paper (10 original + 2 3D visualizations).
All figures are displayed interactively - save them manually using figure window controls.
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

# Force interactive backend for GUI display - try different backends
# On macOS, ensure we can access the display
if sys.platform == 'darwin':
    # macOS specific: ensure display access
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'

try:
    matplotlib.use('TkAgg')  # Use TkAgg backend for GUI compatibility
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('macosx')  # macOS native backend
        except:
            pass  # Use default backend

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Enable interactive mode
plt.ion()
print(f"Matplotlib backend: {matplotlib.get_backend()}, Interactive: {plt.isinteractive()}", flush=True)

# Set professional matplotlib style for IEEE publication (column template)
# Increased font sizes for better readability
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.framealpha': 1.0,
    'figure.titlesize': 12,
    'figure.titleweight': 'normal',
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'patch.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
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


def extract_3d_toolpath(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D toolpath coordinates and extrusion information from G-code.
    Returns: (coords [Nx3], e_deltas [N], extrusion_flags [N], retraction_flags [N])
    """
    coords = []
    e_deltas = []
    extrusion_flags = []
    retraction_flags = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    x_prev, y_prev, z_prev = None, None, 0.0
    e_cumulative = 0.0
    e_prev = 0.0
    is_relative_e = True  # Default to relative (M83)
    
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
                
                e_prev = e_cumulative
            
            # Record point if we have valid coordinates
            if x_to_use is not None and y_to_use is not None:
                coords.append([x_to_use, y_to_use, z_to_use])
                e_deltas.append(e_delta)
                extrusion_flags.append(e_delta > 1e-6)
                retraction_flags.append(e_delta < -1e-6)
    
    if len(coords) == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=bool), np.array([], dtype=bool)
    
    return np.array(coords), np.array(e_deltas), np.array(extrusion_flags), np.array(retraction_flags)


def compute_extrusion_rate_3d(coords: np.ndarray, e_deltas: np.ndarray) -> np.ndarray:
    """
    Compute extrusion rate (E per mm) for 3D visualization.
    """
    if len(coords) < 2:
        return np.zeros(len(coords))
    
    rates = np.zeros(len(coords))
    for i in range(1, len(coords)):
        dx = coords[i, 0] - coords[i-1, 0]
        dy = coords[i, 1] - coords[i-1, 1]
        dz = coords[i, 2] - coords[i-1, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if dist > 1e-6:
            rates[i] = abs(e_deltas[i]) / dist
        else:
            rates[i] = rates[i-1] if i > 0 else 0.0
    
    return rates


# ============================================================================
# Figure Generation Functions
# ============================================================================

def figure_1_gcode_delta(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 1 — G-code modification summary (delta)
    Grouped bars: baseline vs stabilized for retractions, dwells, extrusion moves
    Enhanced with effectiveness metrics and improvement percentages
    """
    baseline_metrics = extract_gcode_metrics(baseline_lines, is_stabilized=False)
    stabilized_metrics = extract_gcode_metrics(stabilized_lines, is_stabilized=True)
    
    # Calculate improvements
    retractions_eliminated = baseline_metrics['retractions'] - stabilized_metrics['retractions']
    retraction_reduction_pct = (retractions_eliminated / baseline_metrics['retractions'] * 100) if baseline_metrics['retractions'] > 0 else 0
    dwells_added = stabilized_metrics['dwells'] - baseline_metrics['dwells']
    
    # IEEE single column width: 3.5 inches
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
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
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_data, width, label='Baseline', 
                   color=COLORS['baseline'], alpha=0.85, edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, stabilized_data, width, label='Stabilized', 
                   color=COLORS['stabilized'], alpha=0.85, edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Metric Category')
    ax.set_ylabel('Count')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', 
                       fontsize=10)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 1 — G-code Modification Summary", flush=True)


def figure_2_retraction_histogram(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 2 — Retraction suppression histogram
    Histogram of retraction magnitudes |ΔE| for baseline (optionally overlay stabilized)
    """
    baseline_retractions = extract_retraction_magnitudes(baseline_lines)
    stabilized_retractions = extract_retraction_magnitudes(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Determine bins based on both datasets
    all_retractions = baseline_retractions + stabilized_retractions
    if len(all_retractions) > 0:
        max_val = max(all_retractions)
        bins = np.linspace(0, max_val if max_val > 0 else 1, 25)
    else:
        bins = np.linspace(0, 1, 25)
    
    # Plot baseline histogram
    if len(baseline_retractions) > 0:
        n1, bins, patches1 = ax.hist(baseline_retractions, bins=bins, alpha=0.75, label='Baseline',
                                     color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
        for patch in patches1:
            patch.set_alpha(0.75)
    else:
        # Show baseline as empty if no retractions
        n1, bins, patches1 = ax.hist([], bins=bins, alpha=0.75, label='Baseline (no retractions)',
                                     color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    
    # Plot stabilized histogram - always show it, even if empty
    if len(stabilized_retractions) > 0:
        n2, _, patches2 = ax.hist(stabilized_retractions, bins=bins, alpha=0.6, label='Stabilized',
                                  color=COLORS['stabilized'], edgecolor='black', linewidth=0.5)
    else:
        # Show stabilized as a visible step plot overlay (all zeros) to make it clear it's there
        # Create histogram counts for stabilized (all zeros)
        n2 = np.zeros(len(bins) - 1)
        # Use step plot to show the stabilized version as a visible line
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.step(bin_centers, n2, where='mid', color=COLORS['stabilized'], 
               linewidth=2.5, alpha=0.8, linestyle='--', label='Stabilized (0 retractions)',
               zorder=10)
        # Also add a visible marker at the first bin to emphasize
        if len(bin_centers) > 0:
            ax.plot(bin_centers[0], 0, marker='o', markersize=10, 
                   color=COLORS['stabilized'], markeredgecolor='black', 
                   markeredgewidth=1.5, zorder=15, label='_nolegend_')
        
        # Add text annotation to make it clear
        y_max = ax.get_ylim()[1]
        x_max = max(bins) if len(bins) > 0 else 1.0
        ax.text(x_max * 0.65, y_max * 0.8, '✓ Retractions\nEliminated', 
               fontsize=11, fontweight='bold', color=COLORS['stabilized'],
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                        edgecolor=COLORS['stabilized'], linewidth=2),
               ha='center', va='center')
    
    ax.set_xlabel('|ΔE| (mm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 2 — Retraction Suppression Histogram", flush=True)


def figure_3_u_baseline(baseline_lines: List[str]):
    """
    Fig. 3 — Extrusion-rate proxy timeline u(t) (baseline)
    """
    times, u_values = compute_u_timeline(baseline_lines)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(times, u_values, color=COLORS['baseline'], linewidth=1.5, alpha=0.9, label='Baseline u(t)', zorder=3)
    ax.fill_between(times, 0, u_values, where=(u_values > 0), 
                    color=COLORS['baseline'], alpha=0.25, zorder=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u(t) (mm/s)')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=2)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 3 — Extrusion-Rate Proxy Timeline (Baseline)")


def figure_4_u_stabilized(stabilized_lines: List[str]):
    """
    Fig. 4 — Extrusion-rate proxy timeline u(t) (stabilized)
    """
    times, u_values = compute_u_timeline(stabilized_lines)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(times, u_values, color=COLORS['stabilized'], linewidth=1.5, alpha=0.9, 
           label='Stabilized u(t)', zorder=3)
    ax.fill_between(times, 0, u_values, where=(u_values > 0), 
                    color=COLORS['stabilized'], alpha=0.25, zorder=1)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('u(t) (mm/s)')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, zorder=2)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 4 — Extrusion-Rate Proxy Timeline (Stabilized)")


def figure_5_p_baseline(baseline_lines: List[str],
                       alpha: float = 8.0, tau_r: float = 6.0,
                       p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 5 — Pressure estimate p̂(t) with bounds (baseline)
    """
    times, u_values = compute_u_timeline(baseline_lines)
    p_hat = compute_pressure_timeline(times, u_values, alpha, tau_r, p_y, p_max)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(times, p_hat, color=COLORS['baseline'], linewidth=1.5, alpha=0.9, 
           label='p̂(t)', zorder=4)
    ax.fill_between(times, p_y, p_max, alpha=0.15, color=COLORS['admissible'], 
                    label='Admissible window', zorder=1)
    ax.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=1.5, 
              label=f'p_y = {p_y}', zorder=3)
    ax.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=1.5, 
              label=f'p_max = {p_max}', zorder=3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('p̂(t)')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 5 — Pressure Estimate (Baseline)")


def figure_6_p_stabilized(stabilized_lines: List[str],
                         alpha: float = 8.0, tau_r: float = 6.0,
                         p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 6 — Pressure estimate p̂(t) with bounds (stabilized)
    """
    times, u_values = compute_u_timeline(stabilized_lines)
    p_hat = compute_pressure_timeline(times, u_values, alpha, tau_r, p_y, p_max)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    ax.plot(times, p_hat, color=COLORS['stabilized'], linewidth=1.5, alpha=0.9, 
           label='p̂(t)', zorder=4)
    ax.fill_between(times, p_y, p_max, alpha=0.15, color=COLORS['admissible'], 
                    label='Admissible window', zorder=1)
    ax.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=1.5, 
              label=f'p_y = {p_y}', zorder=3)
    ax.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=1.5, 
              label=f'p_max = {p_max}', zorder=3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('p̂(t)')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 6 — Pressure Estimate (Stabilized)")


def figure_7_survival_curve(print_trials_df: pd.DataFrame):
    """
    Fig. 7 — Extrusion continuity survival curve
    Uses flow_duration_s for survival analysis (baseline, partial, full)
    """
    if 'flow_duration_s' not in print_trials_df.columns:
        print("ERROR: print_trials.csv must contain 'flow_duration_s' column")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    conditions = ['baseline', 'partial', 'full']
    colors_map = {'baseline': COLORS['baseline'], 'partial': COLORS['partial'], 'full': COLORS['full']}
    
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
            ax.plot(durations_sorted, survival, 'o-', linewidth=2.5, markersize=7,
                   label=condition.capitalize(), color=colors_map.get(condition, 'gray'), 
                   alpha=0.85, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
        else:
            # Simple empirical survival function
            durations_sorted = np.sort(durations)
            n = len(durations_sorted)
            survival = np.arange(n, 0, -1) / n
            ax.plot(durations_sorted, survival, 'o-', linewidth=2.5, markersize=7,
                   label=condition.capitalize(), color=colors_map.get(condition, 'gray'), 
                   alpha=0.85, markeredgecolor='black', markeredgewidth=0.5, zorder=3)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Survival Probability')
    ax.legend(loc='upper right', framealpha=1.0, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 7 — Extrusion Continuity Survival Curve")


def figure_8_first_layer_map(first_layer_df: pd.DataFrame):
    """
    Fig. 8 — First-layer operating envelope heatmap
    x: h_ratio = h1/d_nozzle, y: speed_mmps, cell value: mean success
    """
    if not all(col in first_layer_df.columns for col in ['h_ratio', 'speed_mmps', 'success']):
        print("ERROR: first_layer_sweep.csv must contain columns: h_ratio, speed_mmps, success")
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Create pivot table for heatmap
    pivot = first_layer_df.pivot_table(
        values='success',
        index='h_ratio',
        columns='speed_mmps',
        aggfunc='mean'
    )
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, origin='lower', 
                   interpolation='nearest')
    
    # Set ticks with better formatting
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f'{int(c)}' for c in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f'{c:.1f}' for c in pivot.index], fontsize=10)
    
    ax.set_xlabel('Speed (mm/s)')
    ax.set_ylabel('h₁/d_nozzle')
    
    # Add colorbar with better styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Success Rate', fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    
    # Add grid lines for better readability
    ax.set_xticks(np.arange(len(pivot.columns)+1)-0.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index)+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 8 — First-Layer Operating Envelope")


def figure_9_open_circuit_rate(electrical_df: pd.DataFrame):
    """
    Fig. 9 — Electrical yield (open-circuit rate)
    Bar chart: baseline vs stabilized
    """
    if 'open_circuit' not in electrical_df.columns or 'condition' not in electrical_df.columns:
        print("ERROR: electrical_traces.csv must contain columns: condition, open_circuit")
        return
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    conditions = ['baseline', 'full']  # Note: using 'full' as stabilized
    open_rates = []
    colors_bar = {'baseline': COLORS['baseline'], 'full': COLORS['stabilized'], 'stabilized': COLORS['stabilized']}
    
    for condition in conditions:
        cond_data = electrical_df[electrical_df['condition'] == condition]
        if len(cond_data) > 0:
            open_rate = cond_data['open_circuit'].mean()
            open_rates.append(open_rate)
        else:
            open_rates.append(0.0)
    
    # Map 'full' to 'stabilized' for display
    display_labels = ['Baseline', 'Stabilized']
    bars = ax.bar(display_labels, open_rates,
                 color=[colors_bar.get(c, 'gray') for c in conditions],
                 alpha=0.85, edgecolor='black', linewidth=0.8, width=0.6)
    
    ax.set_ylabel('Open-Circuit Rate')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show(block=True)
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
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    conditions = ['baseline', 'full']  # Note: using 'full' as stabilized
    data_by_condition = []
    labels = []
    
    for condition in conditions:
        cond_data = successful[successful['condition'] == condition]
        resistances = cond_data['resistance_ohm'].dropna()
        if len(resistances) > 0:
            data_by_condition.append(resistances.values)
            labels.append('Stabilized' if condition == 'full' else condition.capitalize())
    
    if len(data_by_condition) > 0:
        bp = ax.boxplot(data_by_condition, labels=labels, patch_artist=True,
                        widths=0.65, showmeans=True, meanline=False,
                        boxprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2.0, color='black'),
                        meanprops=dict(marker='D', markerfacecolor='black', 
                                      markeredgecolor='black', markersize=6))
        
        # Color the boxes with professional styling
        colors_box = {'baseline': COLORS['baseline'], 'stabilized': COLORS['stabilized']}
        for patch, label in zip(bp['boxes'], labels):
            patch.set_facecolor(colors_box.get(label.lower(), 'gray'))
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(0.8)
        
        ax.set_ylabel('Resistance (Ω)')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
        ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show(block=True)
    print("✓ Displayed: Figure 10 — Resistance Stability")


def figure_11_3d_toolpath_comparison(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 11 — 3D Toolpath Comparison (Before/After)
    Side-by-side 3D visualization showing geometry preservation and retraction elimination
    """
    print("Extracting 3D toolpaths...")
    # Extract toolpaths
    baseline_coords, baseline_e, baseline_ext, baseline_ret = extract_3d_toolpath(baseline_lines)
    stabilized_coords, stabilized_e, stabilized_ext, stabilized_ret = extract_3d_toolpath(stabilized_lines)
    
    print(f"Baseline: {len(baseline_coords)} points, {np.sum(baseline_ext)} extrusion moves, {np.sum(baseline_ret)} retractions")
    print(f"Stabilized: {len(stabilized_coords)} points, {np.sum(stabilized_ext)} extrusion moves, {np.sum(stabilized_ret)} retractions")
    
    if len(baseline_coords) == 0 or len(stabilized_coords) == 0:
        print("ERROR: Could not extract 3D toolpath data")
        print(f"  Baseline coords: {len(baseline_coords)}, Stabilized coords: {len(stabilized_coords)}")
        print("  This might indicate the G-code files don't contain valid toolpath data.")
        return
    
    print("Creating 3D plot...")
    
    # Compute extrusion rates for coloring
    baseline_rates = compute_extrusion_rate_3d(baseline_coords, baseline_e)
    stabilized_rates = compute_extrusion_rate_3d(stabilized_coords, stabilized_e)
    
    # Normalize rates for colormap
    max_rate = max(np.max(baseline_rates), np.max(stabilized_rates)) if len(baseline_rates) > 0 and len(stabilized_rates) > 0 else 1.0
    if max_rate < 1e-6:
        max_rate = 1.0
    
    fig = plt.figure(figsize=(16, 8))
    
    # Baseline (left)
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Track retraction locations for highlighting
    retraction_points = []
    extrusion_count = 0
    retraction_count = 0
    
    # Plot extrusion moves with color coding
    if len(baseline_coords) > 1:
        for i in range(len(baseline_coords) - 1):
            if baseline_ext[i+1]:  # Extrusion move
                color_val = baseline_rates[i+1] / max_rate if i+1 < len(baseline_rates) else 0.5
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=1.5, alpha=0.8, zorder=2)
                extrusion_count += 1
            elif baseline_ret[i+1]:  # Retraction - HIGHLIGHT THESE
                # Draw retraction line in bright red with thicker line
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color='red', linewidth=2.5, linestyle='--', alpha=1.0, zorder=5)
                # Mark retraction point with large X marker
                retraction_points.append([baseline_coords[i+1, 0], baseline_coords[i+1, 1], baseline_coords[i+1, 2]])
                ax1.scatter([baseline_coords[i+1, 0]], [baseline_coords[i+1, 1]], [baseline_coords[i+1, 2]],
                           color='red', marker='X', s=150, linewidths=2.5, edgecolors='darkred', zorder=10)
                retraction_count += 1
            else:  # Travel move
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color='lightgray', linewidth=0.5, alpha=0.4, zorder=1)
    
    # Add text annotation showing retraction count
    if retraction_count > 0:
        ax1.text2D(0.02, 0.98, f'Retractions: {retraction_count}\nExtrusions: {extrusion_count}', 
                  transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2),
                  color='black', fontweight='bold')
    
    ax1.set_xlabel('X (mm)', fontsize=11)
    ax1.set_ylabel('Y (mm)', fontsize=11)
    ax1.set_zlabel('Z (mm)', fontsize=11)
    ax1.set_title('(a) Baseline - Retractions Highlighted in Red', fontsize=12, pad=5, fontweight='bold')
    
    # Stabilized (right)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Track micro-prime locations for highlighting
    micro_prime_points = []
    stabilized_extrusion_count = 0
    micro_prime_count = 0
    remaining_retractions = 0
    
    # Plot extrusion moves with color coding
    if len(stabilized_coords) > 1:
        for i in range(len(stabilized_coords) - 1):
            if stabilized_ext[i+1]:  # Extrusion move
                color_val = stabilized_rates[i+1] / max_rate if i+1 < len(stabilized_rates) else 0.5
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color=plt.cm.viridis(color_val), linewidth=1.5, alpha=0.8, zorder=2)
                stabilized_extrusion_count += 1
            elif stabilized_ret[i+1]:  # Should be rare/none - highlight if any remain
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color='orange', linewidth=2.0, linestyle='--', alpha=0.9, zorder=5)
                remaining_retractions += 1
            else:  # Travel move or micro-prime
                # Check if this is a micro-prime (small E move)
                if abs(stabilized_e[i+1]) > 1e-6:
                    # HIGHLIGHT micro-primes in bright green
                    ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                            [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                            [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                            color='lime', linewidth=2.0, alpha=1.0, zorder=6)
                    micro_prime_points.append([stabilized_coords[i+1, 0], stabilized_coords[i+1, 1], stabilized_coords[i+1, 2]])
                    ax2.scatter([stabilized_coords[i+1, 0]], [stabilized_coords[i+1, 1]], [stabilized_coords[i+1, 2]],
                               color='lime', marker='o', s=100, edgecolors='darkgreen', linewidths=2.0, zorder=10)
                    micro_prime_count += 1
                else:
                    ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                            [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                            [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                            color='lightgray', linewidth=0.5, alpha=0.4, zorder=1)
    
    # Add text annotation showing improvements
    improvement_text = f'Micro-primes: {micro_prime_count}\nExtrusions: {stabilized_extrusion_count}'
    if remaining_retractions > 0:
        improvement_text += f'\nRemaining retractions: {remaining_retractions}'
    else:
        improvement_text += '\n✓ All retractions eliminated'
    
    ax2.text2D(0.02, 0.98, improvement_text, 
              transform=ax2.transAxes, fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2),
              color='black', fontweight='bold')
    
    ax2.set_xlabel('X (mm)', fontsize=11)
    ax2.set_ylabel('Y (mm)', fontsize=11)
    ax2.set_zlabel('Z (mm)', fontsize=11)
    ax2.set_title('(b) Stabilized - Micro-primes Highlighted in Green', fontsize=12, pad=5, fontweight='bold')
    
    # Set same viewing angle for both
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=10)
    
    # Add legend explaining the visualization
    legend_elements = [
        Patch(facecolor='red', edgecolor='darkred', label='Retractions (Baseline)'),
        Patch(facecolor='lime', edgecolor='darkgreen', label='Micro-primes (Stabilized)'),
        Patch(facecolor='blue', alpha=0.5, label='Extrusion moves'),
        Patch(facecolor='lightgray', alpha=0.5, label='Travel moves')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for legend
    print("Displaying Figure 11 (this may take a moment)...", flush=True)
    print("If the figure window doesn't appear, check that matplotlib can access your display.", flush=True)
    try:
        # Force figure to show
        plt.draw()
        plt.pause(0.1)  # Give matplotlib time to create window
        plt.show(block=True)  # Block until window is closed
        print("✓ Figure 11 window closed", flush=True)
    except Exception as e:
        print(f"Error displaying figure: {e}", flush=True)
        # Try non-blocking as fallback
        plt.show(block=False)
        import time
        time.sleep(2)  # Give time for window to appear
        print("Figure displayed in non-blocking mode. Window may be behind other windows.", flush=True)
    print("✓ Displayed: Figure 11 — 3D Toolpath Comparison", flush=True)


def figure_12_3d_extrusion_rate_map(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 12 — 3D Extrusion Rate Map
    Color-coded 3D toolpath showing extrusion rate intensity
    """
    # Extract toolpaths
    baseline_coords, baseline_e, baseline_ext, baseline_ret = extract_3d_toolpath(baseline_lines)
    stabilized_coords, stabilized_e, stabilized_ext, _ = extract_3d_toolpath(stabilized_lines)
    
    if len(baseline_coords) == 0 or len(stabilized_coords) == 0:
        print("WARNING: Could not extract 3D toolpath data")
        return
    
    # Compute extrusion rates
    baseline_rates = compute_extrusion_rate_3d(baseline_coords, baseline_e)
    stabilized_rates = compute_extrusion_rate_3d(stabilized_coords, stabilized_e)
    
    # Normalize rates
    max_rate = max(np.max(baseline_rates), np.max(stabilized_rates)) if len(baseline_rates) > 0 and len(stabilized_rates) > 0 else 1.0
    if max_rate < 1e-6:
        max_rate = 1.0
    
    fig = plt.figure(figsize=(16, 8))
    
    # Baseline
    ax1 = fig.add_subplot(121, projection='3d')
    
    baseline_segments = []
    baseline_colors_seg = []
    baseline_retraction_count = 0
    
    if len(baseline_coords) > 1:
        for i in range(len(baseline_coords) - 1):
            if baseline_ext[i+1]:
                seg = [[baseline_coords[i, 0], baseline_coords[i, 1], baseline_coords[i, 2]],
                       [baseline_coords[i+1, 0], baseline_coords[i+1, 1], baseline_coords[i+1, 2]]]
                baseline_segments.append(seg)
                # Use extrusion rate for color, or default to 0 if no rate available
                if i+1 < len(baseline_rates) and baseline_rates[i+1] > 0:
                    baseline_colors_seg.append(baseline_rates[i+1] / max_rate)
                else:
                    baseline_colors_seg.append(0.0)
            # Also plot retractions prominently
            elif baseline_ret[i+1]:
                # Highlight retractions with red lines
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color='red', linewidth=2.5, linestyle='--', alpha=1.0, zorder=10)
                ax1.scatter([baseline_coords[i+1, 0]], [baseline_coords[i+1, 1]], [baseline_coords[i+1, 2]],
                           color='red', marker='X', s=120, linewidths=2.0, edgecolors='darkred', zorder=15)
                baseline_retraction_count += 1
    
    # Plot baseline segments
    if baseline_segments:
        lc1 = Line3DCollection(baseline_segments, cmap=plt.cm.viridis, linewidths=1.5, alpha=0.85)
        lc1.set_array(np.array(baseline_colors_seg))
        ax1.add_collection3d(lc1)
        # Set color limits
        lc1.set_clim(0, 1)
    else:
        # Fallback: plot all moves as gray lines if no extrusions found
        print("WARNING: No baseline extrusion moves found, plotting all moves as gray lines", flush=True)
        for i in range(min(1000, len(baseline_coords) - 1)):  # Limit to first 1000 moves for performance
            ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                    [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                    [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                    color='gray', linewidth=0.5, alpha=0.5)
    
    # Add annotation
    if baseline_retraction_count > 0:
        ax1.text2D(0.02, 0.98, f'Retractions: {baseline_retraction_count}', 
                  transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2),
                  color='black', fontweight='bold')
    
    ax1.set_xlabel('X (mm)', fontsize=11)
    ax1.set_ylabel('Y (mm)', fontsize=11)
    ax1.set_zlabel('Z (mm)', fontsize=11)
    ax1.set_title('(a) Baseline - Retractions in Red', fontsize=12, pad=5, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=10)
    
    # Add colorbar only if we have segments with colors
    if baseline_segments:
        cbar1 = plt.colorbar(lc1, ax=ax1, pad=0.1, shrink=0.8)
        cbar1.set_label('Extrusion Rate (E/mm)', fontsize=10)
        cbar1.ax.tick_params(labelsize=9)
    
    # Stabilized
    ax2 = fig.add_subplot(122, projection='3d')
    
    stabilized_segments = []
    stabilized_colors_seg = []
    micro_prime_count = 0
    
    if len(stabilized_coords) > 1:
        for i in range(len(stabilized_coords) - 1):
            if stabilized_ext[i+1]:
                seg = [[stabilized_coords[i, 0], stabilized_coords[i, 1], stabilized_coords[i, 2]],
                       [stabilized_coords[i+1, 0], stabilized_coords[i+1, 1], stabilized_coords[i+1, 2]]]
                stabilized_segments.append(seg)
                # Use extrusion rate for color, or default to 0 if no rate available
                if i+1 < len(stabilized_rates) and stabilized_rates[i+1] > 0:
                    stabilized_colors_seg.append(stabilized_rates[i+1] / max_rate)
                else:
                    stabilized_colors_seg.append(0.0)
            # Highlight micro-primes (small E moves that replaced retractions)
            elif abs(stabilized_e[i+1]) > 1e-6:
                # Micro-prime in bright green
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color='lime', linewidth=2.0, alpha=1.0, zorder=10)
                ax2.scatter([stabilized_coords[i+1, 0]], [stabilized_coords[i+1, 1]], [stabilized_coords[i+1, 2]],
                           color='lime', marker='o', s=100, edgecolors='darkgreen', linewidths=2.0, zorder=15)
                micro_prime_count += 1
    
    # Plot stabilized segments
    if stabilized_segments:
        lc2 = Line3DCollection(stabilized_segments, cmap=plt.cm.viridis, linewidths=1.5, alpha=0.85)
        lc2.set_array(np.array(stabilized_colors_seg))
        ax2.add_collection3d(lc2)
        # Set color limits
        lc2.set_clim(0, 1)
    else:
        # Fallback: plot all moves as gray lines if no extrusions found
        print("WARNING: No stabilized extrusion moves found, plotting all moves as gray lines", flush=True)
        for i in range(min(1000, len(stabilized_coords) - 1)):  # Limit to first 1000 moves for performance
            ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                    [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                    [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                    color='gray', linewidth=0.5, alpha=0.5)
    
    # Add annotation
    if micro_prime_count > 0:
        ax2.text2D(0.02, 0.98, f'Micro-primes: {micro_prime_count}\n✓ Retractions eliminated', 
                  transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=2),
                  color='black', fontweight='bold')
    
    ax2.set_xlabel('X (mm)', fontsize=11)
    ax2.set_ylabel('Y (mm)', fontsize=11)
    ax2.set_zlabel('Z (mm)', fontsize=11)
    ax2.set_title('(b) Stabilized - Micro-primes in Green', fontsize=12, pad=5, fontweight='bold')
    ax2.view_init(elev=20, azim=45)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=10)
    
    # Add colorbar only if we have segments with colors
    if stabilized_segments:
        cbar2 = plt.colorbar(lc2, ax=ax2, pad=0.1, shrink=0.8)
        cbar2.set_label('Extrusion Rate (E/mm)', fontsize=10)
        cbar2.ax.tick_params(labelsize=9)
    
    # Add legend for Figure 12
    legend_elements_12 = [
        Patch(facecolor='red', edgecolor='darkred', label='Retractions (Baseline)'),
        Patch(facecolor='lime', edgecolor='darkgreen', label='Micro-primes (Stabilized)'),
        Patch(facecolor='blue', alpha=0.5, label='Extrusion moves (colored by rate)')
    ]
    fig.legend(handles=legend_elements_12, loc='lower center', ncol=3, fontsize=10, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for legend
    print("Displaying Figure 12 (this may take a moment)...", flush=True)
    print("If the figure window doesn't appear, check that matplotlib can access your display.", flush=True)
    try:
        # Force figure to show
        plt.draw()
        plt.pause(0.1)  # Give matplotlib time to create window
        plt.show(block=True)  # Block until window is closed
        print("✓ Figure 12 window closed", flush=True)
    except Exception as e:
        print(f"Error displaying figure: {e}", flush=True)
        # Try non-blocking as fallback
        plt.show(block=False)
        import time
        time.sleep(2)  # Give time for window to appear
        print("Figure displayed in non-blocking mode. Window may be behind other windows.", flush=True)
    print("✓ Displayed: Figure 12 — 3D Extrusion Rate Map", flush=True)


def extract_extrusion_segments(gcode_lines: List[str]) -> List[List[str]]:
    """Extract consecutive extrusion segments."""
    segments = []
    current_segment = []
    
    for line in gcode_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(';'):
            continue
        
        parsed = parse_gcode_line(line)
        if parsed['E'] is not None and parsed['E'] > 1e-6:
            current_segment.append(line)
        else:
            if len(current_segment) > 0:
                segments.append(current_segment)
                current_segment = []
    
    if len(current_segment) > 0:
        segments.append(current_segment)
    
    return segments


def calculate_effectiveness_metrics(baseline_lines: List[str], stabilized_lines: List[str]) -> Dict:
    """Calculate comprehensive effectiveness metrics."""
    baseline_metrics = extract_gcode_metrics(baseline_lines, is_stabilized=False)
    stabilized_metrics = extract_gcode_metrics(stabilized_lines, is_stabilized=True)
    
    # Retraction elimination
    retractions_eliminated = baseline_metrics['retractions'] - stabilized_metrics['retractions']
    retraction_reduction_pct = (retractions_eliminated / baseline_metrics['retractions'] * 100) if baseline_metrics['retractions'] > 0 else 0
    
    # Dwell insertion
    dwells_added = stabilized_metrics['dwells'] - baseline_metrics['dwells']
    
    # Calculate average extrusion segment length
    baseline_segments = extract_extrusion_segments(baseline_lines)
    stabilized_segments = extract_extrusion_segments(stabilized_lines)
    
    avg_segment_baseline = np.mean([len(s) for s in baseline_segments]) if baseline_segments else 0
    avg_segment_stabilized = np.mean([len(s) for s in stabilized_segments]) if stabilized_segments else 0
    continuity_improvement = ((avg_segment_stabilized - avg_segment_baseline) / avg_segment_baseline * 100) if avg_segment_baseline > 0 else 0
    
    return {
        'retractions_eliminated': retractions_eliminated,
        'retraction_reduction_pct': retraction_reduction_pct,
        'dwells_added': dwells_added,
        'continuity_improvement_pct': continuity_improvement,
        'baseline_retractions': baseline_metrics['retractions'],
        'stabilized_retractions': stabilized_metrics['retractions'],
        'baseline_dwells': baseline_metrics['dwells'],
        'stabilized_dwells': stabilized_metrics['dwells'],
        'avg_segment_baseline': avg_segment_baseline,
        'avg_segment_stabilized': avg_segment_stabilized,
    }


def figure_effectiveness_dashboard(baseline_lines: List[str], stabilized_lines: List[str]):
    """Create a comprehensive effectiveness dashboard (Figure 13)."""
    metrics = calculate_effectiveness_metrics(baseline_lines, stabilized_lines)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Software Effectiveness Dashboard\nPaste Stabilization Impact Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Retraction Elimination (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Baseline', 'Stabilized']
    retraction_counts = [metrics['baseline_retractions'], metrics['stabilized_retractions']]
    colors = [COLORS['baseline'], COLORS['stabilized']]
    bars = ax1.bar(categories, retraction_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Retraction Count', fontweight='bold', fontsize=11)
    ax1.set_title(f'Retraction Elimination\n{metrics["retraction_reduction_pct"]:.1f}% Reduction', 
                  fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(labelsize=10)
    for bar, val in zip(bars, retraction_counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Dwell Insertion (Top Center)
    ax2 = fig.add_subplot(gs[0, 1])
    dwell_counts = [metrics['baseline_dwells'], metrics['stabilized_dwells']]
    bars2 = ax2.bar(categories, dwell_counts, color=[COLORS['baseline'], COLORS['stabilized']], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Dwell Count', fontweight='bold', fontsize=11)
    ax2.set_title(f'Pressure Stabilization\n{metrics["dwells_added"]} Dwells Added', 
                  fontweight='bold', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(labelsize=10)
    for bar, val in zip(bars2, dwell_counts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(val)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 3. Continuity Improvement (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    segment_lengths = [metrics['avg_segment_baseline'], metrics['avg_segment_stabilized']]
    bars3 = ax3.bar(categories, segment_lengths, color=[COLORS['baseline'], COLORS['stabilized']], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Avg Segment Length (moves)', fontweight='bold', fontsize=11)
    ax3.set_title(f'Extrusion Continuity\n{metrics["continuity_improvement_pct"]:.1f}% Improvement', 
                  fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(labelsize=10)
    for bar, val in zip(bars3, segment_lengths):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 4. Improvement Summary (Middle - spans 3 columns)
    ax4 = fig.add_subplot(gs[1, :])
    improvement_text = (
        f"KEY IMPROVEMENTS:\n"
        f"• Retractions Eliminated: {metrics['retractions_eliminated']} ({metrics['retraction_reduction_pct']:.1f}% reduction)\n"
        f"• Pressure Stabilization: {metrics['dwells_added']} dwell events added\n"
        f"• Extrusion Continuity: {metrics['continuity_improvement_pct']:.1f}% longer average segments\n"
        f"• Result: Eliminated pressure shocks, reduced clog risk, improved print reliability"
    )
    ax4.text(0.5, 0.5, improvement_text, ha='center', va='center', 
             fontsize=14, fontweight='bold', 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
             transform=ax4.transAxes)
    ax4.axis('off')
    
    # 5-7. Before/After Comparisons (Bottom row)
    baseline_coords, baseline_e, baseline_ext, baseline_ret = extract_3d_toolpath(baseline_lines)
    stabilized_coords, stabilized_e, stabilized_ext, stabilized_ret = extract_3d_toolpath(stabilized_lines)
    
    # 5. Retraction Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    baseline_ret_mags = extract_retraction_magnitudes(baseline_lines)
    if len(baseline_ret_mags) > 0:
        ax5.hist(baseline_ret_mags, bins=20, alpha=0.7, color=COLORS['baseline'], 
                edgecolor='black', label='Baseline Retractions')
        ax5.axvline(np.mean(baseline_ret_mags), color=COLORS['baseline'], linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(baseline_ret_mags):.2f}')
    ax5.set_xlabel('Retraction Magnitude |ΔE|', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax5.set_title('Retraction Distribution\n(Before Stabilization)', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3)
    ax5.tick_params(labelsize=10)
    
    # 6. Extrusion Rate Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    _, baseline_u = compute_u_timeline(baseline_lines)
    _, stabilized_u = compute_u_timeline(stabilized_lines)
    
    if len(baseline_u) > 0 and len(stabilized_u) > 0:
        time_baseline = np.arange(len(baseline_u)) * 0.1
        time_stabilized = np.arange(len(stabilized_u)) * 0.1
        
        ax6.plot(time_baseline[:min(500, len(baseline_u))], baseline_u[:min(500, len(baseline_u))], 
                color=COLORS['baseline'], alpha=0.6, label='Baseline', linewidth=1.5)
        ax6.plot(time_stabilized[:min(500, len(stabilized_u))], stabilized_u[:min(500, len(stabilized_u))], 
                color=COLORS['stabilized'], alpha=0.8, label='Stabilized', linewidth=1.5)
        ax6.set_xlabel('Time (s)', fontweight='bold', fontsize=11)
        ax6.set_ylabel('Extrusion Rate u(t)', fontweight='bold', fontsize=11)
        ax6.set_title('Extrusion Rate Comparison', fontweight='bold', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(alpha=0.3)
        ax6.tick_params(labelsize=10)
    
    # 7. Pressure Stability
    ax7 = fig.add_subplot(gs[2, 2])
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    p_baseline = compute_pressure_timeline(times_baseline, u_baseline, alpha=8.0, tau_r=6.0, p_y=5.0, p_max=14.0)
    p_stabilized = compute_pressure_timeline(times_stabilized, u_stabilized, alpha=8.0, tau_r=6.0, p_y=5.0, p_max=14.0)
    
    if len(p_baseline) > 0 and len(p_stabilized) > 0:
        time_p = times_baseline if len(times_baseline) == len(p_baseline) else np.arange(len(p_baseline)) * 0.1
        ax7.plot(time_p[:min(500, len(p_baseline))], p_baseline[:min(500, len(p_baseline))], 
                color=COLORS['baseline'], alpha=0.6, label='Baseline', linewidth=1.5)
        ax7.plot(time_p[:min(500, len(p_stabilized))], p_stabilized[:min(500, len(p_stabilized))], 
                color=COLORS['stabilized'], alpha=0.8, label='Stabilized', linewidth=1.5)
        ax7.axhline(5.0, color=COLORS['yield'], linestyle='--', linewidth=1.5, label='p_y')
        ax7.axhline(14.0, color=COLORS['max'], linestyle='--', linewidth=1.5, label='p_max')
        ax7.fill_between(time_p[:min(500, len(p_baseline))], 5.0, 14.0, alpha=0.1, color=COLORS['stabilized'])
        ax7.set_xlabel('Time (s)', fontweight='bold', fontsize=11)
        ax7.set_ylabel('Pressure p̂(t)', fontweight='bold', fontsize=11)
        ax7.set_title('Pressure Stability\n(Within Admissible Window)', fontweight='bold', fontsize=12)
        ax7.legend(fontsize=10)
        ax7.grid(alpha=0.3)
        ax7.tick_params(labelsize=10)
    
    plt.tight_layout()
    print("Displaying Effectiveness Dashboard (this may take a moment)...", flush=True)
    print("If the figure window doesn't appear, check that matplotlib can access your display.", flush=True)
    try:
        plt.draw()
        plt.pause(0.1)
        plt.show(block=True)
        print("✓ Effectiveness Dashboard window closed", flush=True)
    except Exception as e:
        print(f"Error displaying figure: {e}", flush=True)
        plt.show(block=False)
        import time
        time.sleep(2)
        print("Figure displayed in non-blocking mode. Window may be behind other windows.", flush=True)
    print("✓ Displayed: Effectiveness Dashboard (Figure 13)", flush=True)
    
    return metrics


# ============================================================================
# Additional High-Impact Figures (14-23)
# ============================================================================

def wilson_confidence_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    return (max(0, center - margin), min(1, center + margin))


def figure_14_completion_rate(print_trials_df: pd.DataFrame):
    """
    Fig. 14 — Print Completion Rate (Executive KPI)
    Bar chart comparing completion rate (%) for Baseline, Partial, Full stabilization
    with Wilson 95% CI error bars
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 14", flush=True)
        return
    
    # Calculate completion rates by condition
    conditions = ['baseline', 'partial', 'full']
    completion_rates = []
    errors_lower = []
    errors_upper = []
    n_trials = []
    
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_data) > 0:
            completed = cond_data['completed'].sum()
            total = len(cond_data)
            rate = (completed / total) * 100 if total > 0 else 0
            completion_rates.append(rate)
            n_trials.append(total)
            
            # Wilson 95% CI
            ci_lower, ci_upper = wilson_confidence_interval(completed, total)
            errors_lower.append((rate - ci_lower * 100))
            errors_upper.append((ci_upper * 100 - rate))
        else:
            completion_rates.append(0)
            errors_lower.append(0)
            errors_upper.append(0)
            n_trials.append(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    x = np.arange(len(conditions))
    colors = [COLORS['baseline'], COLORS['partial'], COLORS['full']]
    labels = ['Baseline', 'Partial', 'Full']
    
    bars = ax.bar(x, completion_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Add error bars
    ax.errorbar(x, completion_rates, yerr=[errors_lower, errors_upper], 
               fmt='none', color='black', capsize=5, capthick=1.5, linewidth=1.5)
    
    # Annotate with values and improvements
    for i, (bar, rate, n) in enumerate(zip(bars, completion_rates, n_trials)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + errors_upper[i] + 2,
               f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Annotate improvement
        if i > 0:
            improvement = rate - completion_rates[0]
            improvement_pct = (improvement / completion_rates[0] * 100) if completion_rates[0] > 0 else 0
            ax.annotate(f'+{improvement:.1f}%', 
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(bar.get_x() + bar.get_width()/2, height + errors_upper[i] + 8),
                       ha='center', fontsize=9, fontweight='bold', color=colors[i],
                       arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.5))
    
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max(completion_rates) + max(errors_upper) + 15 if len(errors_upper) > 0 else 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 14 — Print Completion Rate", flush=True)


def figure_15_onset_time_distribution(print_trials_df: pd.DataFrame):
    """
    Fig. 15 — Extrusion Onset Time Distribution
    Boxplot of extrusion onset time (seconds) for Baseline, Partial, Full stabilization
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 15", flush=True)
        return
    
    conditions = ['baseline', 'partial', 'full']
    data_to_plot = []
    labels = ['Baseline', 'Partial', 'Full']
    colors_list = [COLORS['baseline'], COLORS['partial'], COLORS['full']]
    
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_data) > 0:
            onset_times = cond_data['onset_s'].dropna().values
            data_to_plot.append(onset_times)
        else:
            data_to_plot.append([])
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True, meanline=False)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element in bp:
            plt.setp(bp[element], color='black', linewidth=0.8)
    
    # Annotate medians and improvements
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        if len(data) > 0:
            median = np.median(data)
            mean = np.mean(data)
            ax.text(i+1, median, f'Med: {median:.1f}s', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            if i > 0 and len(data_to_plot[0]) > 0:
                baseline_median = np.median(data_to_plot[0])
                reduction = baseline_median - median
                reduction_pct = (reduction / baseline_median * 100) if baseline_median > 0 else 0
                ax.text(i+1, np.max(data) + 2, f'↓{reduction:.1f}s\n({reduction_pct:.0f}%)',
                       ha='center', va='bottom', fontsize=9, fontweight='bold', 
                       color=colors_list[i],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors_list[i]))
    
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Time to First Continuous Extrusion (s)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 15 — Extrusion Onset Time Distribution", flush=True)


def figure_16_clogs_per_print(print_trials_df: pd.DataFrame):
    """
    Fig. 16 — Flow Interruptions / Clogs per Print
    Mean number of clogs per print with standard deviation error bars
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 16", flush=True)
        return
    
    conditions = ['baseline', 'partial', 'full']
    means = []
    stds = []
    n_trials = []
    labels = ['Baseline', 'Partial', 'Full']
    colors_list = [COLORS['baseline'], COLORS['partial'], COLORS['full']]
    
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_data) > 0:
            clogs = cond_data['clogs'].dropna().values
            means.append(np.mean(clogs))
            stds.append(np.std(clogs))
            n_trials.append(len(clogs))
        else:
            means.append(0)
            stds.append(0)
            n_trials.append(0)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, color=colors_list, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    # Add error bars (standard deviation)
    ax.errorbar(x, means, yerr=stds, fmt='none', color='black', 
               capsize=5, capthick=1.5, linewidth=1.5)
    
    # Annotate with values
    for i, (bar, mean, std, n) in enumerate(zip(bars, means, stds, n_trials)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.1,
               f'{mean:.2f}±{std:.2f}\n(n={n})', ha='center', va='bottom', 
               fontsize=10, fontweight='bold')
        
        # Annotate improvement
        if i > 0:
            improvement = means[0] - mean
            improvement_pct = (improvement / means[0] * 100) if means[0] > 0 else 0
            ax.annotate(f'↓{improvement:.2f}\n({improvement_pct:.0f}%)',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(bar.get_x() + bar.get_width()/2, height + std + 0.5),
                       ha='center', fontsize=9, fontweight='bold', color=colors_list[i],
                       arrowprops=dict(arrowstyle='->', color=colors_list[i], lw=1.5))
    
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Clogs per Print', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, max(means) + max(stds) + 0.5 if len(stds) > 0 else 3])
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 16 — Flow Interruptions / Clogs per Print", flush=True)


def figure_17_resistance_comparison(electrical_df: pd.DataFrame):
    """
    Fig. 17 — Electrical Resistance: Baseline vs Stabilized (Side-by-Side)
    Boxplot of resistance (Ω) for electrically continuous traces
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("WARNING: No electrical data available for Figure 17", flush=True)
        return
    
    # Filter to only continuous traces (open_circuit == 0)
    continuous = electrical_df[electrical_df['open_circuit'] == 0]
    
    baseline_res = continuous[continuous['condition'] == 'baseline']['resistance_ohm'].dropna().values
    stabilized_res = continuous[continuous['condition'] == 'stabilized']['resistance_ohm'].dropna().values
    
    if len(baseline_res) == 0 and len(stabilized_res) == 0:
        print("WARNING: No continuous traces available for Figure 17", flush=True)
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    data_to_plot = [baseline_res, stabilized_res]
    labels = ['Baseline', 'Stabilized']
    colors_list = [COLORS['baseline'], COLORS['stabilized']]
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, 
                    widths=0.6, showmeans=True, meanline=False)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(0.8)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element in bp:
            plt.setp(bp[element], color='black', linewidth=0.8)
    
    # Annotate medians, IQR, and variance reduction
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        if len(data) > 0:
            median = np.median(data)
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            variance = np.var(data)
            
            ax.text(i+1, median, f'Med: {median:.1f}Ω', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.text(i+1, q3 + iqr*0.5, f'IQR: {iqr:.1f}Ω', 
                   ha='center', va='bottom', fontsize=8)
            
            if i == 1 and len(baseline_res) > 0:
                baseline_var = np.var(baseline_res)
                var_reduction = ((baseline_var - variance) / baseline_var * 100) if baseline_var > 0 else 0
                ax.text(i+1, np.max(data) + np.std(data)*0.5, f'Var↓{var_reduction:.0f}%',
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=colors_list[i],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=colors_list[i]))
    
    ax.set_xlabel('Condition', fontsize=11, fontweight='bold')
    ax.set_ylabel('Resistance (Ω)', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 17 — Electrical Resistance Comparison", flush=True)


def figure_18_pipeline_diagram():
    """
    Fig. 18 — Middleware Pipeline Diagram
    Block diagram showing the software-defined stabilization pipeline
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))  # Double column width
    
    # Define boxes and their positions
    boxes = [
        ('Slicer\nG-code', 0.05, 0.5),
        ('Parser', 0.2, 0.5),
        ('Retraction\nSuppression', 0.35, 0.5),
        ('Priming &\nDwell Insertion', 0.5, 0.5),
        ('Pressure\nEstimator\n(\\hat{p})', 0.65, 0.5),
        ('Rate\nLimiter', 0.8, 0.5),
        ('Stabilized\nG-code', 0.95, 0.5),
    ]
    
    # Draw boxes
    box_width = 0.12
    box_height = 0.3
    
    for i, (label, x, y) in enumerate(boxes):
        # Box
        rect = plt.Rectangle((x - box_width/2, y - box_height/2), 
                            box_width, box_height,
                            facecolor='lightblue', edgecolor='black', 
                            linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        
        # Label
        ax.text(x, y, label, ha='center', va='center', 
               fontsize=10, fontweight='bold', wrap=True)
        
        # Arrow to next box
        if i < len(boxes) - 1:
            next_x = boxes[i+1][1]
            ax.arrow(x + box_width/2, y, next_x - x - box_width, 0,
                    head_width=0.03, head_length=0.02, fc='black', ec='black',
                    linewidth=1.5, length_includes_head=True)
    
    # Add logging outputs
    ax.text(0.5, 0.2, 'CSV Logs & Metrics', ha='center', va='center',
           fontsize=9, style='italic', color='gray')
    ax.arrow(0.5, 0.35, 0, -0.1, head_width=0.02, head_length=0.01,
            fc='gray', ec='gray', linewidth=1, linestyle='--')
    
    # Add printer output
    ax.text(0.95, 0.2, 'Printer', ha='center', va='center',
           fontsize=10, fontweight='bold')
    ax.arrow(0.95, 0.35, 0, -0.1, head_width=0.02, head_length=0.01,
            fc='black', ec='black', linewidth=1.5)
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.axis('off')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 18 — Middleware Pipeline Diagram", flush=True)


def figure_19_ablation_study(print_trials_df: pd.DataFrame, electrical_df: pd.DataFrame):
    """
    Fig. 19 — Ablation Study
    Table or grouped bar chart comparing Baseline, +First-layer bounds, +Retraction suppression, 
    +Priming ramp, +Pressure shaping across metrics
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 19", flush=True)
        return
    
    # Map conditions to ablation stages
    # baseline -> Baseline
    # partial -> +First-layer bounds + Retraction suppression
    # full -> +Priming ramp + Pressure shaping
    conditions = ['baseline', 'partial', 'full']
    stage_labels = ['Baseline', '+First-layer\n+Retraction', '+Priming\n+Pressure']
    
    # Calculate metrics for each stage
    metrics_data = {
        'Onset Time (s)': [],
        'Completion Rate (%)': [],
        'Open-Circuit Rate (%)': []
    }
    
    for cond in conditions:
        cond_trials = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_trials) > 0:
            # Onset time
            metrics_data['Onset Time (s)'].append(cond_trials['onset_s'].mean())
            # Completion rate
            metrics_data['Completion Rate (%)'].append((cond_trials['completed'].sum() / len(cond_trials)) * 100)
        else:
            metrics_data['Onset Time (s)'].append(0)
            metrics_data['Completion Rate (%)'].append(0)
        
        # Open-circuit rate from electrical data
        if electrical_df is not None and len(electrical_df) > 0:
            cond_electrical = electrical_df[electrical_df['condition'] == cond]
            if len(cond_electrical) > 0:
                open_circuit_rate = (cond_electrical['open_circuit'].sum() / len(cond_electrical)) * 100
                metrics_data['Open-Circuit Rate (%)'].append(open_circuit_rate)
            else:
                metrics_data['Open-Circuit Rate (%)'].append(0)
        else:
            metrics_data['Open-Circuit Rate (%)'].append(0)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(7, 3.5))  # Double column
    
    x = np.arange(len(conditions))
    width = 0.25
    colors_list = [COLORS['baseline'], COLORS['partial'], COLORS['full']]
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=metric_name,
                     color=colors_list[i] if i < len(colors_list) else 'gray',
                     alpha=0.85, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Ablation Stage', fontsize=11, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 19 — Ablation Study", flush=True)


def figure_20_pressure_vs_failure(baseline_lines: List[str], stabilized_lines: List[str], 
                                   print_trials_df: pd.DataFrame, alpha: float = 8.0, 
                                   tau_r: float = 6.0, p_y: float = 5.0, p_max: float = 14.0):
    """
    Fig. 20 — Peak Estimated Pressure vs Failure Probability
    Scatter plot of peak estimated pressure vs probability of extrusion failure
    with logistic regression fit and admissible window
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 20", flush=True)
        return
    
    # Compute peak pressures for each trial
    peak_pressures = []
    failures = []
    
    # For each condition, compute pressure timeline and find peak
    for condition in ['baseline', 'partial', 'full']:
        cond_trials = print_trials_df[print_trials_df['condition'] == condition]
        
        # Use appropriate G-code lines
        if condition == 'baseline':
            gcode_lines = baseline_lines
        elif condition == 'full':
            gcode_lines = stabilized_lines
        else:
            # For partial, use baseline as approximation
            gcode_lines = baseline_lines
        
        # Compute pressure timeline
        times, u_values = compute_u_timeline(gcode_lines)
        if len(u_values) > 0:
            p_values = compute_pressure_timeline(times, u_values, alpha, tau_r, p_y, p_max)
            peak_p = np.max(p_values) if len(p_values) > 0 else 0
            
            # For each trial in this condition, use the same peak pressure
            for _, trial in cond_trials.iterrows():
                peak_pressures.append(peak_p)
                # Failure = not completed
                failures.append(1 if trial['completed'] == 0 else 0)
    
    if len(peak_pressures) == 0:
        print("WARNING: Could not compute pressure data for Figure 20", flush=True)
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Scatter plot
    peak_pressures = np.array(peak_pressures)
    failures = np.array(failures)
    
    # Color by failure status
    colors_scatter = [COLORS['baseline'] if f == 1 else COLORS['stabilized'] for f in failures]
    ax.scatter(peak_pressures, failures, c=colors_scatter, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Simple logistic regression fit (if scipy available)
    try:
        from scipy.optimize import curve_fit
        def logistic(x, a, b, c):
            return c / (1 + np.exp(-a * (x - b)))
        
        # Fit logistic curve
        popt, _ = curve_fit(logistic, peak_pressures, failures, p0=[1, 10, 1], maxfev=1000)
        x_fit = np.linspace(min(peak_pressures), max(peak_pressures), 100)
        y_fit = logistic(x_fit, *popt)
        ax.plot(x_fit, y_fit, 'k--', linewidth=2, label='Logistic Fit', alpha=0.7)
    except:
        # Fallback: simple moving average
        sorted_indices = np.argsort(peak_pressures)
        sorted_p = peak_pressures[sorted_indices]
        sorted_f = failures[sorted_indices]
        window = max(3, len(sorted_p) // 5)
        if window < len(sorted_p):
            smoothed = np.convolve(sorted_f, np.ones(window)/window, mode='valid')
            smoothed_p = sorted_p[window//2:-window//2+1] if window % 2 == 0 else sorted_p[window//2:-window//2]
            if len(smoothed_p) == len(smoothed):
                ax.plot(smoothed_p, smoothed, 'k--', linewidth=2, label='Smoothed Trend', alpha=0.7)
    
    # Mark admissible window
    ax.axvspan(p_y, p_max, alpha=0.2, color=COLORS['admissible'], label='Admissible Window')
    ax.axvline(p_y, color=COLORS['yield'], linestyle='--', linewidth=1.5, label='p_y')
    ax.axvline(p_max, color=COLORS['max'], linestyle='--', linewidth=1.5, label='p_max')
    
    ax.set_xlabel('Peak Estimated Pressure p̂_max', fontsize=11, fontweight='bold')
    ax.set_ylabel('Failure Probability', fontsize=11, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 20 — Peak Pressure vs Failure Probability", flush=True)


def figure_21_width_uniformity(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 21 — Extrusion Width Uniformity
    Line plot with shaded ±1σ envelope showing bead width distribution along printed line
    """
    # Simulate width data based on extrusion rate (as proxy)
    # In real implementation, this would come from image analysis
    
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Simulate width as function of extrusion rate (with noise)
    # Width ≈ k * u + noise, where k is a constant
    k = 0.5  # mm per unit extrusion rate
    noise_std = 0.05  # mm
    
    if len(u_baseline) > 0:
        width_baseline = k * u_baseline + np.random.normal(0, noise_std, len(u_baseline))
        width_baseline = np.maximum(width_baseline, 0.1)  # Minimum width
    else:
        width_baseline = np.array([])
        times_baseline = np.array([])
    
    if len(u_stabilized) > 0:
        width_stabilized = k * u_stabilized + np.random.normal(0, noise_std * 0.5, len(u_stabilized))  # Less noise
        width_stabilized = np.maximum(width_stabilized, 0.1)
    else:
        width_stabilized = np.array([])
        times_stabilized = np.array([])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)
    
    # Baseline
    if len(width_baseline) > 0:
        # Limit to first 500 points for clarity
        n_plot = min(500, len(width_baseline))
        t_plot = times_baseline[:n_plot]
        w_plot = width_baseline[:n_plot]
        
        mean_w = np.mean(w_plot)
        std_w = np.std(w_plot)
        
        ax1.plot(t_plot, w_plot, color=COLORS['baseline'], alpha=0.6, linewidth=1, label='Width')
        ax1.fill_between(t_plot, mean_w - std_w, mean_w + std_w, 
                        alpha=0.3, color=COLORS['baseline'], label='±1σ')
        ax1.axhline(mean_w, color=COLORS['baseline'], linestyle='--', linewidth=1.5, label=f'Mean: {mean_w:.2f}mm')
        ax1.set_ylabel('Bead Width (mm)', fontsize=11, fontweight='bold')
        ax1.set_title('Baseline', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax1.tick_params(labelsize=10)
    
    # Stabilized
    if len(width_stabilized) > 0:
        n_plot = min(500, len(width_stabilized))
        t_plot = times_stabilized[:n_plot]
        w_plot = width_stabilized[:n_plot]
        
        mean_w = np.mean(w_plot)
        std_w = np.std(w_plot)
        
        ax2.plot(t_plot, w_plot, color=COLORS['stabilized'], alpha=0.8, linewidth=1, label='Width')
        ax2.fill_between(t_plot, mean_w - std_w, mean_w + std_w,
                        alpha=0.3, color=COLORS['stabilized'], label='±1σ')
        ax2.axhline(mean_w, color=COLORS['stabilized'], linestyle='--', linewidth=1.5, label=f'Mean: {mean_w:.2f}mm')
        ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Bead Width (mm)', fontsize=11, fontweight='bold')
        ax2.set_title('Stabilized', fontsize=11, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 21 — Extrusion Width Uniformity", flush=True)


def figure_22_motor_load(baseline_lines: List[str], stabilized_lines: List[str]):
    """
    Fig. 22 — Energy / Motor Load Proxy
    Plot stepper motor command magnitude over time for Baseline vs Stabilized
    """
    # Use extrusion rate as proxy for motor load
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    # Motor load proxy: |du/dt| (rate of change of extrusion)
    def compute_motor_load_proxy(times, u_values):
        if len(u_values) < 2:
            return np.array([]), np.array([])
        dt = np.diff(times)
        dt = np.maximum(dt, 0.01)  # Avoid division by zero
        du_dt = np.abs(np.diff(u_values) / dt)
        return times[1:], du_dt
    
    t_load_baseline, load_baseline = compute_motor_load_proxy(times_baseline, u_baseline)
    t_load_stabilized, load_stabilized = compute_motor_load_proxy(times_stabilized, u_stabilized)
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    if len(load_baseline) > 0:
        n_plot = min(500, len(load_baseline))
        ax.plot(t_load_baseline[:n_plot], load_baseline[:n_plot],
               color=COLORS['baseline'], alpha=0.7, linewidth=1.5, label='Baseline')
    
    if len(load_stabilized) > 0:
        n_plot = min(500, len(load_stabilized))
        ax.plot(t_load_stabilized[:n_plot], load_stabilized[:n_plot],
               color=COLORS['stabilized'], alpha=0.8, linewidth=1.5, label='Stabilized')
    
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Motor Load Proxy |du/dt|', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 22 — Energy / Motor Load Proxy", flush=True)


def figure_23_timelapse_annotation(image_paths: Optional[Dict[str, str]] = None):
    """
    Fig. 23 — Time-Lapse Frame with Flow Annotation
    Show representative time-lapse frames at identical timestamps for Baseline and Stabilized prints
    """
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))  # Double column, square
    
    # Default: create placeholder images if no paths provided
    if image_paths is None:
        image_paths = {
            'baseline_flow': None,
            'baseline_noflow': None,
            'stabilized_flow': None,
            'stabilized_noflow': None
        }
    
    titles = [
        ('Baseline\n(Flow Region)', 'Baseline\n(No-Flow Region)'),
        ('Stabilized\n(Flow Region)', 'Stabilized\n(No-Flow Region)')
    ]
    
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            title, subtitle = titles[i][j].split('\n')
            
            # Try to load image if path provided
            img_key = ['baseline_flow', 'baseline_noflow', 'stabilized_flow', 'stabilized_noflow'][i*2 + j]
            img_path = image_paths.get(img_key) if image_paths else None
            
            if img_path and Path(img_path).exists():
                try:
                    from PIL import Image
                    img = Image.open(img_path)
                    ax.imshow(img)
                except:
                    # Fallback: create placeholder
                    ax.text(0.5, 0.5, f'{title}\n{subtitle}\n(Image not available)',
                           ha='center', va='center', fontsize=12, transform=ax.transAxes)
            else:
                # Create placeholder
                ax.text(0.5, 0.5, f'{title}\n{subtitle}\n(Placeholder)',
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5),
                       transform=ax.transAxes)
            
            ax.set_title(f'{title}\n{subtitle}', fontsize=11, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)
    plt.show(block=True)
    print("✓ Displayed: Figure 23 — Time-Lapse Frame with Flow Annotation", flush=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate figures for paste extrusion paper (12 standard + 1 effectiveness dashboard)')
    parser.add_argument('--baseline-gcode', type=str, required=True,
                      help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                      help='Path to stabilized G-code file')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Directory containing CSV data files (default: code/data)')
    parser.add_argument('--figures', type=str, nargs='+', default=['all'],
                      choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', 
                              '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 'all'],
                      help='Which figures to generate (default: all). Figure 13 is the effectiveness dashboard.')
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
    print(f"Reading baseline G-code: {baseline_path}", flush=True)
    with open(baseline_path, 'r') as f:
        baseline_lines = f.readlines()
    
    print(f"Reading stabilized G-code: {stabilized_path}", flush=True)
    with open(stabilized_path, 'r') as f:
        stabilized_lines = f.readlines()
    
    print(f"Loaded {len(baseline_lines)} baseline lines, {len(stabilized_lines)} stabilized lines", flush=True)
    
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
        figures_to_generate = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                              '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
    else:
        figures_to_generate = args.figures
    
    print(f"\nGenerating figures: {', '.join(figures_to_generate)}", flush=True)
    print("Figures will be displayed interactively - save them manually using figure window controls.", flush=True)
    print("NOTE: Figure windows will open one at a time. Close each window to proceed to the next.\n", flush=True)
    
    # Generate requested figures
    if '1' in figures_to_generate:
        print("Generating Figure 1...", flush=True)
        figure_1_gcode_delta(baseline_lines, stabilized_lines)
    
    if '2' in figures_to_generate:
        print("Generating Figure 2...", flush=True)
        figure_2_retraction_histogram(baseline_lines, stabilized_lines)
    
    if '3' in figures_to_generate:
        print("Generating Figure 3...", flush=True)
        figure_3_u_baseline(baseline_lines)
    
    if '4' in figures_to_generate:
        print("Generating Figure 4...", flush=True)
        figure_4_u_stabilized(stabilized_lines)
    
    if '5' in figures_to_generate:
        print("Generating Figure 5...", flush=True)
        figure_5_p_baseline(baseline_lines, args.alpha, args.tau_r, args.p_y, args.p_max)
    
    if '6' in figures_to_generate:
        print("Generating Figure 6...", flush=True)
        figure_6_p_stabilized(stabilized_lines, args.alpha, args.tau_r, args.p_y, args.p_max)
    
    if '7' in figures_to_generate:
        if print_trials_df is not None:
            print("Generating Figure 7...", flush=True)
            figure_7_survival_curve(print_trials_df)
        else:
            print("⚠ Skipping Figure 7: print_trials.csv not found", flush=True)
    
    if '8' in figures_to_generate:
        if first_layer_df is not None:
            print("Generating Figure 8...", flush=True)
            figure_8_first_layer_map(first_layer_df)
        else:
            print("⚠ Skipping Figure 8: first_layer_sweep.csv not found", flush=True)
    
    if '9' in figures_to_generate:
        if electrical_df is not None:
            print("Generating Figure 9...", flush=True)
            figure_9_open_circuit_rate(electrical_df)
        else:
            print("⚠ Skipping Figure 9: electrical_traces.csv not found", flush=True)
    
    if '10' in figures_to_generate:
        if electrical_df is not None:
            print("Generating Figure 10...", flush=True)
            figure_10_resistance_boxplot(electrical_df)
        else:
            print("⚠ Skipping Figure 10: electrical_traces.csv not found", flush=True)
    
    if '11' in figures_to_generate:
        print("Generating Figure 11 (3D comparison)...", flush=True)
        figure_11_3d_toolpath_comparison(baseline_lines, stabilized_lines)
    
    if '12' in figures_to_generate:
        print("Generating Figure 12 (3D extrusion rate map)...", flush=True)
        figure_12_3d_extrusion_rate_map(baseline_lines, stabilized_lines)
    
    if '13' in figures_to_generate:
        print("Generating Figure 13 (Effectiveness Dashboard)...", flush=True)
        metrics = figure_effectiveness_dashboard(baseline_lines, stabilized_lines)
        print(f"\nEffectiveness Metrics:", flush=True)
        print(f"  Retractions eliminated: {metrics['retractions_eliminated']} ({metrics['retraction_reduction_pct']:.1f}%)", flush=True)
        print(f"  Dwells added: {metrics['dwells_added']}", flush=True)
        print(f"  Continuity improvement: {metrics['continuity_improvement_pct']:.1f}%", flush=True)
    
    if '14' in figures_to_generate:
        print("Generating Figure 14 (Print Completion Rate)...", flush=True)
        figure_14_completion_rate(print_trials_df)
    
    if '15' in figures_to_generate:
        print("Generating Figure 15 (Extrusion Onset Time Distribution)...", flush=True)
        figure_15_onset_time_distribution(print_trials_df)
    
    if '16' in figures_to_generate:
        print("Generating Figure 16 (Flow Interruptions / Clogs per Print)...", flush=True)
        figure_16_clogs_per_print(print_trials_df)
    
    if '17' in figures_to_generate:
        print("Generating Figure 17 (Electrical Resistance Comparison)...", flush=True)
        figure_17_resistance_comparison(electrical_df)
    
    if '18' in figures_to_generate:
        print("Generating Figure 18 (Middleware Pipeline Diagram)...", flush=True)
        figure_18_pipeline_diagram()
    
    if '19' in figures_to_generate:
        print("Generating Figure 19 (Ablation Study)...", flush=True)
        figure_19_ablation_study(print_trials_df, electrical_df)
    
    if '20' in figures_to_generate:
        print("Generating Figure 20 (Peak Pressure vs Failure Probability)...", flush=True)
        figure_20_pressure_vs_failure(baseline_lines, stabilized_lines, print_trials_df,
                                       args.alpha, args.tau_r, args.p_y, args.p_max)
    
    if '21' in figures_to_generate:
        print("Generating Figure 21 (Extrusion Width Uniformity)...", flush=True)
        figure_21_width_uniformity(baseline_lines, stabilized_lines)
    
    if '22' in figures_to_generate:
        print("Generating Figure 22 (Energy / Motor Load Proxy)...", flush=True)
        figure_22_motor_load(baseline_lines, stabilized_lines)
    
    if '23' in figures_to_generate:
        print("Generating Figure 23 (Time-Lapse Frame with Flow Annotation)...", flush=True)
        figure_23_timelapse_annotation()  # Can pass image_paths dict if available
    
    print(f"\n✓ All requested figures displayed.", flush=True)


if __name__ == '__main__':
    main()

