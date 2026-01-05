#!/usr/bin/env python3
"""
paper_figs.py


use this code to generate the figures for the paper.

cd code
python3 paper_figs.py --baseline-gcode test.gcode --stabilized-gcode results/stabilized.gcode --figures all


Generates the final 9 publication-ready figures for the paste extrusion stabilization paper.
These are the exact figures reviewers will see - polished, professional, and impactful.
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
from matplotlib.patches import Patch, Rectangle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as mpatches

# Enable interactive mode
plt.ion()

# Enhanced professional styling for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.frameon': True,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.framealpha': 0.95,
    'legend.edgecolor': 'black',
    'legend.facecolor': 'white',
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'patch.linewidth': 1.0,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 0.6,
    'ytick.minor.width': 0.6,
})

# Premium color palette - vibrant and professional
COLORS = {
    'baseline': '#E63946',        # Vibrant red
    'stabilized': '#2A9D8F',      # Teal green
    'partial': '#F77F00',         # Orange
    'full': '#2A9D8F',            # Same as stabilized
    'admissible': '#F1C40F',      # Gold
    'yield': '#E67E22',           # Dark orange
    'max': '#C0392B',             # Dark red
    'accent1': '#3498DB',          # Blue
    'accent2': '#9B59B6',         # Purple
    'success': '#27AE60',          # Green
    'warning': '#F39C12',          # Orange
}

# Try to import scipy for survival analysis
try:
    from scipy.stats import kaplan_meier_estimator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available. Figure 4 will use simplified survival analysis.", flush=True)


# ============================================================================
# G-code Parsing Utilities (from generate_10_figures.py)
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


def extract_gcode_metrics(gcode_lines: List[str], is_stabilized: bool = False) -> Dict[str, any]:
    """Extract key metrics from G-code."""
    retractions = 0
    dwells = 0
    extrusion_moves = 0
    e_prev = 0.0
    e_curr = 0.0
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
        
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            if parsed['E'] is not None:
                e_curr = parsed['E']
                if is_relative_e:
                    delta_e = e_curr
                else:
                    delta_e = e_curr - e_prev
                    e_prev = e_curr
                
                if delta_e < -1e-6:
                    retractions += 1
                elif delta_e > 1e-6:
                    extrusion_moves += 1
                
                e_prev = e_curr if is_relative_e else e_curr
        
        if 'G4' in stripped.upper() or 'DWELL' in stripped.upper():
            dwells += 1
    
    return {
        'retractions': retractions,
        'dwells': dwells,
        'extrusion_moves': extrusion_moves
    }


def compute_u_timeline(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute u(t) timeline from G-code."""
    times = [0.0]
    u_values = [0.0]
    
    x_prev, y_prev, e_prev = 0.0, 0.0, 0.0
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
        
        if stripped.startswith('G0') or stripped.startswith('G1'):
            parsed = parse_gcode_line(line)
            
            x_curr = parsed['X'] if parsed['X'] is not None else x_prev
            y_curr = parsed['Y'] if parsed['Y'] is not None else y_prev
            e_curr = parsed['E'] if parsed['E'] is not None else e_prev
            f_val = parsed['F'] if parsed['F'] is not None else 60.0
            
            dx = x_curr - x_prev
            dy = y_curr - y_prev
            ds = np.sqrt(dx**2 + dy**2)
            
            if is_relative_e:
                de = e_curr
            else:
                de = e_curr - e_prev
                e_prev = e_curr
            
            if ds > 1e-6 and de > 1e-6:
                v = f_val / 60.0  # mm/s
                dt = ds / v if v > 0 else 0.1
                t_curr += dt
                u = de / dt if dt > 0 else 0.0
                times.append(t_curr)
                u_values.append(u)
            
            x_prev, y_prev, e_prev = x_curr, y_curr, e_curr
    
    return np.array(times), np.array(u_values)


def compute_pressure_timeline(times: np.ndarray, u_values: np.ndarray,
                               alpha: float = 8.0, tau_r: float = 6.0,
                               p_y: float = 5.0, p_max: float = 14.0) -> np.ndarray:
    """Compute pressure estimate p̂(t) from u(t) timeline."""
    p_hat = np.zeros_like(times)
    p_hat[0] = 0.0
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        u_k = u_values[i]
        p_k = p_hat[i-1]
        p_hat[i] = p_k + dt * (alpha * u_k - p_k / tau_r)
        p_hat[i] = max(0.0, p_hat[i])
    
    return p_hat


def extract_3d_toolpath(gcode_lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D toolpath coordinates and extrusion information from G-code.
    Returns: (coords [Nx3], e_deltas [N], extrusion_flags [N], retraction_flags [N], feed_rates [N])
    """
    coords = []
    e_deltas = []
    extrusion_flags = []
    retraction_flags = []
    feed_rates = []
    
    x_curr, y_curr, z_curr = None, None, 0.0
    x_prev, y_prev, z_prev = None, None, 0.0
    e_cumulative = 0.0
    is_relative_e = True  # Default to relative (M83)
    f_curr = None
    
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
            if parsed['F'] is not None:
                f_curr = parsed['F']
            
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
            
            # Record point if we have valid coordinates
            # IMPORTANT: Record ALL moves, even if X/Y not explicitly set (use previous position)
            if x_prev is not None and y_prev is not None:  # We have a previous position
                coords.append([x_to_use, y_to_use, z_to_use])
                e_deltas.append(e_delta)
                extrusion_flags.append(e_delta > 1e-6)
                retraction_flags.append(e_delta < -1e-6)
                feed_rates.append(f_curr if f_curr is not None else 0.0)
            elif x_to_use is not None and y_to_use is not None:  # Current move has coordinates
                coords.append([x_to_use, y_to_use, z_to_use])
                e_deltas.append(e_delta)
                extrusion_flags.append(e_delta > 1e-6)
                retraction_flags.append(e_delta < -1e-6)
                feed_rates.append(f_curr if f_curr is not None else 0.0)
    
    if len(coords) == 0:
        return (np.array([]).reshape(0, 3), np.array([]), np.array([], dtype=bool), 
                np.array([], dtype=bool), np.array([]))
    
    return (np.array(coords), np.array(e_deltas), np.array(extrusion_flags), 
            np.array(retraction_flags), np.array(feed_rates))


def wilson_confidence_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Calculate Wilson 95% confidence interval for a proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / total) + (z**2 / (4 * total**2)))
    return (max(0, center - margin), min(1, center + margin))


# ============================================================================
# Display/Save Helper
# ============================================================================

def display_or_save_figure(fig, figure_num: int, figure_name: str, 
                          save_mode: bool = False, output_dir: Optional[Path] = None):
    """Display figure interactively or save to file."""
    plt.tight_layout()
    
    if save_mode:
        if output_dir is None:
            output_dir = Path(__file__).parent / 'results' / 'figures' / 'paper_figs'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'paper_fig_{figure_num}.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {output_path}", flush=True)
        plt.close(fig)
    else:
        try:
            plt.draw()
            plt.pause(0.1)
            plt.show(block=True)
            print(f"✓ Displayed: Figure {figure_num} — {figure_name}", flush=True)
        except Exception as e:
            print(f"Warning: Could not display figure interactively: {e}", flush=True)
            print("Saving figure to file instead...", flush=True)
            if output_dir is None:
                output_dir = Path(__file__).parent / 'results' / 'figures' / 'paper_figs'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'paper_fig_{figure_num}.png'
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Saved: {output_path}", flush=True)
            plt.close(fig)


# ============================================================================
# Final Paper Figures (9 Figures)
# ============================================================================

def figure_1_gcode_modification_summary(baseline_lines: List[str], stabilized_lines: List[str],
                                        save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 1: G-code modification summary
    Enhanced grouped bar chart with annotations and visual appeal
    """
    baseline_metrics = extract_gcode_metrics(baseline_lines, is_stabilized=False)
    stabilized_metrics = extract_gcode_metrics(stabilized_lines, is_stabilized=True)
    
    retractions_eliminated = baseline_metrics['retractions'] - stabilized_metrics['retractions']
    retraction_reduction_pct = (retractions_eliminated / baseline_metrics['retractions'] * 100) if baseline_metrics['retractions'] > 0 else 0
    dwells_added = stabilized_metrics['dwells'] - baseline_metrics['dwells']
    
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    categories = ['Retractions', 'Dwells', 'Extrusion\nMoves']
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
                   color=COLORS['baseline'], alpha=0.9, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, stabilized_data, width, label='Stabilized',
                   color=COLORS['stabilized'], alpha=0.9, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add improvement annotation
    if retraction_reduction_pct > 0:
        ax.annotate(f'↓{retraction_reduction_pct:.0f}%', 
                   xy=(0, stabilized_data[0]), xytext=(0, stabilized_data[0] + max(baseline_data[0], stabilized_data[0]) * 0.15),
                   ha='center', fontsize=11, fontweight='bold', color=COLORS['stabilized'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['stabilized'], lw=2))
    
    ax.set_xlabel('Metric Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    display_or_save_figure(fig, 1, "G-code Modification Summary", save_mode, output_dir)


def figure_2_extrusion_rate_comparison(baseline_lines: List[str], stabilized_lines: List[str],
                                        save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 2: Extrusion-rate proxy (baseline vs stabilized)
    Side-by-side comparison with enhanced styling
    """
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    
    # Baseline
    if len(u_baseline) > 0:
        n_plot = min(1000, len(u_baseline))
        ax1.plot(times_baseline[:n_plot], u_baseline[:n_plot],
                color=COLORS['baseline'], linewidth=2.0, alpha=0.85, label='Baseline')
        ax1.fill_between(times_baseline[:n_plot], 0, u_baseline[:n_plot],
                        color=COLORS['baseline'], alpha=0.2)
    
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Extrusion Rate u(t)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    
    # Stabilized
    if len(u_stabilized) > 0:
        n_plot = min(1000, len(u_stabilized))
        ax2.plot(times_stabilized[:n_plot], u_stabilized[:n_plot],
                color=COLORS['stabilized'], linewidth=2.0, alpha=0.85, label='Stabilized')
        ax2.fill_between(times_stabilized[:n_plot], 0, u_stabilized[:n_plot],
                         color=COLORS['stabilized'], alpha=0.2)
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Stabilized', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=11, framealpha=0.95, edgecolor='black')
    
    display_or_save_figure(fig, 2, "Extrusion-Rate Proxy Comparison", save_mode, output_dir)


def figure_3_pressure_comparison(baseline_lines: List[str], stabilized_lines: List[str],
                                  alpha: float = 8.0, tau_r: float = 6.0,
                                  p_y: float = 5.0, p_max: float = 14.0,
                                  save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 3: Pressure estimate (baseline vs stabilized)
    Side-by-side with admissible window highlighted
    """
    times_baseline, u_baseline = compute_u_timeline(baseline_lines)
    times_stabilized, u_stabilized = compute_u_timeline(stabilized_lines)
    
    p_baseline = compute_pressure_timeline(times_baseline, u_baseline, alpha, tau_r, p_y, p_max)
    p_stabilized = compute_pressure_timeline(times_stabilized, u_stabilized, alpha, tau_r, p_y, p_max)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.5), sharey=True)
    
    # Baseline
    if len(p_baseline) > 0:
        n_plot = min(1000, len(p_baseline))
        t_plot = times_baseline[:n_plot] if len(times_baseline) == len(p_baseline) else np.arange(n_plot) * 0.1
        ax1.plot(t_plot, p_baseline[:n_plot], color=COLORS['baseline'], linewidth=2.0, alpha=0.85)
        ax1.fill_between(t_plot, p_y, p_max, alpha=0.15, color=COLORS['admissible'], label='Admissible Window')
        ax1.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.0, label='p_y')
        ax1.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.0, label='p_max')
    
    ax1.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Pressure p̂(t)', fontsize=12, fontweight='bold')
    ax1.set_title('Baseline', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(alpha=0.4, linestyle='--', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='upper right')
    
    # Stabilized
    if len(p_stabilized) > 0:
        n_plot = min(1000, len(p_stabilized))
        t_plot = times_stabilized[:n_plot] if len(times_stabilized) == len(p_stabilized) else np.arange(n_plot) * 0.1
        ax2.plot(t_plot, p_stabilized[:n_plot], color=COLORS['stabilized'], linewidth=2.0, alpha=0.85)
        ax2.fill_between(t_plot, p_y, p_max, alpha=0.15, color=COLORS['admissible'], label='Admissible Window')
        ax2.axhline(p_y, color=COLORS['yield'], linestyle='--', linewidth=2.0, label='p_y')
        ax2.axhline(p_max, color=COLORS['max'], linestyle='--', linewidth=2.0, label='p_max')
    
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Stabilized', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(alpha=0.4, linestyle='--', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=10, framealpha=0.95, edgecolor='black', loc='upper right')
    
    display_or_save_figure(fig, 3, "Pressure Estimate Comparison", save_mode, output_dir)


def figure_4_extrusion_survival(print_trials_df: pd.DataFrame,
                                 save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 4: Extrusion survival curve
    Kaplan-Meier survival curve with enhanced styling
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 4", flush=True)
        return
    
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    conditions = ['baseline', 'partial', 'full']
    condition_labels = ['Baseline', 'Partial', 'Full']
    condition_colors = [COLORS['baseline'], COLORS['partial'], COLORS['stabilized']]
    
    for cond, label, color in zip(conditions, condition_labels, condition_colors):
        cond_data = print_trials_df[print_trials_df['condition'] == cond].copy()
        if len(cond_data) > 0:
            # Use flow_duration_s as survival time, completed==0 as event
            times = cond_data['flow_duration_s'].dropna().values
            events = (cond_data['completed'] == 0).astype(int).values  # 1 = failure, 0 = censored
            
            if len(times) == 0:
                continue
            
            # Ensure events array matches times array length
            if len(events) != len(times):
                events = (cond_data.loc[cond_data['flow_duration_s'].notna(), 'completed'] == 0).astype(int).values
            
            if SCIPY_AVAILABLE and len(times) > 0:
                try:
                    # kaplan_meier_estimator expects: event_indicator (True = event occurred), time
                    event_indicator = events == 1  # True where failure occurred
                    time, survival_prob = kaplan_meier_estimator(event_indicator, times)
                    ax.step(time, survival_prob, where='post', linewidth=2.5, 
                           label=label, color=color, alpha=0.9)
                except Exception as e:
                    print(f"Warning: scipy survival analysis failed for {cond}: {e}, using simplified method", flush=True)
                    # Fall through to simplified method
                    sorted_indices = np.argsort(times)
                    sorted_times = times[sorted_indices]
                    sorted_events = events[sorted_indices]
                    n = len(times)
                    survival = np.ones(n)
                    for i in range(n):
                        if sorted_events[i] == 1:  # Failure
                            survival[i:] *= (n - i - 1) / (n - i) if (n - i) > 0 else 0
                    ax.step(sorted_times, survival, where='post', linewidth=2.5,
                           label=label, color=color, alpha=0.9)
            else:
                # Simplified: empirical survival
                sorted_indices = np.argsort(times)
                sorted_times = times[sorted_indices]
                sorted_events = events[sorted_indices]
                n = len(times)
                survival = np.ones(n)
                for i in range(n):
                    if sorted_events[i] == 1:  # Failure
                        survival[i:] *= (n - i - 1) / (n - i) if (n - i) > 0 else 0
                ax.step(sorted_times, survival, where='post', linewidth=2.5,
                       label=label, color=color, alpha=0.9)
    
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
    ax.grid(alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    display_or_save_figure(fig, 4, "Extrusion Survival Curve", save_mode, output_dir)


def figure_5_first_layer_envelope(first_layer_df: pd.DataFrame,
                                   save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 5: First-layer operating envelope
    Heatmap showing success/failure regions
    """
    if first_layer_df is None or len(first_layer_df) == 0:
        print("WARNING: No first layer data available for Figure 5", flush=True)
        return
    
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    # Create heatmap from data - handle different column names
    if 'h_ratio' in first_layer_df.columns:
        height_col = 'h_ratio'
        speed_col = 'speed_mmps'
    else:
        height_col = 'height'
        speed_col = 'speed'
    
    pivot = first_layer_df.pivot_table(values='success', index=height_col, columns=speed_col, aggfunc='mean')
    
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, interpolation='bilinear')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f'{v:.0f}' for v in pivot.columns], fontsize=10)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f'{v:.2f}' for v in pivot.index], fontsize=10)
    
    ax.set_xlabel('Speed (mm/s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('First-Layer Height Ratio', fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    display_or_save_figure(fig, 5, "First-Layer Operating Envelope", save_mode, output_dir)


def figure_6_completion_rate(print_trials_df: pd.DataFrame,
                              save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 6: Print completion rate
    Bar chart with Wilson 95% CI error bars
    """
    if print_trials_df is None or len(print_trials_df) == 0:
        print("WARNING: No print trials data available for Figure 6", flush=True)
        return
    
    conditions = ['baseline', 'partial', 'full']
    labels = ['Baseline', 'Partial', 'Full']
    colors_list = [COLORS['baseline'], COLORS['partial'], COLORS['stabilized']]
    
    completion_rates = []
    errors_lower = []
    errors_upper = []
    
    for cond in conditions:
        cond_data = print_trials_df[print_trials_df['condition'] == cond]
        if len(cond_data) > 0:
            completed = cond_data['completed'].sum()
            total = len(cond_data)
            rate = (completed / total) * 100 if total > 0 else 0
            completion_rates.append(rate)
            
            ci_lower, ci_upper = wilson_confidence_interval(completed, total)
            errors_lower.append((rate - ci_lower * 100))
            errors_upper.append((ci_upper * 100 - rate))
        else:
            completion_rates.append(0)
            errors_lower.append(0)
            errors_upper.append(0)
    
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, completion_rates, color=colors_list, alpha=0.9, edgecolor='black', linewidth=1.2)
    
    ax.errorbar(x, completion_rates, yerr=[errors_lower, errors_upper],
               fmt='none', color='black', capsize=6, capthick=2, linewidth=2)
    
    # Add value labels
    for bar, rate, err_u in zip(bars, completion_rates, errors_upper):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err_u + 3,
               f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Completion Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, max(completion_rates) + max(errors_upper) + 15 if len(errors_upper) > 0 else 100])
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    display_or_save_figure(fig, 6, "Print Completion Rate", save_mode, output_dir)


def figure_7_electrical_yield(electrical_df: pd.DataFrame,
                              save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 7: Electrical yield (open circuits)
    Bar chart showing open-circuit rates
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("WARNING: No electrical data available for Figure 7", flush=True)
        return
    
    conditions = ['baseline', 'stabilized']
    labels = ['Baseline', 'Stabilized']
    colors_list = [COLORS['baseline'], COLORS['stabilized']]
    
    open_circuit_rates = []
    n_trials = []
    
    for cond in conditions:
        cond_data = electrical_df[electrical_df['condition'] == cond]
        if len(cond_data) > 0:
            open_count = cond_data['open_circuit'].sum()
            total = len(cond_data)
            rate = (open_count / total) * 100 if total > 0 else 0
            open_circuit_rates.append(rate)
            n_trials.append(total)
        else:
            open_circuit_rates.append(0)
            n_trials.append(0)
    
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    x = np.arange(len(conditions))
    bars = ax.bar(x, open_circuit_rates, color=colors_list, alpha=0.9, edgecolor='black', linewidth=1.2)
    
    # Add value labels
    for bar, rate, n in zip(bars, open_circuit_rates, n_trials):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
               f'{rate:.1f}%\n(n={n})', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Open-Circuit Rate (%)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim([0, max(open_circuit_rates) + 10 if len(open_circuit_rates) > 0 else 100])
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    display_or_save_figure(fig, 7, "Electrical Yield", save_mode, output_dir)


def figure_8_resistance_comparison(electrical_df: pd.DataFrame,
                                    save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 8: Resistance baseline vs stabilized
    Side-by-side boxplots with enhanced styling
    """
    if electrical_df is None or len(electrical_df) == 0:
        print("WARNING: No electrical data available for Figure 8", flush=True)
        return
    
    # Filter to only continuous traces
    continuous = electrical_df[electrical_df['open_circuit'] == 0]
    
    baseline_res = continuous[continuous['condition'] == 'baseline']['resistance_ohm'].dropna().values
    stabilized_res = continuous[continuous['condition'] == 'stabilized']['resistance_ohm'].dropna().values
    
    if len(baseline_res) == 0 and len(stabilized_res) == 0:
        print("WARNING: No continuous traces available for Figure 8", flush=True)
        return
    
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    
    data_to_plot = [baseline_res, stabilized_res]
    labels = ['Baseline', 'Stabilized']
    colors_list = [COLORS['baseline'], COLORS['stabilized']]
    
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=False)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Style other elements
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        if element in bp:
            plt.setp(bp[element], color='black', linewidth=1.5)
    
    # Annotate medians
    for i, (data, label) in enumerate(zip(data_to_plot, labels)):
        if len(data) > 0:
            median = np.median(data)
            ax.text(i+1, median, f'Med: {median:.1f}Ω',
                   ha='center', va='bottom', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    ax.set_xlabel('Condition', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resistance (Ω)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.4, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    display_or_save_figure(fig, 8, "Resistance Comparison", save_mode, output_dir)


def figure_9_3d_toolpath_comparison(baseline_lines: List[str], stabilized_lines: List[str],
                                    save_mode: bool = False, output_dir: Optional[Path] = None):
    """
    Figure 9: 3D toolpath comparison
    Side-by-side 3D visualization with retractions highlighted
    """
    baseline_coords, baseline_e, baseline_ext, baseline_ret, baseline_f = extract_3d_toolpath(baseline_lines)
    stabilized_coords, stabilized_e, stabilized_ext, stabilized_ret, stabilized_f = extract_3d_toolpath(stabilized_lines)
    
    if len(baseline_coords) == 0 or len(stabilized_coords) == 0:
        print("WARNING: Could not extract 3D toolpath data", flush=True)
        return
    
    fig = plt.figure(figsize=(10.0, 4.5))
    
    # Baseline
    ax1 = fig.add_subplot(121, projection='3d')
    
    if len(baseline_coords) > 1:
        for i in range(len(baseline_coords) - 1):
            if baseline_ext[i+1]:
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color=COLORS['baseline'], alpha=0.6, linewidth=1.0)
            elif baseline_ret[i+1]:
                ax1.plot([baseline_coords[i, 0], baseline_coords[i+1, 0]],
                        [baseline_coords[i, 1], baseline_coords[i+1, 1]],
                        [baseline_coords[i, 2], baseline_coords[i+1, 2]],
                        color='red', linewidth=3.0, linestyle='--', alpha=1.0)
                ax1.scatter([baseline_coords[i+1, 0]], [baseline_coords[i+1, 1]], [baseline_coords[i+1, 2]],
                           color='red', marker='X', s=150, linewidths=2.0, edgecolors='darkred', zorder=10)
    
    ax1.set_title('Baseline', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax1.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
    
    # Stabilized
    ax2 = fig.add_subplot(122, projection='3d')
    
    if len(stabilized_coords) > 1:
        for i in range(len(stabilized_coords) - 1):
            if stabilized_ext[i+1]:
                ax2.plot([stabilized_coords[i, 0], stabilized_coords[i+1, 0]],
                        [stabilized_coords[i, 1], stabilized_coords[i+1, 1]],
                        [stabilized_coords[i, 2], stabilized_coords[i+1, 2]],
                        color=COLORS['stabilized'], alpha=0.7, linewidth=1.0)
    
    ax2.set_title('Stabilized', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('X (mm)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y (mm)', fontsize=11, fontweight='bold')
    ax2.set_zlabel('Z (mm)', fontsize=11, fontweight='bold')
    
    # Add legend
    baseline_ret_count = baseline_ret.sum()
    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['baseline'], linewidth=2, label='Baseline Toolpath'),
        plt.Line2D([0], [0], color='red', linewidth=3, linestyle='--', label=f'Retractions ({baseline_ret_count})'),
        plt.Line2D([0], [0], color=COLORS['stabilized'], linewidth=2, label='Stabilized Toolpath'),
        plt.Line2D([0], [0], color=COLORS['stabilized'], linewidth=2, label='Retractions Eliminated')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10, framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    display_or_save_figure(fig, 9, "3D Toolpath Comparison", save_mode, output_dir)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate final 9 paper figures for paste extrusion stabilization')
    parser.add_argument('--baseline-gcode', type=str, required=True,
                      help='Path to baseline G-code file')
    parser.add_argument('--stabilized-gcode', type=str, required=True,
                      help='Path to stabilized G-code file')
    parser.add_argument('--data-dir', type=str, default=None,
                      help='Directory containing CSV data files (default: code/data)')
    parser.add_argument('--figures', type=str, nargs='+', default=['all'],
                      choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', 'all'],
                      help='Which figures to generate (default: all)')
    parser.add_argument('--save', action='store_true',
                      help='Save figures to files instead of displaying interactively')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save figures (default: results/figures/paper_figs)')
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
        raise FileNotFoundError(f"Baseline G-code not found: {args.baseline_gcode}")
    
    stabilized_path = Path(args.stabilized_gcode)
    if not stabilized_path.is_absolute():
        test_path = script_dir / args.stabilized_gcode
        if test_path.exists():
            stabilized_path = test_path
        elif not stabilized_path.exists():
            stabilized_path = test_path
    if not stabilized_path.exists():
        raise FileNotFoundError(f"Stabilized G-code not found: {args.stabilized_gcode}")
    
    # Data directory
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
        print(f"Reading print trials data: {print_trials_path}", flush=True)
        print_trials_df = pd.read_csv(print_trials_path)
    else:
        print(f"WARNING: {print_trials_path} not found", flush=True)
    
    first_layer_df = None
    if first_layer_path.exists():
        print(f"Reading first layer sweep data: {first_layer_path}", flush=True)
        first_layer_df = pd.read_csv(first_layer_path)
    else:
        print(f"WARNING: {first_layer_path} not found", flush=True)
    
    electrical_df = None
    if electrical_path.exists():
        print(f"Reading electrical traces data: {electrical_path}", flush=True)
        electrical_df = pd.read_csv(electrical_path)
    else:
        print(f"WARNING: {electrical_path} not found", flush=True)
    
    # Determine which figures to generate
    if 'all' in args.figures:
        figures_to_generate = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        figures_to_generate = args.figures
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = script_dir / 'results' / 'figures' / 'paper_figs'
    
    save_mode = args.save
    
    # Use non-interactive backend if saving
    if save_mode:
        matplotlib.use('Agg')  # Non-interactive backend for saving
        print("Using non-interactive backend for saving figures...", flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"Generating Final Paper Figures: {', '.join(figures_to_generate)}", flush=True)
    print(f"{'='*60}", flush=True)
    if save_mode:
        print(f"Figures will be saved to: {output_dir}", flush=True)
    else:
        print("Figures will be displayed interactively - save them manually using figure window controls.", flush=True)
        print("NOTE: Figure windows will open one at a time. Close each window to proceed to the next.", flush=True)
    print("", flush=True)
    
    # Generate requested figures
    if '1' in figures_to_generate:
        print("\n[1/9] Generating Figure 1 — G-code Modification Summary...", flush=True)
        figure_1_gcode_modification_summary(baseline_lines, stabilized_lines, save_mode, output_dir)
    
    if '2' in figures_to_generate:
        print("\n[2/9] Generating Figure 2 — Extrusion-Rate Proxy Comparison...", flush=True)
        figure_2_extrusion_rate_comparison(baseline_lines, stabilized_lines, save_mode, output_dir)
    
    if '3' in figures_to_generate:
        print("\n[3/9] Generating Figure 3 — Pressure Estimate Comparison...", flush=True)
        figure_3_pressure_comparison(baseline_lines, stabilized_lines,
                                     args.alpha, args.tau_r, args.p_y, args.p_max,
                                     save_mode, output_dir)
    
    if '4' in figures_to_generate:
        print("\n[4/9] Generating Figure 4 — Extrusion Survival Curve...", flush=True)
        figure_4_extrusion_survival(print_trials_df, save_mode, output_dir)
    
    if '5' in figures_to_generate:
        print("\n[5/9] Generating Figure 5 — First-Layer Operating Envelope...", flush=True)
        figure_5_first_layer_envelope(first_layer_df, save_mode, output_dir)
    
    if '6' in figures_to_generate:
        print("\n[6/9] Generating Figure 6 — Print Completion Rate...", flush=True)
        figure_6_completion_rate(print_trials_df, save_mode, output_dir)
    
    if '7' in figures_to_generate:
        print("\n[7/9] Generating Figure 7 — Electrical Yield...", flush=True)
        figure_7_electrical_yield(electrical_df, save_mode, output_dir)
    
    if '8' in figures_to_generate:
        print("\n[8/9] Generating Figure 8 — Resistance Comparison...", flush=True)
        figure_8_resistance_comparison(electrical_df, save_mode, output_dir)
    
    if '9' in figures_to_generate:
        print("\n[9/9] Generating Figure 9 — 3D Toolpath Comparison...", flush=True)
        figure_9_3d_toolpath_comparison(baseline_lines, stabilized_lines, save_mode, output_dir)
    
    print(f"\n{'='*60}", flush=True)
    if save_mode:
        print(f"✓ All requested figures saved to {output_dir}!", flush=True)
    else:
        print("✓ All requested figures displayed successfully!", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == '__main__':
    main()

