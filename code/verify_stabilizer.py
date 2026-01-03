#!/usr/bin/env python3
"""
verify_stabilizer.py

Verification / QA script for paste_stabilizer_v2 output.

What it verifies:
1) Stabilization header inserted
2) Modes set as expected (G90 + M83 present near header)
3) Retraction suppression occurred if input contains negative E
4) Output contains no negative E extrusion moves (excluding comments)
5) Evidence of shaping:
   - feed scaling present on extrusion moves (F tokens rewritten)
   - OR actions in CSV log: low_prime / relax_dwell
6) CSV log sanity:
   - p_hat present and numeric
   - p_hat not NaN; compute % time within bounds

This is not a physics proof; it is a reproducibility and invariants check.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MOVE_RE = re.compile(r"^(G0|G1)\s", re.IGNORECASE)

def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()

def strip_comment(line: str) -> str:
    if ";" in line:
        return line.split(";", 1)[0].strip()
    return line.strip()

def parse_float_token(line: str, letter: str) -> Optional[float]:
    m = re.search(rf"{letter}([-+]?\d*\.?\d+)", line, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))

def is_move(code: str) -> bool:
    return bool(MOVE_RE.match(code.strip()))

def find_header(lines: List[str]) -> int:
    for i, ln in enumerate(lines):
        if "Paste Stabilization Layer" in ln:
            return i
    return -1

def count_negative_e_moves(lines: List[str]) -> int:
    c = 0
    for ln in lines:
        code = strip_comment(ln)
        if not code:
            continue
        if is_move(code):
            e = parse_float_token(code, "E")
            if e is not None and e < 0:
                c += 1
    return c

def any_negative_e_moves(lines: List[str]) -> List[Tuple[int, str]]:
    hits = []
    for i, ln in enumerate(lines, start=1):
        code = strip_comment(ln)
        if not code:
            continue
        if is_move(code):
            e = parse_float_token(code, "E")
            if e is not None and e < 0:
                hits.append((i, ln))
    return hits

def count_retraction_suppressed(lines: List[str]) -> int:
    return sum(1 for ln in lines if "RETRACTION SUPPRESSED" in ln)

def has_mode_near_header(lines: List[str], header_idx: int, token: str, window: int = 60) -> bool:
    if header_idx < 0:
        return False
    start = max(0, header_idx)
    end = min(len(lines), header_idx + window)
    block = "\n".join(lines[start:end]).upper()
    return token.upper() in block

def has_feed_rewrite_evidence(out_lines: List[str]) -> bool:
    """
    Heuristic: find extrusion moves with explicit F tokens.
    If shaping occurred, many extrusion lines will include F
    (even if original did not), and/or there will be comments about p_hat actions.
    """
    extrusion_with_f = 0
    extrusion_total = 0
    for ln in out_lines:
        code = strip_comment(ln)
        if not code:
            continue
        if is_move(code):
            e = parse_float_token(code, "E")
            if e is not None and e > 0:
                extrusion_total += 1
                f = parse_float_token(code, "F")
                if f is not None:
                    extrusion_with_f += 1
    # If there are few extrusion moves, don't force this check
    if extrusion_total < 5:
        return False
    return extrusion_with_f / max(extrusion_total, 1) > 0.5

def print_result(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    if detail:
        print(f"[{status}] {name}: {detail}")
    else:
        print(f"[{status}] {name}")

def resolve_results_path(file_path: Path) -> Path:
    """
    Resolve a file path, checking the results directory if the file doesn't exist
    at the specified location. If the path is just a filename (no directory),
    check results folder first.
    """
    # First, check if the original path exists (handles paths like code/test.gcode)
    if file_path.exists():
        return file_path
    
    # If it's just a filename (no parent directory), check results first
    if not file_path.parent or file_path.parent == Path("."):
        results_dir = Path("results")
        results_path = results_dir / file_path.name
        if results_path.exists():
            return results_path
    
    # Fallback: check in results directory (for output files)
    results_dir = Path("results")
    results_path = results_dir / file_path.name
    if results_path.exists():
        return results_path
    
    # Return original path (will raise error later if not found)
    return file_path

def extract_extrusion_data(lines: List[str]) -> Tuple[List[float], List[float]]:
    """Extract E values and their approximate positions (line numbers) from G-code."""
    e_values = []
    positions = []
    for i, ln in enumerate(lines):
        code = strip_comment(ln)
        if not code:
            continue
        if is_move(code):
            e = parse_float_token(code, "E")
            if e is not None:
                e_values.append(e)
                positions.append(i)
    return positions, e_values

def extract_feed_rates(lines: List[str]) -> Tuple[List[float], List[float]]:
    """Extract F (feed rate) values and their positions."""
    f_values = []
    positions = []
    for i, ln in enumerate(lines):
        code = strip_comment(ln)
        if not code:
            continue
        if is_move(code):
            f = parse_float_token(code, "F")
            if f is not None:
                f_values.append(f)
                positions.append(i)
    return positions, f_values

def extract_3d_toolpath(lines: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 3D toolpath (X, Y, Z) and extrusion rates from G-code.
    Returns: (coordinates array [Nx3], E values array [N], feed rates array [N], segment_flags array [N], retraction_flags array [N])
    segment_flags: True if this point starts an extrusion segment, False for travel
    retraction_flags: True if this point is a retraction (negative E)
    """
    coords = []
    e_values = []
    feed_rates = []
    segment_flags = []  # True = extrusion move, False = travel move
    retraction_flags = []  # True = retraction move
    
    # Track current position (for absolute mode)
    x_curr, y_curr, z_curr = None, None, 0.0
    e_cumulative = 0.0
    is_relative_e = True  # Default to relative (M83) for paste printing
    f_curr = None
    
    for ln in lines:
        code = strip_comment(ln)
        if not code:
            continue
        
        # Check for M83 (relative extrusion) or M82 (absolute extrusion)
        if "M83" in code.upper():
            is_relative_e = True
        elif "M82" in code.upper():
            is_relative_e = False
        
        # Check for G28/G92 which might reset positions
        if "G28" in code.upper():  # Home
            x_curr, y_curr, z_curr = None, None, 0.0
        elif "G92" in code.upper():  # Set position
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
            
            # Track if this move changes position
            x_prev, y_prev, z_prev = x_curr, y_curr, z_curr
            
            # Update current position
            if x is not None:
                x_curr = x
            if y is not None:
                y_curr = y
            if z is not None:
                z_curr = z
            if f is not None:
                f_curr = f
            
            # Handle extrusion
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
                has_extrusion = (e_delta > 1e-6)  # Positive extrusion
                is_retraction = (e_delta < -1e-6)  # Negative extrusion (retraction)
            
            # Check for position change (including Z-only moves)
            has_position_change = (x is not None or y is not None or z is not None) and \
                                 (x_prev != x_curr or y_prev != y_curr or z_prev != z_curr)
            
            # Use last known X/Y if current move doesn't specify them (for Z-only or E-only moves)
            x_to_use = x_curr if x_curr is not None else x_prev
            y_to_use = y_curr if y_curr is not None else y_prev
            z_to_use = z_curr if z_curr is not None else (z_prev if z_prev is not None else 0.0)
            
            # Record moves if:
            # 1. We have valid X/Y coordinates (current or previous), AND
            # 2. There's a position change OR any E value (extrusion or retraction)
            should_record = False
            if x_to_use is not None and y_to_use is not None:
                # We have valid coordinates
                if has_position_change or e is not None:
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
                segment_flags.append(has_extrusion)  # Only positive extrusion flags as extrusion segment
                retraction_flags.append(is_retraction)
    
    if len(coords) == 0:
        return np.array([]).reshape(0, 3), np.array([]), np.array([]), np.array([], dtype=bool), np.array([], dtype=bool)
    
    return np.array(coords), np.array(e_values), np.array(feed_rates), np.array(segment_flags), np.array(retraction_flags)

def compute_extrusion_rate(coords: np.ndarray, e_values: np.ndarray, feed_rates: np.ndarray) -> np.ndarray:
    """
    Compute extrusion rate (E per unit distance) along the toolpath.
    """
    if len(coords) < 2:
        return np.zeros_like(e_values)
    
    # Compute distances between consecutive points
    distances = np.zeros(len(coords))
    distances[1:] = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    
    # Compute extrusion rate (E per mm)
    rates = np.zeros_like(e_values)
    mask = distances > 1e-6  # Avoid division by zero
    rates[mask] = np.abs(e_values[mask]) / distances[mask]
    
    return rates

def plot_3d_toolpath_comparison(in_lines: List[str], out_lines: List[str], 
                                  in_df: Optional[pd.DataFrame], out_df: Optional[pd.DataFrame],
                                  fig_dir: Path) -> None:
    """Create 3D visualization comparing input and output toolpaths with extrusion rates."""
    
    # Extract toolpaths
    in_coords, in_e, in_f, in_flags, in_retract = extract_3d_toolpath(in_lines)
    out_coords, out_e, out_f, out_flags, out_retract = extract_3d_toolpath(out_lines)
    
    if len(in_coords) == 0 and len(out_coords) == 0:
        print("Warning: No valid 3D coordinates found in G-code")
        return
    
    # Debug output
    print(f"  Extracted toolpaths: Input={len(in_coords)} points, Output={len(out_coords)} points")
    print(f"  Extrusion moves: Input={(in_flags).sum()}, Output={(out_flags).sum()}")
    print(f"  Retraction moves: Input={(in_retract).sum()}, Output={(out_retract).sum()}")
    print(f"  Extrusion values: Input E>0: {(in_e > 0).sum()}, Output E>0: {(out_e > 0).sum()}")
    
    # Warn if input has no extrusion
    if (in_flags).sum() == 0:
        print("  NOTE: Input has no positive extrusion moves (only retractions/travel).")
        print("        This is expected if input G-code only contains retractions.")
    
    # Compute extrusion rates only for extrusion moves
    in_rates = compute_extrusion_rate(in_coords, in_e, in_f)
    out_rates = compute_extrusion_rate(out_coords, out_e, out_f)
    
    # Debug: check for issues
    if (out_e > 0).sum() == 0:
        print("  WARNING: No positive E values found in output! This suggests a problem.")
    if len(out_coords) < len(in_coords) * 0.5:
        print(f"  WARNING: Output has significantly fewer points ({len(out_coords)} vs {len(in_coords)})")
    
    # Normalize rates for color mapping (use same scale for both)
    all_rates = np.concatenate([in_rates[in_rates > 0], out_rates[out_rates > 0]])
    if len(all_rates) > 0:
        vmin, vmax = 0, np.percentile(all_rates, 95)  # Use 95th percentile to avoid outliers
    else:
        vmin, vmax = 0, 1
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 8))
    
    # Input toolpath
    ax1 = fig.add_subplot(121, projection='3d')
    if len(in_coords) > 0:
        has_extrusion = in_flags.any()
        
        # Plot extrusion segments (only connect consecutive extrusion moves)
        i = 0
        while i < len(in_coords) - 1:
            if in_flags[i] and in_flags[i+1]:
                # Both points are extrusion - connect them
                rate = (in_rates[i] + in_rates[i+1]) / 2
                if rate > 1e-6:
                    color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    ax1.plot([in_coords[i, 0], in_coords[i+1, 0]],
                            [in_coords[i, 1], in_coords[i+1, 1]],
                            [in_coords[i, 2], in_coords[i+1, 2]],
                            color=plt.cm.viridis(color_val), linewidth=2.5, alpha=0.9)
            elif not in_flags[i] and not in_flags[i+1]:
                # Both are travel - draw gray line (more visible if no extrusion)
                alpha_val = 0.4 if not has_extrusion else 0.15
                linewidth_val = 1.0 if not has_extrusion else 0.3
                ax1.plot([in_coords[i, 0], in_coords[i+1, 0]],
                        [in_coords[i, 1], in_coords[i+1, 1]],
                        [in_coords[i, 2], in_coords[i+1, 2]],
                        'gray', alpha=alpha_val, linewidth=linewidth_val)
            # Mixed (extrusion to travel or vice versa) - don't connect
            i += 1
        
        # Add scatter points for extrusion moves (or travel if no extrusion)
        if has_extrusion:
            scatter1 = ax1.scatter(in_coords[in_flags, 0], 
                                   in_coords[in_flags, 1], 
                                   in_coords[in_flags, 2],
                                   c=in_rates[in_flags], cmap='viridis', 
                                   s=25, alpha=0.95, edgecolors='black', linewidths=0.5,
                                   vmin=vmin, vmax=vmax, label='Extrusion')
        else:
            # Show travel moves as gray points if no extrusion
            scatter1 = ax1.scatter(in_coords[:, 0], in_coords[:, 1], in_coords[:, 2],
                                   c='gray', s=10, alpha=0.5, edgecolors='black', linewidths=0.3)
        
        # Show retractions with red markers
        if in_retract.any():
            # Draw lines to retraction points
            for i in range(len(in_coords) - 1):
                if in_retract[i] or in_retract[i+1]:
                    ax1.plot([in_coords[i, 0], in_coords[i+1, 0]],
                            [in_coords[i, 1], in_coords[i+1, 1]],
                            [in_coords[i, 2], in_coords[i+1, 2]],
                            'r--', alpha=0.6, linewidth=1.5)
            # Mark retraction points
            ax1.scatter(in_coords[in_retract, 0], 
                       in_coords[in_retract, 1], 
                       in_coords[in_retract, 2],
                       c='red', marker='x', s=100, alpha=0.9, linewidths=2,
                       label='Retractions', zorder=10)
        
        ax1.set_xlabel('X (mm)', fontsize=10)
        ax1.set_ylabel('Y (mm)', fontsize=10)
        ax1.set_zlabel('Z (mm)', fontsize=10)
        if has_extrusion:
            retract_info = f", {in_retract.sum()} retractions" if in_retract.any() else ""
            title = f'Input G-code: 3D Toolpath\n(Color = Extrusion Rate, {in_flags.sum()}/{len(in_coords)} extrusion moves{retract_info})'
        else:
            retract_info = f" ({in_retract.sum()} retractions)" if in_retract.any() else ""
            title = f'Input G-code: 3D Toolpath\n(Travel moves only - no extrusion{retract_info}, {len(in_coords)} points)'
        ax1.set_title(title, fontsize=11)
        if has_extrusion:
            cbar1 = plt.colorbar(scatter1, ax=ax1, label='Extrusion Rate (E/mm)', shrink=0.6)
            cbar1.ax.tick_params(labelsize=8)
        if in_retract.any():
            ax1.legend(loc='upper right', fontsize=8)
    
    # Output toolpath
    ax2 = fig.add_subplot(122, projection='3d')
    if len(out_coords) > 0:
        # Plot extrusion segments (only connect consecutive extrusion moves)
        i = 0
        while i < len(out_coords) - 1:
            if out_flags[i] and out_flags[i+1]:
                # Both points are extrusion - connect them
                rate = (out_rates[i] + out_rates[i+1]) / 2
                if rate > 1e-6:
                    color_val = np.clip((rate - vmin) / (vmax - vmin + 1e-6), 0, 1)
                    ax2.plot([out_coords[i, 0], out_coords[i+1, 0]],
                            [out_coords[i, 1], out_coords[i+1, 1]],
                            [out_coords[i, 2], out_coords[i+1, 2]],
                            color=plt.cm.viridis(color_val), linewidth=2.5, alpha=0.9)
            elif not out_flags[i] and not out_flags[i+1]:
                # Both are travel - draw thin gray line
                ax2.plot([out_coords[i, 0], out_coords[i+1, 0]],
                        [out_coords[i, 1], out_coords[i+1, 1]],
                        [out_coords[i, 2], out_coords[i+1, 2]],
                        'lightgray', alpha=0.15, linewidth=0.3)
            # Mixed (extrusion to travel or vice versa) - don't connect
            i += 1
        
        # Add scatter points only for extrusion moves
        if out_flags.any():
            scatter2 = ax2.scatter(out_coords[out_flags, 0], 
                                  out_coords[out_flags, 1], 
                                  out_coords[out_flags, 2],
                                  c=out_rates[out_flags], cmap='viridis', 
                                  s=25, alpha=0.95, edgecolors='black', linewidths=0.5,
                                  vmin=vmin, vmax=vmax)
        
        ax2.set_xlabel('X (mm)', fontsize=10)
        ax2.set_ylabel('Y (mm)', fontsize=10)
        ax2.set_zlabel('Z (mm)', fontsize=10)
        title = f'Output G-code: Stabilized Toolpath\n(Color = Extrusion Rate, {out_flags.sum()}/{len(out_coords)} extrusion moves)'
        ax2.set_title(title, fontsize=11)
        if out_flags.any():
            cbar2 = plt.colorbar(scatter2, ax=ax2, label='Extrusion Rate (E/mm)', shrink=0.6)
            cbar2.ax.tick_params(labelsize=8)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_3d_toolpath_comparison.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_3d_toolpath_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3D toolpath comparison: {fig_dir}/verification_3d_toolpath_comparison.pdf")

def plot_3d_toolpath_with_pressure(out_lines: List[str], df: pd.DataFrame, 
                                    p_y: float, p_max: float, fig_dir: Path) -> None:
    """Create 3D toolpath colored by pressure state."""
    
    out_coords, out_e, out_f, out_flags, out_retract = extract_3d_toolpath(out_lines)
    
    if len(out_coords) == 0:
        return
    
    # Match pressure states to coordinates (approximate by time)
    valid = df.dropna(subset=["p_hat", "t_s"]).copy()
    if len(valid) == 0:
        return
    
    valid["p_hat"] = pd.to_numeric(valid["p_hat"], errors="coerce")
    valid = valid.dropna(subset=["p_hat"])
    
    # Interpolate pressure to match coordinate count
    # This is approximate - assumes uniform time distribution
    if len(valid) > 0 and len(out_coords) > 0:
        # Simple mapping: assume coordinates are roughly evenly distributed in time
        t_coords = np.linspace(valid["t_s"].min(), valid["t_s"].max(), len(out_coords))
        p_interp = np.interp(t_coords, valid["t_s"].values, valid["p_hat"].values)
    else:
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot connected path colored by pressure (only extrusion segments)
    vmin, vmax = 0, p_max * 1.1
    for i in range(len(out_coords) - 1):
        if out_flags[i] and out_flags[i+1]:
            # Both are extrusion moves - connect them
            p_avg = (p_interp[i] + p_interp[i+1]) / 2
            color_val = np.clip((p_avg - vmin) / (vmax - vmin + 1e-6), 0, 1)
            ax.plot([out_coords[i, 0], out_coords[i+1, 0]],
                    [out_coords[i, 1], out_coords[i+1, 1]],
                    [out_coords[i, 2], out_coords[i+1, 2]],
                    color=plt.cm.RdYlGn(color_val), linewidth=2.0, alpha=0.8)
    
    # Add scatter points only for extrusion moves
    if out_flags.any():
        scatter = ax.scatter(out_coords[out_flags, 0], out_coords[out_flags, 1], out_coords[out_flags, 2],
                            c=p_interp[out_flags], cmap='RdYlGn', s=25, alpha=0.9,
                            edgecolors='black', linewidths=0.5,
                            vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_zlabel('Z (mm)', fontsize=11)
    ax.set_title('3D Toolpath Colored by Estimated Pressure State\n(Green=Low, Yellow=Mid, Red=High)', fontsize=12)
    
    if out_flags.any():
        cbar = plt.colorbar(scatter, ax=ax, label='Estimated Pressure (p_hat)', shrink=0.7)
        cbar.ax.tick_params(labelsize=9)
        # Add reference lines to colorbar
        y_norm_py = (p_y - vmin) / (vmax - vmin)
        y_norm_pmax = (p_max - vmin) / (vmax - vmin)
        cbar.ax.axhline(y_norm_py, color='orange', linestyle='--', linewidth=2)
        cbar.ax.axhline(y_norm_pmax, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_3d_toolpath_pressure.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_3d_toolpath_pressure.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 3D toolpath with pressure: {fig_dir}/verification_3d_toolpath_pressure.pdf")

def plot_pressure_trace(df: pd.DataFrame, p_y: float, p_max: float, fig_dir: Path) -> None:
    """Plot pressure state over time with bounds and actions."""
    valid = df.dropna(subset=["p_hat"]).copy()
    if len(valid) == 0:
        return
    
    valid["p_hat"] = pd.to_numeric(valid["p_hat"], errors="coerce")
    valid = valid.dropna(subset=["p_hat"])
    
    t = valid["t_s"].values
    p = valid["p_hat"].values
    
    plt.figure(figsize=(10, 6))
    plt.plot(t, p, label="p_hat(t)", linewidth=1.5, alpha=0.8)
    plt.axhline(p_y, linestyle="--", color="orange", label=f"p_y (yield) = {p_y}", linewidth=2)
    plt.axhline(p_max, linestyle="--", color="red", label=f"p_max (upper bound) = {p_max}", linewidth=2)
    
    # Highlight actions
    low_prime_mask = valid["action"] == "low_prime"
    relax_mask = valid["action"] == "relax_dwell"
    retract_mask = valid["action"] == "retract_suppressed"
    
    if low_prime_mask.any():
        plt.scatter(t[low_prime_mask], p[low_prime_mask], color="green", 
                   marker="^", s=50, label="low_prime", zorder=5)
    if relax_mask.any():
        plt.scatter(t[relax_mask], p[relax_mask], color="red", 
                   marker="v", s=50, label="relax_dwell", zorder=5)
    if retract_mask.any():
        plt.scatter(t[retract_mask], p[retract_mask], color="purple", 
                   marker="x", s=50, label="retract_suppressed", zorder=5)
    
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated pressure (arb. units)")
    plt.title("Pressure State Over Time with Stabilization Actions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_pressure_trace.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_pressure_trace.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pressure trace plot: {fig_dir}/verification_pressure_trace.pdf")

def plot_extrusion_comparison(in_lines: List[str], out_lines: List[str], fig_dir: Path) -> None:
    """Plot extrusion (E) values comparison between input and output."""
    in_pos, in_e = extract_extrusion_data(in_lines)
    out_pos, out_e = extract_extrusion_data(out_lines)
    
    if len(in_e) == 0 and len(out_e) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Input extrusion
    if len(in_e) > 0:
        ax1.plot(in_pos, in_e, 'b-', linewidth=1, alpha=0.7, label="Input E values")
        ax1.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax1.fill_between(in_pos, 0, in_e, where=np.array(in_e) < 0, 
                         color='red', alpha=0.3, label="Retractions (negative E)")
        ax1.fill_between(in_pos, 0, in_e, where=np.array(in_e) > 0, 
                         color='blue', alpha=0.3, label="Extrusion (positive E)")
    ax1.set_ylabel("Extrusion (E)")
    ax1.set_title("Input G-code: Extrusion Values")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Output extrusion
    if len(out_e) > 0:
        ax2.plot(out_pos, out_e, 'g-', linewidth=1, alpha=0.7, label="Output E values")
        ax2.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax2.fill_between(out_pos, 0, out_e, where=np.array(out_e) > 0, 
                         color='green', alpha=0.3, label="Extrusion (positive E)")
    ax2.set_xlabel("Line Number")
    ax2.set_ylabel("Extrusion (E)")
    ax2.set_title("Output G-code: Extrusion Values (Retractions Removed)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_extrusion_comparison.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_extrusion_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved extrusion comparison plot: {fig_dir}/verification_extrusion_comparison.pdf")

def plot_feed_rate_changes(out_lines: List[str], df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot feed rate changes and scaling factors."""
    if len(df) == 0:
        return
    
    valid = df.dropna(subset=["feed_scale"]).copy()
    if len(valid) == 0:
        return
    
    valid["feed_scale"] = pd.to_numeric(valid["feed_scale"], errors="coerce")
    valid = valid.dropna(subset=["feed_scale"])
    
    if len(valid) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Feed scale over time
    t = valid["t_s"].values
    feed_scale = valid["feed_scale"].values
    
    ax1.plot(t, feed_scale, 'b-', linewidth=1.5, alpha=0.7, label="Feed scale factor")
    ax1.axhline(1.0, color='k', linestyle='--', linewidth=1, label="No scaling (1.0)")
    ax1.set_ylabel("Feed Scale Factor")
    ax1.set_title("Feed Rate Scaling Applied by Stabilizer")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of feed scales
    ax2.hist(feed_scale, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(1.0, color='r', linestyle='--', linewidth=2, label="No scaling (1.0)")
    ax2.set_xlabel("Feed Scale Factor")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Distribution of Feed Rate Scaling")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_feed_scaling.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_feed_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feed scaling plot: {fig_dir}/verification_feed_scaling.pdf")

def plot_action_summary(df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot summary of actions taken by the stabilizer."""
    if len(df) == 0:
        return
    
    actions = df["action"].astype(str).value_counts()
    
    if len(actions) == 0:
        return
    
    plt.figure(figsize=(10, 6))
    colors = {'low_prime': 'green', 'relax_dwell': 'red', 'retract_suppressed': 'purple', 
              'emit': 'blue', 'default': 'gray'}
    bar_colors = [colors.get(action, colors['default']) for action in actions.index]
    
    bars = plt.bar(actions.index, actions.values, color=bar_colors, alpha=0.7, edgecolor='black')
    plt.xlabel("Action Type")
    plt.ylabel("Count")
    plt.title("Stabilizer Actions Summary")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(fig_dir / "verification_action_summary.pdf", bbox_inches="tight")
    plt.savefig(fig_dir / "verification_action_summary.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved action summary plot: {fig_dir}/verification_action_summary.pdf")

def main():
    ap = argparse.ArgumentParser(description="Verify paste stabilizer output.")
    ap.add_argument("--in", dest="infile", required=True, help="Input G-code")
    ap.add_argument("--out", dest="outfile", required=True, help="Output stabilized G-code")
    ap.add_argument("--csv", dest="csvfile", required=False, help="CSV log file (run_log.csv)")
    ap.add_argument("--p_y", type=float, default=5.0, help="Yield threshold used in estimator")
    ap.add_argument("--p_max", type=float, default=14.0, help="Upper pressure bound used in estimator")
    ap.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = ap.parse_args()

    # Check results folder for all files
    in_path = resolve_results_path(Path(args.infile))
    out_path = resolve_results_path(Path(args.outfile))

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")
    if not out_path.exists():
        raise SystemExit(f"Output not found: {out_path}")

    in_lines = read_lines(in_path)
    out_lines = read_lines(out_path)

    # 1) Header
    header_idx = find_header(out_lines)
    print_result("Header inserted", header_idx >= 0, f"line={header_idx+1}" if header_idx >= 0 else "")

    # 2) Modes near header
    g90_ok = has_mode_near_header(out_lines, header_idx, "G90")
    m83_ok = has_mode_near_header(out_lines, header_idx, "M83")
    print_result("G90 near header", g90_ok)
    print_result("M83 near header", m83_ok)

    # 3) Retraction suppression
    in_neg = count_negative_e_moves(in_lines)
    out_neg_hits = any_negative_e_moves(out_lines)
    suppressed_count = count_retraction_suppressed(out_lines)

    if in_neg > 0:
        print_result("Input had negative-E moves", True, f"count={in_neg}")
        # Expect suppression evidence
        print_result("Retraction suppression evidence", suppressed_count > 0, f"suppressed={suppressed_count}")
    else:
        print_result("Input had negative-E moves", False, "count=0 (not a retraction stress test)")

    # 4) Output should not contain negative E moves
    # Some edge-cases: If user wants to keep specific negative moves, you'd relax this.
    no_neg_out = len(out_neg_hits) == 0
    if not no_neg_out:
        sample = "\n".join([f"  line {i}: {ln}" for i, ln in out_neg_hits[:5]])
        print_result("No negative-E moves in output", False, f"Found {len(out_neg_hits)}. Sample:\n{sample}")
    else:
        print_result("No negative-E moves in output", True)

    # 5) Evidence of shaping
    feed_evidence = has_feed_rewrite_evidence(out_lines)
    comment_evidence = any("p_hat=" in ln for ln in out_lines)
    print_result("Shaping evidence in G-code", feed_evidence or comment_evidence,
                 f"feed_rewrite={feed_evidence}, p_hat_comments={comment_evidence}")

    # 6) CSV checks (optional)
    if args.csvfile:
        csv_path = resolve_results_path(Path(args.csvfile))
        if not csv_path.exists():
            print_result("CSV log exists", False, f"missing: {csv_path}")
        else:
            print_result("CSV log exists", True, str(csv_path))
            df = pd.read_csv(csv_path)
            required_cols = {"t_s", "p_hat", "action"}
            cols_ok = required_cols.issubset(set(df.columns))
            print_result("CSV has required columns", cols_ok, f"needed={sorted(required_cols)}")

            if cols_ok:
                df["p_hat"] = pd.to_numeric(df["p_hat"], errors="coerce")
                valid = df.dropna(subset=["p_hat"]).copy()
                if len(valid) == 0:
                    print_result("CSV p_hat numeric", False, "no valid numeric p_hat rows")
                else:
                    print_result("CSV p_hat numeric", True, f"rows={len(valid)}")

                    p_y = args.p_y
                    p_max = args.p_max
                    within = ((valid["p_hat"] > p_y) & (valid["p_hat"] < p_max)).mean()
                    low = (valid["p_hat"] <= p_y).mean()
                    high = (valid["p_hat"] >= p_max).mean()

                    # Interventions
                    actions = valid["action"].astype(str).value_counts().to_dict()
                    low_prime = actions.get("low_prime", 0)
                    relax = actions.get("relax_dwell", 0)

                    print_result("Pressure window compliance (informational)", True,
                                 f"within={within*100:.1f}%, low={low*100:.1f}%, high={high*100:.1f}%, "
                                 f"low_prime={low_prime}, relax_dwell={relax}")

    # Generate plots
    if not args.no_plots:
        fig_dir = Path("results") / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating verification plots...")
        print("="*60)
        print(f"Input file: {in_path}")
        print(f"Output file: {out_path}")
        print(f"Input lines: {len(in_lines)}, Output lines: {len(out_lines)}")
        
        # Plot 1: Extrusion comparison (always available)
        try:
            plot_extrusion_comparison(in_lines, out_lines, fig_dir)
        except Exception as e:
            print(f"Warning: Could not generate extrusion comparison plot: {e}")
        
        # Plot 2-4: Require CSV data
        df_plot = None
        if args.csvfile:
            csv_path = resolve_results_path(Path(args.csvfile))
            if csv_path.exists():
                try:
                    df_plot = pd.read_csv(csv_path)
                    print(f"CSV file: {csv_path} ({len(df_plot)} rows)")
                    plot_pressure_trace(df_plot, args.p_y, args.p_max, fig_dir)
                    plot_feed_rate_changes(out_lines, df_plot, fig_dir)
                    plot_action_summary(df_plot, fig_dir)
                except Exception as e:
                    print(f"Warning: Could not generate CSV-based plots: {e}")
        
        # Plot 5-6: 3D toolpath visualizations
        print("\nGenerating 3D toolpath visualizations...")
        try:
            plot_3d_toolpath_comparison(in_lines, out_lines, None, df_plot, fig_dir)
            print("  ✓ 3D toolpath comparison (input vs output)")
        except Exception as e:
            print(f"Warning: Could not generate 3D toolpath comparison: {e}")
        
        if df_plot is not None:
            try:
                plot_3d_toolpath_with_pressure(out_lines, df_plot, args.p_y, args.p_max, fig_dir)
                print("  ✓ 3D toolpath with pressure coloring")
            except Exception as e:
                print(f"Warning: Could not generate 3D pressure toolpath: {e}")
        
        print(f"\nAll plots saved to: {fig_dir}/")

    # Final summary hint
    print("\nNotes:")
    print("- If your input contains mostly negative E moves, the stabilizer will remove many of them;")
    print("  this indicates the slicer output is not paste-compatible and must be reconfigured.")
    print("- For a full shaping test, ensure the input has positive E deposition moves (G1 ... E+...).")

if __name__ == "__main__":
    main()
