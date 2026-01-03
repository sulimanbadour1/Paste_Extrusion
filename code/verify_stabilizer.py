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

# All plotting functions removed - plotting code deleted from this file

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
    # Plotting functionality removed
    if not args.no_plots:
        print("\nNote: Plotting functionality has been removed from this tool.")
        print("      Use verification checks above to validate stabilization.")

    # Final summary hint
    print("\nNotes:")
    print("- If your input contains mostly negative E moves, the stabilizer will remove many of them;")
    print("  this indicates the slicer output is not paste-compatible and must be reconfigured.")
    print("- For a full shaping test, ensure the input has positive E deposition moves (G1 ... E+...).")

if __name__ == "__main__":
    main()
