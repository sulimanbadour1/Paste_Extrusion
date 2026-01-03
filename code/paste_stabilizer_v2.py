#!/usr/bin/env python3
"""
Software-defined pressure/flow stabilization middleware for low-cost paste extrusion printers.

Usage:
    python3 paste_stabilizer_v2.py --in input.gcode --out stabilized.gcode --csv run_log.csv --log changes.log

Key upgrades (implements the paper's "pressure model shaping"):
1) Discrete-time pressure estimator:
      p_hat[k+1] = p_hat[k] + Ts*(alpha*u[k] - p_hat[k]/tau_r)
   where u[k] is an extrusion-rate proxy derived from G-code moves.

2) Command shaping to enforce pressure window:
      p_y < p_hat < p_max
   via:
   - rate limiting of u (avoids impulsive pressure steps)
   - adaptive feed scaling for extrusion segments
   - dwell insertion when p_hat is too high (pressure relaxation)
   - automatic priming injection when p_hat is too low

3) Retraction suppression kept (negative E moves replaced by dwell + micro-prime).

Outputs:
- stabilized G-code
- CSV log for plots (p_hat, u, actions, etc.)

Assumptions:
- Paste extrusion mapped to E axis.
- Relative extrusion (M83) is enforced (recommended for paste).
- XY moves are absolute (G90).
- G1 with X/Y and E is the dominant deposition command.

This is open-loop "software-defined control" consistent with the paper.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

MOVE_RE = re.compile(r"^(G0|G1)\s", re.IGNORECASE)

def parse_float_token(line: str, letter: str) -> Optional[float]:
    m = re.search(rf"{letter}([-+]?\d*\.?\d+)", line, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))

def strip_comment(line: str) -> Tuple[str, str]:
    if ";" not in line:
        return line.rstrip("\n"), ""
    code, comment = line.split(";", 1)
    return code.rstrip("\n").rstrip(), comment.rstrip("\n")

def is_move(code: str) -> bool:
    return bool(MOVE_RE.match(code.strip()))

def is_retraction_move(code: str) -> bool:
    if not is_move(code):
        return False
    e = parse_float_token(code, "E")
    return (e is not None) and (e < 0.0)

def has_extrusion(code: str) -> bool:
    if not is_move(code):
        return False
    return parse_float_token(code, "E") is not None

def gcode_comment(text: str) -> str:
    return f"; {text}"

@dataclasses.dataclass
class StabilizerConfig:
    # ---- Kinematics/time assumptions ----
    # If an extrusion move is missing XY length (pure E prime), we approximate dt from E and feed.
    min_dt_s: float = 0.02

    # ---- Pressure estimator (discrete) ----
    Ts: float = 0.10             # estimator sampling interval (s) used internally per processed move
    alpha: float = 1.0           # pressure gain (arbitrary units per u)
    tau_r: float = 6.0           # relaxation time constant (s)
    p_y: float = 5.0             # effective yield threshold (arb units)
    p_max: float = 14.0          # upper safe bound (arb units)

    # ---- u proxy and shaping ----
    # u is a proxy for extrusion rate (E per second) scaled to fit the estimator
    u_scale: float = 1.0         # scale factor from (E/s) to u
    du_max: float = 0.35         # max change in u per estimator step (rate limiter)

    # When p_hat is too high: dwell to relax pressure
    relax_dwell_s: float = 0.30
    relax_trigger_margin: float = 0.5  # trigger if p_hat > p_max - margin

    # When p_hat is too low: inject priming (small E) to raise pressure above yield
    lowprime_e: float = 0.6
    lowprime_feed: float = 120.0
    lowprime_steps_max: int = 3

    # ---- Start header: priming ramp + purge ----
    prime_total_e: float = 8.0
    prime_steps: int = 8
    prime_feed: float = 120.0

    purge_x0: float = 10.0
    purge_y0: float = 10.0
    purge_x1: float = 80.0
    purge_y1: float = 10.0
    purge_z: float = 1.2
    purge_e: float = 12.0
    purge_feed_xy: float = 600.0
    purge_feed_e: float = 180.0

    # ---- Retraction suppression ----
    retract_dwell_s: float = 0.35
    micro_prime_e: float = 0.6
    micro_prime_feed: float = 120.0

    # ---- Modes ----
    enforce_absolute_xy: bool = True
    enforce_relative_extrusion: bool = True
    disable_retractions: bool = True

    # ---- Feed scaling limits ----
    # We shape u mostly by adjusting extrusion feed rate F on extrusion moves.
    # We keep it within safe min/max bounds to avoid stalling or splattering.
    feed_scale_min: float = 0.5
    feed_scale_max: float = 1.5

@dataclasses.dataclass
class KinematicState:
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    f: Optional[float] = None  # mm/min last feed

@dataclasses.dataclass
class PressureState:
    p_hat: float = 0.0
    u_prev: float = 0.0

def emit_header_block(cfg: StabilizerConfig) -> List[str]:
    out: List[str] = []
    out.append("; ===== Paste Stabilization Layer (v2, auto-generated) =====")
    out.append("; Method: priming ramp + purge line + pressure-state shaping + retraction suppression")
    out.append("; ============================================================")
    if cfg.enforce_absolute_xy:
        out.append("G90 ; absolute positioning (XY)")
    if cfg.enforce_relative_extrusion:
        out.append("M83 ; relative extrusion (E)")
    out.append(f"G1 Z{cfg.purge_z:.3f} F600 ; set purge Z")
    out.append(f"G1 X{cfg.purge_x0:.3f} Y{cfg.purge_y0:.3f} F3000 ; purge start")
    out.append("; --- Pressure priming ramp (open-loop) ---")
    step_e = cfg.prime_total_e / max(cfg.prime_steps, 1)
    for i in range(cfg.prime_steps):
        out.append(f"G1 E{step_e:.3f} F{cfg.prime_feed:.1f} ; prime step {i+1}/{cfg.prime_steps}")
    out.append("; --- Purge line ---")
    out.append(f"G1 E{(cfg.purge_e*0.2):.3f} F{cfg.purge_feed_e:.1f} ; pre-purge")
    out.append(f"G1 X{cfg.purge_x1:.3f} Y{cfg.purge_y1:.3f} E{cfg.purge_e:.3f} F{cfg.purge_feed_xy:.1f} ; purge line")
    out.append("; ===== End stabilization header =====")
    return out

def emit_retraction_replacement(cfg: StabilizerConfig, original_line: str, 
                                 x: Optional[float] = None, y: Optional[float] = None,
                                 z: Optional[float] = None, f: Optional[float] = None) -> List[str]:
    """
    Replace retraction with dwell + micro-prime, preserving any XY/Z movement.
    """
    out: List[str] = []
    out.append(f"; [stabilizer] RETRACTION SUPPRESSED: {original_line.strip()}")
    
    # Preserve XY/Z movement if present
    has_xy_movement = (x is not None) or (y is not None) or (z is not None)
    
    if cfg.retract_dwell_s > 0:
        out.append(f"G4 S{cfg.retract_dwell_s:.2f} ; dwell instead of retract")
    
    # Build the micro-prime move, preserving geometry
    micro_prime_parts = ["G1"]
    if x is not None:
        micro_prime_parts.append(f"X{x:.3f}")
    if y is not None:
        micro_prime_parts.append(f"Y{y:.3f}")
    if z is not None:
        micro_prime_parts.append(f"Z{z:.3f}")
    if cfg.micro_prime_e > 0:
        micro_prime_parts.append(f"E{cfg.micro_prime_e:.3f}")
    if f is not None:
        micro_prime_parts.append(f"F{f:.1f}")
    elif cfg.micro_prime_feed > 0:
        micro_prime_parts.append(f"F{cfg.micro_prime_feed:.1f}")
    
    if len(micro_prime_parts) > 1:  # More than just "G1"
        comment = "micro-prime after dwell"
        if has_xy_movement:
            comment += " (geometry preserved)"
        out.append(" ".join(micro_prime_parts) + f" ; {comment}")
    elif cfg.micro_prime_e > 0:
        # Pure E move if no geometry
        out.append(f"G1 E{cfg.micro_prime_e:.3f} F{cfg.micro_prime_feed:.1f} ; micro-prime after dwell")
    
    return out

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def xy_length(prev: KinematicState, x: Optional[float], y: Optional[float]) -> Optional[float]:
    if x is None and y is None:
        return 0.0
    if prev.x is None or prev.y is None or x is None or y is None:
        return None
    dx = x - prev.x
    dy = y - prev.y
    return math.hypot(dx, dy)

class PasteStabilizerV2:
    def __init__(self, cfg: StabilizerConfig):
        self.cfg = cfg
        self.k = KinematicState()
        self.p = PressureState()
        self.inserted_header = False
        self.seen_first_motion = False
        self.changes: List[str] = []

        # CSV log rows: list of dicts
        self.log_rows: List[Dict[str, object]] = []
        self.t_sim: float = 0.0  # simulated time in seconds (from dt estimates)

    def _log(self, action: str, line_in: str, line_out: str,
             e: Optional[float], dt: float, u_raw: Optional[float],
             u_shaped: Optional[float], feed_scale: Optional[float]) -> None:
        self.log_rows.append({
            "t_s": round(self.t_sim, 4),
            "action": action,
            "p_hat": round(self.p.p_hat, 6),
            "u_prev": round(self.p.u_prev, 6),
            "e_cmd": "" if e is None else e,
            "dt_s": round(dt, 6),
            "u_raw": "" if u_raw is None else round(u_raw, 6),
            "u_shaped": "" if u_shaped is None else round(u_shaped, 6),
            "feed_scale": "" if feed_scale is None else round(feed_scale, 6),
            "in": line_in.strip(),
            "out": line_out.strip()
        })

    def _maybe_insert_header(self, out: List[str], code: str) -> None:
        if self.inserted_header:
            return
        if is_move(code):
            self.seen_first_motion = True
        if self.seen_first_motion:
            header_lines = emit_header_block(self.cfg)
            out.extend(header_lines)
            self.inserted_header = True
            self.changes.append("Inserted stabilization header (priming ramp + purge line).")
            
            # Initialize pressure state after priming header
            # The header primes the system, so we estimate initial pressure
            prime_e_total = self.cfg.prime_total_e + self.cfg.purge_e * 1.2
            # Rough time estimate for priming operations
            prime_time = max(self.cfg.prime_steps * self.cfg.Ts, 2.0)  # at least 2 seconds
            # Estimate average extrusion rate during priming
            u_prime_avg = self.cfg.u_scale * (prime_e_total / prime_time)
            # Update pressure to reflect priming (multiple steps for accuracy)
            steps = max(5, int(prime_time / self.cfg.Ts))
            for _ in range(steps):
                self._pressure_update(u_prime_avg, self.cfg.Ts)
            self.t_sim += prime_time

    def _estimate_dt(self, code: str, x: Optional[float], y: Optional[float], e: Optional[float], f: Optional[float]) -> float:
        """
        Estimate the duration of a move for logging + u-proxy.
        For XY moves: dt ≈ length / speed
        speed = F (mm/min) -> mm/s = F/60
        """
        # Feed rate token overrides previous
        feed = f if f is not None else self.k.f
        if feed is None:
            feed = 600.0  # default if missing
        v = feed / 60.0  # mm/s

        L = xy_length(self.k, x, y)
        if L is None:
            # If we cannot compute length (no previous XY), approximate a small dt
            return max(self.cfg.min_dt_s, self.cfg.Ts)
        if L == 0.0:
            # pure extrusion or Z hop: approximate dt from extrusion magnitude and feed
            # We cannot map E to mm distance; keep conservative.
            return max(self.cfg.min_dt_s, self.cfg.Ts)
        dt = L / max(v, 1e-6)
        return max(self.cfg.min_dt_s, dt)

    def _pressure_update(self, u: float, dt: float) -> None:
        """
        Integrate p_hat over the estimated dt using Euler steps with step Ts.
        This makes the estimator robust if a single move is long in time.
        Pressure is bounded below by 0 (cannot be negative).
        """
        if dt <= 0:
            return
        Ts = self.cfg.Ts
        steps = max(1, int(math.ceil(dt / Ts)))
        h = dt / steps
        for _ in range(steps):
            dp = self.cfg.alpha * u - (1.0 / self.cfg.tau_r) * self.p.p_hat
            self.p.p_hat = max(0.0, self.p.p_hat + h * dp)  # Ensure non-negative pressure

    def _rate_limit(self, u_raw: float) -> float:
        """
        Apply rate limiting to avoid impulsive changes that cause pressure overshoot.
        """
        du = u_raw - self.p.u_prev
        du_limited = clamp(du, -self.cfg.du_max, self.cfg.du_max)
        u = self.p.u_prev + du_limited
        return u

    def _compute_u_raw(self, e: float, dt: float) -> float:
        """
        u proxy from extrusion increment per time.
        u ~ u_scale * (E/dt)
        """
        return self.cfg.u_scale * (e / max(dt, 1e-6))

    def _apply_shaping_to_move(self, code: str, x: Optional[float], y: Optional[float],
                               e: float, f_token: Optional[float]) -> Tuple[List[str], str]:
        """
        For an extrusion move, shape u and adjust feed rate accordingly.
        Returns (output_lines, action_label).
        """
        out_lines: List[str] = []

        dt = self._estimate_dt(code, x, y, e, f_token)
        u_raw = self._compute_u_raw(e, dt)
        u_shaped = self._rate_limit(u_raw)

        # Update pressure estimate using shaped u (not raw)
        self._pressure_update(u_shaped, dt)

        # Decide interventions based on pressure window
        action = "emit"

        # 1) Too high pressure: dwell (relax), optionally reduce u next
        if self.p.p_hat > (self.cfg.p_max - self.cfg.relax_trigger_margin):
            action = "relax_dwell"
            p_before = self.p.p_hat
            out_lines.append(f"; [stabilizer] p_hat={self.p.p_hat:.2f} near/over p_max, relaxing")
            out_lines.append(f"G4 S{self.cfg.relax_dwell_s:.2f} ; relax pressure")

            # Log the relax dwell action
            self._log(action="relax_dwell",
                     line_in=code, line_out=f"G4 S{self.cfg.relax_dwell_s:.2f}",
                     e=None, dt=self.cfg.relax_dwell_s,
                     u_raw=None, u_shaped=None, feed_scale=None)

            # Let estimator relax during dwell: u=0, dt=dwell
            self._pressure_update(0.0, self.cfg.relax_dwell_s)
            self.t_sim += self.cfg.relax_dwell_s

        # 2) Too low pressure: inject priming steps until above yield (bounded)
        lowprime_count = 0
        while self.p.p_hat < self.cfg.p_y and lowprime_count < self.cfg.lowprime_steps_max:
            action = "low_prime"
            p_before = self.p.p_hat
            out_lines.append(f"; [stabilizer] p_hat={self.p.p_hat:.2f} below p_y, priming")
            prime_line = f"G1 E{self.cfg.lowprime_e:.3f} F{self.cfg.lowprime_feed:.1f}"
            out_lines.append(f"{prime_line} ; low-prime")
            
            # Estimate dt for prime: use Ts
            dtp = self.cfg.Ts
            u_p = self.cfg.u_scale * (self.cfg.lowprime_e / dtp)
            u_p = self._rate_limit(u_p)
            
            # Log the low_prime action
            self._log(action="low_prime",
                     line_in=code, line_out=prime_line,
                     e=self.cfg.lowprime_e, dt=dtp,
                     u_raw=self.cfg.u_scale * (self.cfg.lowprime_e / dtp),
                     u_shaped=u_p, feed_scale=None)
            
            self._pressure_update(u_p, dtp)
            self.p.u_prev = u_p  # Update u_prev for rate limiting continuity
            self.t_sim += dtp
            lowprime_count += 1

        # Feed scaling:
        # Keep E fixed (geometry integrity) but adjust feed to match u_shaped vs u_raw.
        # If u_shaped < u_raw (rate limited), reduce feed rate to match the reduced extrusion rate.
        # Formula: F_new = F_old * (u_shaped/u_raw) to maintain consistent effective extrusion rate.
        feed = f_token if f_token is not None else (self.k.f if self.k.f is not None else 600.0)
        feed_scale = 1.0
        if abs(u_raw) > 1e-6:
            # Scale feed rate proportionally to match shaped extrusion rate
            feed_scale = clamp(u_shaped / u_raw, self.cfg.feed_scale_min, self.cfg.feed_scale_max)
        elif abs(u_shaped) > 1e-6:
            # If u_raw is zero but u_shaped isn't (shouldn't happen), don't scale
            feed_scale = 1.0
        f_new = feed * feed_scale

        # Rewrite the line: set F to f_new (keep XY and E the same)
        # If line already has F, replace; else append.
        if re.search(r"\bF[-+]?\d*\.?\d+", code, flags=re.IGNORECASE):
            code_out = re.sub(r"\bF([-+]?\d*\.?\d+)", f"F{f_new:.1f}", code, flags=re.IGNORECASE)
        else:
            code_out = f"{code} F{f_new:.1f}"

        out_lines.append(code_out)

        # Update prev u and simulated time
        self.p.u_prev = u_shaped
        self.t_sim += dt

        # Update kinematic memory (for next length)
        if x is not None:
            self.k.x = x
        if y is not None:
            self.k.y = y
        if f_new is not None:
            self.k.f = f_new

        # Logging (log only the final emitted move; priming/dwell are logged via separate calls in transform)
        self._log(action=action, line_in=code, line_out=code_out,
                  e=e, dt=dt, u_raw=u_raw, u_shaped=u_shaped, feed_scale=feed_scale)

        return out_lines, action

    def transform(self, in_lines: List[str]) -> List[str]:
        out: List[str] = []

        for idx, raw in enumerate(in_lines):
            code, comment = strip_comment(raw)
            code_stripped = code.strip()

            # Keep blank and pure comment lines
            if not code_stripped:
                out.append(raw.rstrip("\n"))
                continue
            if code_stripped.startswith(";"):
                out.append(raw.rstrip("\n"))
                continue

            # Insert header early
            self._maybe_insert_header(out, code_stripped)

            # Retraction suppression
            if self.cfg.disable_retractions and is_retraction_move(code_stripped):
                # Extract XY/Z/F from the retraction move to preserve geometry
                x = parse_float_token(code_stripped, "X")
                y = parse_float_token(code_stripped, "Y")
                z = parse_float_token(code_stripped, "Z")
                f = parse_float_token(code_stripped, "F")
                e_retract = parse_float_token(code_stripped, "E")
                
                # Update kinematic state if XY/Z present (for next moves)
                if x is not None:
                    self.k.x = x
                if y is not None:
                    self.k.y = y
                if z is not None:
                    self.k.z = z
                if f is not None:
                    self.k.f = f
                
                # Replace retraction, preserving geometry
                repl = emit_retraction_replacement(self.cfg, raw, x, y, z, f)
                out.extend(repl)
                self.changes.append(f"Suppressed retraction at line {idx+1} (geometry preserved).")
                
                # Log the replacement as an event with dt ~ dwell
                self._log(action="retract_suppressed",
                          line_in=raw, line_out=" | ".join(repl),
                          e=e_retract,
                          dt=self.cfg.retract_dwell_s,
                          u_raw=None, u_shaped=None, feed_scale=None)
                # simulate dwell relaxation in estimator:
                self._pressure_update(0.0, self.cfg.retract_dwell_s)
                self.t_sim += self.cfg.retract_dwell_s
                continue

            # For moves with extrusion, apply shaping
            if is_move(code_stripped) and has_extrusion(code_stripped):
                x = parse_float_token(code_stripped, "X")
                y = parse_float_token(code_stripped, "Y")
                e = parse_float_token(code_stripped, "E")
                f = parse_float_token(code_stripped, "F")

                # If E is None, pass through (should not happen here)
                if e is None:
                    out.append(raw.rstrip("\n"))
                    continue

                # If E <= 0: either retraction (handled earlier) or zero; pass through
                if e <= 0:
                    out.append(raw.rstrip("\n"))
                    continue

                shaped_lines, action = self._apply_shaping_to_move(code_stripped, x, y, e, f)
                out.extend(shaped_lines)
                if action != "emit":
                    self.changes.append(f"Applied shaping action '{action}' at line {idx+1}.")
                continue

            # Non-extrusion move: update kinematics and pass through
            if is_move(code_stripped):
                x = parse_float_token(code_stripped, "X")
                y = parse_float_token(code_stripped, "Y")
                z = parse_float_token(code_stripped, "Z")
                f = parse_float_token(code_stripped, "F")

                # Update kinematic state (only update if value is present)
                if x is not None:
                    self.k.x = x
                if y is not None:
                    self.k.y = y
                if z is not None:
                    self.k.z = z
                if f is not None:
                    self.k.f = f

            out.append(raw.rstrip("\n"))

        # Fallback header insertion if no motion detected
        if not self.inserted_header:
            out.extend(emit_header_block(self.cfg))
            self.changes.append("Inserted stabilization header at end (fallback).")

        return out

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser(description="Paste extrusion stabilizer v2 (pressure estimation + command shaping).")
    ap.add_argument("--in", dest="infile", required=True, help="Input slicer G-code file")
    ap.add_argument("--out", dest="outfile", required=True, help="Output stabilized G-code file")
    ap.add_argument("--csv", dest="csvfile", default="run_log.csv", help="CSV log output for plots")
    ap.add_argument("--log", dest="logfile", default="changes.log", help="Human-readable change log")
    ap.add_argument("--no-csv", action="store_true", help="Disable CSV log")
    args = ap.parse_args()

    in_path = Path(args.infile)
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # All outputs go to results directory
    out_path = results_dir / Path(args.outfile).name
    csv_path = results_dir / Path(args.csvfile).name
    log_path = results_dir / Path(args.logfile).name

    if not in_path.exists():
        print(f"ERROR: input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    cfg = StabilizerConfig()
    stab = PasteStabilizerV2(cfg)

    try:
        in_lines = in_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as e:
        print(f"ERROR: Failed to read input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    if len(in_lines) == 0:
        print("WARNING: Input file is empty", file=sys.stderr)
    
    try:
        out_lines = stab.transform(in_lines)
    except Exception as e:
        print(f"ERROR: Failed to transform G-code: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    try:
        out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        log_path.write_text("\n".join(stab.changes) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to write output files: {e}", file=sys.stderr)
        sys.exit(1)

    if not args.no_csv:
        try:
            write_csv(csv_path, stab.log_rows)
        except Exception as e:
            print(f"WARNING: Failed to write CSV log: {e}", file=sys.stderr)

    print(f"✓ Saved stabilized G-code: {out_path} ({len(out_lines)} lines)")
    print(f"✓ Saved change log: {log_path} ({len(stab.changes)} changes)")
    if not args.no_csv:
        print(f"✓ Saved CSV run log: {csv_path} ({len(stab.log_rows)} log entries)")
    
    # Print summary statistics
    if stab.log_rows:
        actions = [r.get("action", "") for r in stab.log_rows]
        action_counts = {a: actions.count(a) for a in set(actions)}
        print(f"\nSummary:")
        print(f"  Total moves processed: {len(stab.log_rows)}")
        for action, count in sorted(action_counts.items()):
            if action:
                print(f"  {action}: {count}")

if __name__ == "__main__":
    main()
