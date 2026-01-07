# stabilizer/planner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .gcode import GCode
from .policy import RetractionPolicy, FirstLayerPolicy, PrimePolicy, PurgeLinePolicy, RecoveryPolicy

@dataclass
class PlannerConfig:
    use_relative_e: bool
    prime: PrimePolicy
    purge_line: PurgeLinePolicy
    retraction: RetractionPolicy
    first_layer: FirstLayerPolicy
    recovery: RecoveryPolicy

class StabilizationPlanner:
    """
    Pure transformation: takes slicer G-code and outputs a stabilized G-code program.
    Runtime recovery (when streaming) is handled in streamer.py; here we embed basic start routines
    and suppress harmful patterns (retractions, first-layer speed).
    """
    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg
        self._layer_z: Optional[float] = None
        self._in_first_layer: bool = True
        self._last_z: Optional[float] = None

    def plan(self, gcode: List[GCode]) -> List[GCode]:
        out: List[GCode] = []

        # ---- Header: enforce safe modes
        out += self._header_block()

        # ---- Add prime + purge before the print starts
        out += self._prime_block()
        out += self._purge_line_block()

        # ---- Process original lines
        for g in gcode:
            # detect layer changes (simple: if Z changes upwards)
            z = g.get("Z", None)
            if z is not None:
                if self._last_z is None:
                    self._last_z = z
                else:
                    if z > self._last_z + 1e-6:
                        self._in_first_layer = False
                    self._last_z = z

            # Apply first-layer speed factor to extruding moves
            if self.cfg.first_layer.enable and self._in_first_layer and g.is_motion() and g.has_extrusion():
                g = self._scale_feedrate(g, self.cfg.first_layer.speed_factor)

            # Optional Z override on first layer (only if explicitly set)
            if self.cfg.first_layer.enable and self._in_first_layer and self.cfg.first_layer.z_override_mm is not None:
                if g.is_motion() and ("Z" in (g.args or {})):
                    g = self._override_z(g, self.cfg.first_layer.z_override_mm)

            # Retraction handling
            if g.is_motion() and g.has_extrusion():
                # In relative E, retractions are negative E moves.
                e = g.get("E", None)
                if e is not None and e < 0:
                    if self.cfg.retraction.mode == "suppress":
                        out.append(GCode(raw=g.raw, cmd=None, args={}, comment="suppressed retraction"))
                        continue
                    elif self.cfg.retraction.mode == "micro":
                        out.append(GCode(raw="; micro retract", cmd=None, args={}, comment=""))
                        out.append(GCode(raw="G1 E{:.4f}".format(-abs(self.cfg.retraction.micro_retract_mm)),
                                        cmd="G1", args={"E": -abs(self.cfg.retraction.micro_retract_mm)}, comment="micro retract"))
                        continue
                    # passthrough otherwise

            out.append(g)

        out += self._footer_block()
        return out

    def _header_block(self) -> List[GCode]:
        lines: List[GCode] = []
        lines.append(GCode(raw="; --- paste stabilizer header ---", cmd=None, args={}, comment=""))
        # units / modes
        lines.append(GCode(raw="G21", cmd="G21", args={}, comment="mm units"))
        lines.append(GCode(raw="G90", cmd="G90", args={}, comment="absolute XYZ"))
        if self.cfg.use_relative_e:
            lines.append(GCode(raw="M83", cmd="M83", args={}, comment="relative E"))
        else:
            lines.append(GCode(raw="M82", cmd="M82", args={}, comment="absolute E"))
        return lines

    def _prime_block(self) -> List[GCode]:
        if not self.cfg.prime.enable:
            return []
        p = self.cfg.prime
        lines: List[GCode] = []
        lines.append(GCode(raw="; --- prime routine ---", cmd=None, args={}, comment=""))
        # move to safe XY, lift Z
        lines.append(GCode(raw=f"G0 X{p.safe_x:g} Y{p.safe_y:g}", cmd="G0", args={"X": p.safe_x, "Y": p.safe_y}, comment="safe XY"))
        lines.append(GCode(raw=f"G0 Z{p.z_lift_mm:g}", cmd="G0", args={"Z": p.z_lift_mm}, comment="lift"))
        # dwell for relaxation
        lines.append(GCode(raw=f"G4 S{p.dwell_s:g}", cmd="G4", args={"S": p.dwell_s}, comment="dwell for pressure build"))
        # slow extrusion in steps (split into ~1s chunks)
        total = max(0.0, p.e_total_mm)
        rate = max(1e-6, p.e_rate_mm_s)
        step_e = min(1.0 * rate, total)  # ~1 second per chunk
        remaining = total
        while remaining > 1e-6:
            e_chunk = min(step_e, remaining)
            # Feedrate for E-only in mm/min:
            f = rate * 60.0
            lines.append(GCode(raw=f"G1 E{e_chunk:g} F{f:g}",
                               cmd="G1", args={"E": e_chunk, "F": f}, comment="prime"))
            remaining -= e_chunk
        return lines

    def _purge_line_block(self) -> List[GCode]:
        if not self.cfg.purge_line.enable:
            return []
        p = self.cfg.purge_line
        lines: List[GCode] = []
        lines.append(GCode(raw="; --- purge line ---", cmd=None, args={}, comment=""))
        # Assume we are at safe XY; draw a line in +X direction
        lines.append(GCode(raw=f"G0 Z{p.z:g}", cmd="G0", args={"Z": p.z}, comment="purge Z"))
        # compute E needed
        e_total = p.length_mm * p.e_per_mm
        f = p.speed_mm_s * 60.0
        lines.append(GCode(raw=f"G1 X{p.length_mm:g} E{e_total:g} F{f:g}",
                           cmd="G1", args={"X": p.length_mm, "E": e_total, "F": f}, comment="purge line"))
        return lines

    def _footer_block(self) -> List[GCode]:
        lines: List[GCode] = []
        lines.append(GCode(raw="; --- end paste stabilizer footer ---", cmd=None, args={}, comment=""))
        return lines

    def _scale_feedrate(self, g: GCode, factor: float) -> GCode:
        if g.args is None:
            return g
        if "F" not in g.args:
            return g
        new_args = dict(g.args)
        new_args["F"] = new_args["F"] * float(factor)
        return GCode(raw=g.raw, cmd=g.cmd, args=new_args, comment=(g.comment + " first-layer slow").strip())

    def _override_z(self, g: GCode, z: float) -> GCode:
        new_args = dict(g.args or {})
        new_args["Z"] = float(z)
        return GCode(raw=g.raw, cmd=g.cmd, args=new_args, comment=(g.comment + " Z override").strip())
