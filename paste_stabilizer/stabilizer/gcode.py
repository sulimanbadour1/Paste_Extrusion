# stabilizer/gcode.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import re

_CMD_RE = re.compile(r"^\s*([GMT]\d+)\s*(.*)$", re.IGNORECASE)

@dataclass
class GCode:
    raw: str
    cmd: Optional[str] = None          # e.g. "G1"
    args: Dict[str, float] = None      # e.g. {"X": 10.0, "E": 0.25}
    comment: str = ""

    def is_motion(self) -> bool:
        return (self.cmd or "").upper() in {"G0", "G1"}

    def has_extrusion(self) -> bool:
        return self.args is not None and ("E" in self.args)

    def get(self, key: str, default=None):
        if not self.args:
            return default
        return self.args.get(key.upper(), default)

    def with_comment(self, extra: str) -> "GCode":
        c = (self.comment + " " + extra).strip()
        return GCode(raw=self.raw, cmd=self.cmd, args=dict(self.args or {}), comment=c)

    def render(self) -> str:
        if self.cmd is None:
            # comment-only or blank
            line = self.raw.strip("\n")
            return line
        parts = [self.cmd.upper()]
        if self.args:
            for k, v in self.args.items():
                parts.append(f"{k.upper()}{v:g}")
        line = " ".join(parts)
        if self.comment:
            line += f" ; {self.comment}"
        return line

def parse_line(line: str) -> GCode:
    original = line.rstrip("\n")
    # split comment
    if ";" in original:
        code_part, comment = original.split(";", 1)
        comment = comment.strip()
    else:
        code_part, comment = original, ""

    code_part = code_part.strip()
    if code_part == "":
        return GCode(raw=original, cmd=None, args={}, comment=comment)

    m = _CMD_RE.match(code_part)
    if not m:
        # unknown line; pass-through
        return GCode(raw=original, cmd=None, args={}, comment=comment)

    cmd = m.group(1).upper()
    rest = m.group(2).strip()

    args: Dict[str, float] = {}
    if rest:
        tokens = rest.split()
        for t in tokens:
            if len(t) < 2:
                continue
            k = t[0].upper()
            try:
                v = float(t[1:])
            except ValueError:
                continue
            args[k] = v

    return GCode(raw=original, cmd=cmd, args=args, comment=comment)

def read_gcode_file(path: str) -> List[GCode]:
    out: List[GCode] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            out.append(parse_line(line))
    return out

def write_gcode_file(path: str, lines: List[GCode]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for g in lines:
            f.write(g.render().rstrip() + "\n")
