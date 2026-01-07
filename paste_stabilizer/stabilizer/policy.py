# stabilizer/policy.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class RetractionPolicy:
    mode: str                   # suppress | micro | passthrough
    micro_retract_mm: float
    micro_unretract_mm: float
    min_segment_mm: float

@dataclass
class FirstLayerPolicy:
    enable: bool
    speed_factor: float
    z_override_mm: Optional[float]

@dataclass
class PrimePolicy:
    enable: bool
    dwell_s: float
    e_total_mm: float
    e_rate_mm_s: float
    z_lift_mm: float
    safe_x: float
    safe_y: float

@dataclass
class PurgeLinePolicy:
    enable: bool
    length_mm: float
    speed_mm_s: float
    e_per_mm: float
    z: float

@dataclass
class RecoveryPolicy:
    enable: bool
    dwell_s: float
    purge_e_mm: float
    purge_rate_mm_s: float
    max_recoveries: int
