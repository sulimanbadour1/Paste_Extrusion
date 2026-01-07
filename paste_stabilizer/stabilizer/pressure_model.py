# stabilizer/pressure_model.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PressureModel:
    alpha: float
    tau_r_s: float
    p_y: float
    p_max: float
    tau_d_s: float
    k_flow: float
    p: float = 0.0

    def step(self, u: float, dt: float) -> float:
        """
        Discrete pressure update:
          dp/dt = alpha*u - (1/tau_r)*p
        u is an abstract extrusion command rate (e.g., mm/s of E)
        """
        if dt <= 0:
            return self.p
        dp = self.alpha * u - (self.p / max(self.tau_r_s, 1e-6))
        self.p = max(0.0, self.p + dp * dt)
        return self.p

    def flow(self) -> float:
        """Simple yield model: flow proportional to (p - p_y), clipped."""
        if self.p <= self.p_y:
            return 0.0
        return self.k_flow * (self.p - self.p_y)

    def in_window(self) -> bool:
        return (self.p > self.p_y) and (self.p < self.p_max)
