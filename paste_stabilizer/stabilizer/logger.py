# stabilizer/logger.py
from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass

@dataclass
class LogConfig:
    out_dir: str
    csv_name: str
    log_raw_serial: bool = True

class RunLogger:
    def __init__(self, cfg: LogConfig):
        os.makedirs(cfg.out_dir, exist_ok=True)
        self.path = os.path.join(cfg.out_dir, cfg.csv_name)
        self.cfg = cfg
        self._t0 = time.time()
        self._f = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(["t_s", "type", "message"])
        self._f.flush()

    def _ts(self) -> float:
        return time.time() - self._t0

    def log_tx(self, line: str):
        self._w.writerow([f"{self._ts():.3f}", "TX", line])
        self._f.flush()

    def log_rx(self, line: str):
        if not self.cfg.log_raw_serial:
            return
        self._w.writerow([f"{self._ts():.3f}", "RX", line])
        self._f.flush()

    def log_note(self, note: str):
        self._w.writerow([f"{self._ts():.3f}", "NOTE", note])
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
