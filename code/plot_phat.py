import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Save figures in a subfolder inside results
FIG_DIR = Path("results") / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Read the CSV produced by paste_stabilizer_v2.py (from results folder)
csv_path = Path("results") / "run_log.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"CSV log not found: {csv_path}")
df = pd.read_csv(csv_path)

# Some rows may have empty p_hat due to non-move logs; keep valid
df["p_hat"] = pd.to_numeric(df["p_hat"], errors="coerce")
df = df.dropna(subset=["p_hat"])

t = df["t_s"].values
p = df["p_hat"].values

# Use constants consistent with code defaults (adjust if you changed cfg)
p_y = 5.0
p_max = 14.0

plt.figure()
plt.plot(t, p, label="p_hat(t)")
plt.axhline(p_y, linestyle="--", label="p_y (yield)")
plt.axhline(p_max, linestyle="--", label="p_max (upper bound)")
plt.xlabel("Time (s)")
plt.ylabel("Estimated pressure (arb. units)")
plt.title("Estimated pressure state and admissible window")
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "phat_trace.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "phat_trace.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved {FIG_DIR}/phat_trace.pdf and {FIG_DIR}/phat_trace.png")
