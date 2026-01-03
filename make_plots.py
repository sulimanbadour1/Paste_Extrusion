import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(__file__) or "."
FIG_DIR = "figures"

os.makedirs(FIG_DIR, exist_ok=True)

def save_fig(name: str):
    # Save both PDF (best for LaTeX) and PNG (quick preview)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close()

def summarize_print_trials(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("condition")
    out = pd.DataFrame({
        "n": g.size(),
        "first_layer_success_rate": g["first_layer_success"].mean(),
        "completion_rate": g["completed"].mean(),
        "clogs_per_print_mean": g["clogs"].mean(),
        "onset_s_mean": g["onset_s"].mean(),
        "onset_s_std": g["onset_s"].std(ddof=1),
        "flow_duration_s_mean": g["flow_duration_s"].mean(),
        "flow_duration_s_std": g["flow_duration_s"].std(ddof=1),
    }).reset_index()
    return out

def plot_extrusion_onset_and_duration(df: pd.DataFrame):
    # Two-panel style is not allowed if you want strict single-plot; here we create two separate plots.
    order = ["baseline", "partial", "full"]

    # Boxplot: onset time
    plt.figure()
    data = [df.loc[df["condition"] == c, "onset_s"].values for c in order]
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Extrusion onset time (s)")
    plt.title("Extrusion onset time by configuration")
    save_fig("extrusion_onset_boxplot")

    # Boxplot: continuous flow duration
    plt.figure()
    data = [df.loc[df["condition"] == c, "flow_duration_s"].values for c in order]
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Continuous flow duration (s)")
    plt.title("Continuous flow duration by configuration")
    save_fig("flow_duration_boxplot")

def plot_success_rates(df: pd.DataFrame):
    order = ["baseline", "partial", "full"]
    g = df.groupby("condition")[["first_layer_success", "completed"]].mean().reindex(order)

    # First-layer success bar
    plt.figure()
    plt.bar(g.index, g["first_layer_success"].values)
    plt.ylim(0, 1.0)
    plt.ylabel("Success rate")
    plt.title("First-layer success rate")
    save_fig("first_layer_success_rate")

    # Completion rate bar
    plt.figure()
    plt.bar(g.index, g["completed"].values)
    plt.ylim(0, 1.0)
    plt.ylabel("Completion rate")
    plt.title("Print completion rate")
    save_fig("completion_rate")

def plot_clogs_per_print(df: pd.DataFrame):
    order = ["baseline", "partial", "full"]
    means = df.groupby("condition")["clogs"].mean().reindex(order)
    stds = df.groupby("condition")["clogs"].std(ddof=1).reindex(order)

    plt.figure()
    plt.bar(means.index, means.values, yerr=stds.values, capsize=4)
    plt.ylabel("Clogs per print (mean ± SD)")
    plt.title("Clog frequency by configuration")
    save_fig("clogs_per_print")

def simulate_pressure_model():
    """
    Simple pressure model to generate an explanatory figure:
      p_dot = alpha*u - p/tau_r
      flow occurs when p > p_y (with dead-time tau_d approximated in plotting only)
    We generate baseline-like stop/start u(t) vs stabilized ramped u(t),
    then plot p(t) with p_y and p_max lines.
    """
    dt = 0.05
    t = np.arange(0, 60 + dt, dt)

    alpha = 1.0
    tau_r = 6.0
    p_y = 5.0
    p_max = 14.0

    # Baseline u(t): frequent start/stop
    u_base = np.zeros_like(t)
    for k in range(len(t)):
        # on for 2s, off for 1s repeatedly (aggressive stop/start)
        phase = t[k] % 3.0
        u_base[k] = 1.8 if phase < 2.0 else 0.0

    # Stabilized u(t): ramp + fewer discontinuities
    u_stab = np.zeros_like(t)
    # ramp up 0-5s, then mostly on with gentle modulation
    for k in range(len(t)):
        if t[k] < 5:
            u_stab[k] = 0.35 * t[k]  # ramp
        else:
            u_stab[k] = 1.4 + 0.2 * np.sin(0.2 * t[k])

    def integrate(u):
        p = np.zeros_like(t)
        for k in range(1, len(t)):
            p_dot = alpha * u[k-1] - (1.0 / tau_r) * p[k-1]
            p[k] = p[k-1] + dt * p_dot
        return p

    p_base = integrate(u_base)
    p_stab = integrate(u_stab)

    # Plot baseline pressure
    plt.figure()
    plt.plot(t, p_base, label="baseline p(t)")
    plt.axhline(p_y, linestyle="--", label="yield threshold p_y")
    plt.axhline(p_max, linestyle="--", label="upper bound p_max")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (arb. units)")
    plt.title("Simulated pressure trajectory: baseline stop/start")
    plt.legend()
    save_fig("pressure_simulation_baseline")

    # Plot stabilized pressure
    plt.figure()
    plt.plot(t, p_stab, label="stabilized p(t)")
    plt.axhline(p_y, linestyle="--", label="yield threshold p_y")
    plt.axhline(p_max, linestyle="--", label="upper bound p_max")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (arb. units)")
    plt.title("Simulated pressure trajectory: stabilized execution")
    plt.legend()
    save_fig("pressure_simulation_stabilized")

def plot_electrical_results(df: pd.DataFrame):
    # Resistance distribution for non-open circuits
    df_ok = df[df["open_circuit"] == 0].copy()
    df_ok["resistance_ohm"] = pd.to_numeric(df_ok["resistance_ohm"])

    order = ["baseline", "full"]
    data = [df_ok.loc[df_ok["condition"] == c, "resistance_ohm"].values for c in order]

    plt.figure()
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Resistance (Ω)")
    plt.title("Resistance distribution of conductive traces (non-open)")
    save_fig("resistance_boxplot")

    # Open-circuit rate bar
    g = df.groupby("condition")["open_circuit"].mean().reindex(order)
    plt.figure()
    plt.bar(g.index, g.values)
    plt.ylim(0, 1.0)
    plt.ylabel("Open-circuit rate")
    plt.title("Open-circuit rate by configuration")
    save_fig("open_circuit_rate")

def print_latex_tables(print_df: pd.DataFrame, elec_df: pd.DataFrame):
    # Print summary table rows you can paste into LaTeX if desired
    summ = summarize_print_trials(print_df)
    print("\n=== LaTeX-ready summary (print trials) ===")
    for _, r in summ.iterrows():
        cond = r["condition"]
        fl = int(print_df[print_df["condition"] == cond]["first_layer_success"].sum())
        comp = int(print_df[print_df["condition"] == cond]["completed"].sum())
        n = int(r["n"])
        print(f"{cond} & {r['clogs_per_print_mean']:.1f} & {fl}/{n} & {comp}/{n} \\\\")
    print("")

    # Electrical
    print("=== LaTeX-ready summary (electrical) ===")
    g = elec_df.groupby("condition")
    for cond, d in g:
        n = len(d)
        opens = int(d["open_circuit"].sum())
        ok = d[d["open_circuit"] == 0].copy()
        ok["resistance_ohm"] = pd.to_numeric(ok["resistance_ohm"], errors="coerce")
        if len(ok) > 1:
            var = np.var(ok["resistance_ohm"].values, ddof=1)
        elif len(ok) == 1:
            var = 0.0
        else:
            var = np.nan
        print(f"{cond} & {opens}/{n} & {var:.1f} \\\\")
    print("")

def main():
    print_trials = pd.read_csv(os.path.join(DATA_DIR, "print_trials.csv"))
    electrical = pd.read_csv(os.path.join(DATA_DIR, "electrical_traces.csv"))

    plot_extrusion_onset_and_duration(print_trials)
    plot_success_rates(print_trials)
    plot_clogs_per_print(print_trials)
    simulate_pressure_model()
    plot_electrical_results(electrical)

    print_latex_tables(print_trials, electrical)
    print(f"Plots saved to: {FIG_DIR}/ (PDF + PNG)")

if __name__ == "__main__":
    main()
