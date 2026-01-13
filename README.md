# Paste Extrusion Stabilization Layer
## Software-Defined Pressure & Flow Shaping for Low-Cost 3D Printers

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IEEE-orange.svg)]()

> **Transform any FDM slicer output into paste-stable G-code** — No hardware modifications required!

This project implements the software middleware layer described in the research paper:
**"Software-Defined Pressure and Flow Stabilization of Low-Cost Paste Extrusion 3D Printers for Structural and Electrical Printing"**

---

## 🎯 What This Does

Paste extrusion behaves fundamentally differently from FDM printing. Standard slicer-generated G-code contains aggressive retractions and stop/start patterns that **collapse paste pressure and cause clogs**. This stabilizer transforms that G-code into paste-compatible commands that maintain continuous flow and stable pressure.

### Key Innovation
- **Zero hardware modifications** — Pure software solution
- **Geometry preservation** — Maintains exact print dimensions
- **Active pressure control** — Real-time estimation and shaping
- **Publication-ready** — Automated figure generation for research papers

---

## 🚀 Quick Start

### Option 1: GUI (Recommended for Beginners)

```bash
cd code
python3 stabilizer_gui.py
```

The GUI provides a complete workflow:
1. **Stabilize** your G-code
2. **Verify** the output
3. **Visualize** 3D comparisons
4. **Generate** publication figures

### Option 2: Command Line

```bash
# 1. Stabilize G-code
python3 paste_stabilizer_v2.py --in test.gcode --out results/stabilized.gcode \
  --csv results/run_log.csv --log results/changes.log

# 2. Verify output
python3 verify_stabilizer.py --in test.gcode --out results/stabilized.gcode \
  --csv results/run_log.csv

# 3. Generate figures
python3 generate_10_figures.py --baseline-gcode test.gcode \
  --stabilized-gcode results/stabilized.gcode --figures all
```

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Workflow Guide](#-workflow-guide)
- [Figure Generation](#-figure-generation)
- [Input Data Files](#-input-data-files)
- [CLI Reference](#-cli-reference)
- [Troubleshooting](#-troubleshooting)
- [Scientific Usage](#-scientific-usage)

---

## ✨ Features

### Core Capabilities

| Feature | Description | Benefit |
|---------|-------------|---------|
| **Retraction Suppression** | Replaces negative E moves with dwell + micro-prime | Prevents pressure collapse |
| **Pressure Estimation** | Real-time `p_hat` tracking | Predictive failure detection |
| **Command Shaping** | Active feed rate and extrusion modulation | Maintains pressure window |
| **Geometry Preservation** | Maintains XY/Z coordinates during suppression | Accurate prints |
| **Scientific Logging** | Detailed CSV logs with timestamps | Reproducible research |

### Advanced Features

- 🎨 **24 Publication-Ready Figures** — IEEE-compatible formatting
- 📊 **3D Visualization** — Interactive toolpath rendering
- 🔍 **Comprehensive Analysis** — Statistical metrics and comparisons
- 🖥️ **Professional GUI** — Complete workflow management
- 📈 **Experimental Data Support** — Survival curves, electrical measurements

---

## 📦 Installation

### Requirements

- **Python 3.9+** (3.10+ recommended)
- **Required packages:**
  ```bash
  pip install pandas matplotlib numpy scipy
  ```

### Optional Dependencies

- `scipy` — Enhanced survival analysis (Kaplan-Meier)
- `tkinter` — GUI support (usually included with Python)

### Quick Install

```bash
# Clone or download the repository
cd paste_paper_package/data

# Install dependencies
pip install pandas matplotlib numpy scipy

# Verify installation
python3 -c "import pandas, matplotlib, numpy; print('✓ All packages installed')"
```

---

## 📁 Project Structure

```
code/
├── 🎯 stabilizer_gui.py          # GUI application (START HERE!)
├── ⚙️  paste_stabilizer_v2.py    # Main stabilizer engine
├── ✅ verify_stabilizer.py        # Verification & QA tool
├── 📊 compare_gcode.py           # Comprehensive comparison tool
├── 📈 generate_10_figures.py      # Paper figure generator (24 figures)
├── 📉 paper_figs.py              # Alternative figure generator
├── 🗺️  3d_map.py                 # 3D visualization tools
│
├── 📂 input/                      # Input data directory
│   ├── print_trials.csv          # Experimental print trial data
│   ├── electrical_traces.csv     # Electrical conductivity measurements
│   ├── first_layer_sweep.csv     # First-layer operating envelope (optional)
│   └── README.md                 # Input files documentation
│
├── 📂 results/                    # Output directory (auto-created)
│   ├── stabilized.gcode          # Stabilized output G-code
│   ├── run_log.csv               # Pressure and action log
│   ├── changes.log               # Human-readable change log
│   └── figures/                  # Generated plots
│       ├── comparison_*.png      # Comparison visualizations
│       └── verification_*.png    # Verification plots
│
└── 📄 test.gcode                 # Example input G-code
```

---

## 🔧 How It Works

### The Stabilization Pipeline

```
┌─────────────┐
│ Input G-code│
│ (FDM slicer)│
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  1. Priming Ramp + Purge Line      │
│     → Builds stable pressure        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. Retraction Suppression          │
│     G1 E-4.0 → G4 S0.35 + G1 E0.6  │
│     → Preserves geometry            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. Pressure Estimation (p_hat)    │
│     → Tracks internal pressure      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. Command Shaping                │
│     if p_hat < p_y: inject prime   │
│     if p_hat > p_max: insert dwell │
│     → Maintains pressure window    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────┐
│Stabilized   │
│G-code + Logs│
└─────────────┘
```

### Key Algorithms

1. **Pressure Model**: `p_{k+1} = p_k + Δt_k * (α * u_k - p_k / τ_r)`
2. **Rate Limiting**: `|Δu_k| ≤ du_max` (prevents impulsive changes)
3. **Window Supervision**: `p_y ≤ p_hat ≤ p_max` (safe operating range)

---

## 📖 Workflow Guide

### Complete Workflow (GUI)

1. **Launch GUI**
   ```bash
   python3 stabilizer_gui.py
   ```

2. **Tab 1: Stabilize G-code**
   - Select your input G-code file
   - Click "Run Stabilizer"
   - Output: `stabilized.gcode`, `run_log.csv`, `changes.log`

3. **Tab 2: Verify Output**
   - Verify stabilization worked correctly
   - Check: header inserted, retractions suppressed, shaping occurred

4. **Tab 3: Visualize**
   - Generate 3D toolpath comparisons
   - View before/after visualizations

5. **Tab 4: Research Plots**
   - Configure input data files (CSV files)
   - Generate figures 1-7 (G-code analysis + experimental data)

6. **Tab 5: Paper Figures**
   - Select figures to generate (1-24 available)
   - One-click generation with IEEE-compatible formatting

### Command-Line Workflow

```bash
# Step 1: Stabilize
python3 paste_stabilizer_v2.py \
  --in input.gcode \
  --out results/stabilized.gcode \
  --csv results/run_log.csv \
  --log results/changes.log

# Step 2: Verify
python3 verify_stabilizer.py \
  --in input.gcode \
  --out results/stabilized.gcode \
  --csv results/run_log.csv

# Step 3: Compare
python3 compare_gcode.py input.gcode results/stabilized.gcode

# Step 4: Generate Figures
python3 generate_10_figures.py \
  --baseline-gcode input.gcode \
  --stabilized-gcode results/stabilized.gcode \
  --data-dir input \
  --figures all
```

---

## 📊 Figure Generation

### Available Figures (24 Total)

#### G-code Analysis (Figures 1-6)
- **Fig. 1** — G-code modification summary (delta)
- **Fig. 2** — Retraction suppression histogram
- **Fig. 3** — Extrusion-rate timeline u(t) — Baseline
- **Fig. 4** — Extrusion-rate timeline u(t) — Stabilized
- **Fig. 5** — Pressure estimate p̂(t) with bounds — Baseline
- **Fig. 6** — Pressure estimate p̂(t) with bounds — Stabilized

#### Experimental Data (Figures 7-10, 14-17, 19-20)
- **Fig. 7** — Extrusion continuity survival curve ⭐
- **Fig. 8** — First-layer operating envelope heatmap
- **Fig. 9** — Electrical yield (open-circuit rate)
- **Fig. 10** — Resistance stability (boxplot)
- **Fig. 14** — Print completion rate (Executive KPI)
- **Fig. 15** — Extrusion onset time distribution
- **Fig. 16** — Flow interruptions / clogs per print
- **Fig. 17** — Electrical resistance comparison
- **Fig. 19** — Ablation study
- **Fig. 20** — Peak pressure vs failure probability

#### Visualizations (Figures 11-12)
- **Fig. 11** — 3D Toolpath Comparison (Before/After)
- **Fig. 12** — 3D Extrusion Rate Map

#### Analysis & Metrics (Figures 13, 21-24)
- **Fig. 13** — Effectiveness Dashboard
- **Fig. 21** — Extrusion Width Uniformity ⭐
- **Fig. 22** — Energy / Motor Load Proxy
- **Fig. 23** — Time-Lapse Frame with Flow Annotation
- **Fig. 24** — Time-to-Failure Summary Statistics ⭐

⭐ = Requires experimental data from `input/` directory

### Generating Figures

**Via GUI:**
1. Go to "Research Plots" or "Paper Figures" tab
2. Configure input data files (if needed)
3. Select figures to generate
4. Click "Generate"

**Via Command Line:**
```bash
# Generate specific figures
python3 generate_10_figures.py \
  --baseline-gcode test.gcode \
  --stabilized-gcode results/stabilized.gcode \
  --data-dir input \
  --figures 7 21 24

# Generate all figures
python3 generate_10_figures.py \
  --baseline-gcode test.gcode \
  --stabilized-gcode results/stabilized.gcode \
  --data-dir input \
  --figures all
```

---

## 📂 Input Data Files

Experimental figures require CSV data files in `code/input/`:

### Required Files

| File | Description | Used For |
|------|-------------|----------|
| `print_trials.csv` | Print trial data with survival times | Figures 7, 14-16, 19-20 |
| `electrical_traces.csv` | Electrical conductivity measurements | Figures 9-10, 17, 19 |

### Optional Files

| File | Description | Used For |
|------|-------------|----------|
| `first_layer_sweep.csv` | First-layer operating envelope | Figure 8 |

### File Formats

**`print_trials.csv`** columns:
- `trial` — Trial number
- `condition` — Condition type (baseline, partial, full)
- `onset_s` — Extrusion onset time (seconds)
- `flow_duration_s` — Flow duration / time-to-failure (seconds)
- `first_layer_success` — First layer success (0/1)
- `completed` — Print completion (0/1, 1=censored)
- `clogs` — Number of clogs

**`electrical_traces.csv`** columns:
- `trace_id` — Trace identifier
- `condition` — Condition type
- `length_mm` — Trace length (mm)
- `resistance_ohm` — Electrical resistance (Ω)
- `open_circuit` — Open circuit indicator (0/1)

See `code/input/README.md` for detailed documentation.

---

## 🎨 What the Stabilizer Does

### Core Transformations

#### 1. Priming Ramp + Purge Line
```
Before: [Print starts immediately]
After:  [8-step priming ramp] → [Purge line] → [Print starts]
```
**Purpose**: Builds stable pressure before printing real geometry

#### 2. Retraction Suppression
```
Before: G1 X10 Y20 E-4.0    (retraction)
After:  G4 S0.35            (dwell)
        G1 X10 Y20 E0.6     (micro-prime)
```
**Purpose**: Prevents pressure collapse while preserving geometry

#### 3. Pressure Estimation & Shaping
```
if p_hat < p_y (5.0):
    → Inject low-priming (E=0.6mm)
if p_hat > p_max (14.0):
    → Insert dwell (0.3s)
```
**Purpose**: Maintains pressure in safe operating window

#### 4. Rate Limiting
```
|Δu_k| ≤ 0.35  (max change per step)
```
**Purpose**: Prevents impulsive extrusion changes

---

## 📈 Verification & Analysis

### Verification Checks

The `verify_stabilizer.py` tool performs automated QA:

✅ **Header Check** — Stabilization header inserted  
✅ **Retraction Removal** — All negative E moves suppressed  
✅ **Mode Enforcement** — G90 (absolute XY) and M83 (relative E) set  
✅ **Shaping Evidence** — Feed scaling or pressure actions present  
✅ **Log Validation** — CSV log exists with plausible values  
✅ **Pressure Compliance** — `p_hat` stays within `[p_y, p_max]` window

### Comparison Analysis

The `compare_gcode.py` tool generates 6 comprehensive plots:

1. **3D Toolpath Comparison** — Side-by-side visualization
2. **Extrusion Statistics** — Distributions and cumulative plots
3. **Retraction vs Micro-prime** — Spatial analysis
4. **Feed Rate Comparison** — Before/after distributions
5. **Geometry Preservation** — Toolpath overlay verification
6. **Effectiveness Metrics** — Key performance indicators

### Key Metrics

| Metric | Target | Interpretation |
|--------|--------|----------------|
| Retraction removal rate | ~100% | All retractions suppressed |
| Geometry preservation | <5% move count change | Print dimensions maintained |
| Extrusion continuity | Higher is better | Longer consecutive extrusion runs |
| Pressure compliance | >95% | Time spent in safe window |

---

## 🔬 Scientific Usage

### Using `run_log.csv`

The stabilizer generates detailed logs for scientific analysis:

**Columns:**
- `t_s` — Simulated time (seconds)
- `p_hat` — Estimated pressure
- `u_raw` — Raw extrusion rate
- `u_shaped` — Shaped extrusion rate
- `feed_scale` — Feed rate scaling factor
- `action` — Action taken (emit, low_prime, relax_dwell, etc.)

**Use Cases:**
- Plot `p_hat(t)` with `p_y` and `p_max` bounds
- Count `low_prime` and `relax_dwell` interventions
- Analyze feed scaling patterns
- Correlate pressure with print outcomes

### Using Experimental Data

**`print_trials.csv`** enables:
- Survival analysis (Kaplan-Meier curves)
- Onset time distributions
- Completion rate comparisons
- Clog frequency analysis

**`electrical_traces.csv`** enables:
- Electrical yield analysis
- Resistance stability comparisons
- Open-circuit rate calculations

---

## 🛠️ CLI Reference

### Stabilizer

```bash
python3 paste_stabilizer_v2.py \
  --in INPUT.gcode \
  --out OUTPUT.gcode \
  --csv run_log.csv \
  --log changes.log
```

**Options:**
- `--in` — Input G-code file (required)
- `--out` — Output stabilized G-code file (required)
- `--csv` — CSV log file path (required)
- `--log` — Changes log file path (optional)

### Verifier

```bash
python3 verify_stabilizer.py \
  --in INPUT.gcode \
  --out OUTPUT.gcode \
  --csv run_log.csv \
  --p-y 5.0 \
  --p-max 14.0
```

**Options:**
- `--in` — Input G-code file (required)
- `--out` — Stabilized G-code file (required)
- `--csv` — CSV log file path (optional)
- `--p-y` — Yield pressure threshold (default: 5.0)
- `--p-max` — Maximum pressure bound (default: 14.0)
- `--no-plots` — Disable plot generation

### Figure Generator

```bash
python3 generate_10_figures.py \
  --baseline-gcode BASELINE.gcode \
  --stabilized-gcode STABILIZED.gcode \
  --data-dir input \
  --figures 1 2 3 7 21 24 \
  --alpha 8.0 \
  --tau-r 6.0 \
  --p-y 5.0 \
  --p-max 14.0
```

**Options:**
- `--baseline-gcode` — Baseline G-code file (required)
- `--stabilized-gcode` — Stabilized G-code file (required)
- `--data-dir` — Data directory (default: `input/`)
- `--figures` — Figures to generate (default: `all`)
- `--alpha` — Pressure model parameter α (default: 8.0)
- `--tau-r` — Relaxation time constant (default: 6.0)
- `--p-y` — Yield pressure threshold (default: 5.0)
- `--p-max` — Maximum pressure bound (default: 14.0)

---

## 🐛 Troubleshooting

### Common Issues

**Problem**: Output still contains negative E moves  
**Solution**: Check verifier output for specific lines. Your slicer may emit negative E in non-standard formats. Extend suppression rules in `paste_stabilizer_v2.py`.

**Problem**: No shaping happens  
**Solution**: 
- Verify file contains positive extrusion moves
- Check if using absolute extrusion mode (`M82`) — convert to `M83`
- Increase `u_scale` parameter

**Problem**: `p_hat` always stays low  
**Solution**: Increase `prime_total_e`, `lowprime_e`, or `u_scale` in config

**Problem**: Too many relax dwells  
**Solution**: 
- Decrease `u_scale` or increase `p_max`
- Reduce `du_max` to slow pressure ramping

**Problem**: Figures don't display  
**Solution**: 
- Ensure matplotlib backend is configured correctly
- Check that display is available (for GUI systems)
- Use `--save` flag to save instead of display

### Getting Help

1. Check the verification output for specific error messages
2. Review `changes.log` to see what modifications were made
3. Examine `run_log.csv` for pressure and action patterns
4. Compare with example `test.gcode` file

---

## 📚 Recommended Slicer Settings

For best results with paste extrusion:

### Essential Settings
- ✅ **Relative extrusion**: `M83` (not `M82`)
- ✅ **Disable retractions**: Set retraction length to 0
- ✅ **Disable wipe/coast**: Avoids negative E patterns
- ✅ **Larger first-layer height**: Paste requires clearance (1.0-1.5mm)

### Performance Settings
- ✅ **Lower speeds**: 15-25 mm/s (vs 50-100 mm/s for FDM)
- ✅ **Reduced acceleration**: 500-1000 mm/s² (vs 2000+ for FDM)
- ✅ **Lower jerk**: 5-10 mm/s (vs 20+ for FDM)

### Avoid These
- ❌ Retractions (set to 0)
- ❌ Wipe/coast features
- ❌ Combing with negative E
- ❌ High acceleration/jerk

---

## ⚠️ Safety Notes

Before printing with paste:

1. **Dry Run First**
   - Run stabilized file with empty syringe
   - Verify motion paths are correct
   - Check for collisions

2. **Test Priming**
   - Run only the stabilization header section
   - Confirm prime/purge behavior
   - Verify paste flows correctly

3. **Start Conservative**
   - Use conservative paste viscosity
   - Start with larger nozzle (1.2mm+)
   - Monitor first few layers closely

4. **Monitor Pressure**
   - Watch for over-pressure signs (stalling)
   - Check for under-pressure (no flow)
   - Adjust parameters if needed

---

## 📊 Output Files

### Generated Files

| File | Description | Use Case |
|------|-------------|----------|
| `stabilized.gcode` | Stabilized output G-code | Send to printer |
| `run_log.csv` | Pressure and action log | Scientific analysis |
| `changes.log` | Human-readable change log | Debugging, verification |
| `figures/*.png` | Generated plots | Paper figures, analysis |

### Understanding `run_log.csv`

Each row represents one processed G-code command:

```csv
t_s,action,p_hat,u_prev,e_cmd,dt_s,u_raw,u_shaped,feed_scale,in,out
2.0,relax_dwell,19.18,0.0,,0.3,,,,G1 E-4.0,G4 S0.30
```

**Key columns:**
- `t_s` — Time in simulation (seconds)
- `p_hat` — Estimated pressure at this time
- `action` — Action taken (emit, low_prime, relax_dwell, retract_suppressed)
- `feed_scale` — Feed rate scaling applied (1.0 = no change)

---

## 🎓 Advanced Usage

### Custom Configuration

Modify `StabilizerConfig` in `paste_stabilizer_v2.py`:

```python
config = StabilizerConfig(
    alpha=8.0,              # Pressure gain
    tau_r=6.0,              # Relaxation time (s)
    p_y=5.0,                # Yield threshold
    p_max=14.0,             # Maximum pressure
    du_max=0.35,            # Max rate change
    relax_dwell_s=0.30,     # Dwell duration (s)
    lowprime_e=0.6,         # Low-prime extrusion (mm)
    feed_scale_min=0.5,     # Min feed scaling
    feed_scale_max=1.5      # Max feed scaling
)
```

### Batch Processing

Process multiple files:

```bash
for file in *.gcode; do
    python3 paste_stabilizer_v2.py \
      --in "$file" \
      --out "stabilized_${file}" \
      --csv "log_${file%.gcode}.csv"
done
```

### Integration with Other Tools

The stabilizer can be integrated into slicer post-processing:

1. Export G-code from slicer
2. Run stabilizer automatically
3. Send stabilized G-code to printer

---

## 📖 Citation

If you use this software in your research, please cite:

```bibtex
@article{paste_stabilization_2024,
  title={Software-Defined Pressure and Flow Stabilization of Low-Cost Paste Extrusion 3D Printers},
  author={[Your Names]},
  journal={IEEE [Journal Name]},
  year={2024}
}
```

---

## 🤝 Contributing

This is a research codebase. For questions or contributions:

1. Check existing issues and documentation
2. Review the code structure in `code/` directory
3. Test changes with `test.gcode` example file
4. Ensure verification passes for all changes

---

## 📝 License

[Specify your license here - MIT, GPL, etc.]

---

## 🙏 Acknowledgments

- Built for paste extrusion 3D printing research
- Designed for reproducibility and scientific rigor
- Optimized for IEEE publication standards

---

## 📞 Support

For questions or issues:
- Review the troubleshooting section
- Check `code/input/README.md` for data file formats
- Examine example outputs in `results/` directory

---

**Last Updated**: 2024  
**Version**: 2.1  
**Status**: Production Ready ✅
