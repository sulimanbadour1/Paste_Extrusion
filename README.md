# Paste Extrusion Stabilization Layer (Software-Defined Pressure & Flow Shaping)

This project implements the software layer described in the paper:
**Software-Defined Pressure and Flow Stabilization of Low-Cost Paste Extrusion 3D Printers for Structural and Electrical Printing**

## Project Overview

This codebase provides a **software middleware solution** that transforms standard FDM slicer G-code into paste-extrusion-compatible G-code. The system addresses the fundamental challenge that paste materials (conductive inks, structural pastes, biomaterials) behave differently from thermoplastic filaments—they exhibit yield-stress flow, pressure accumulation, and delayed response to extrusion commands.

### What This Code Does

The project consists of several Python modules that work together:

1. **`paste_stabilizer_v2.py`** - Core stabilization engine
   - Reads input G-code from standard slicers (Cura, PrusaSlicer, etc.)
   - Suppresses retractions (negative E moves) that collapse paste pressure
   - Replaces retractions with pressure-stabilizing dwells and micro-primes
   - Maintains internal pressure estimate (`p_hat`) using a first-order model
   - Actively shapes extrusion commands to keep pressure in a safe window
   - Preserves XY/Z geometry when suppressing retractions
   - Outputs stabilized G-code ready for paste printing

2. **`verify_stabilizer.py`** - Quality assurance and verification
   - Validates that stabilization worked correctly
   - Checks for retraction removal, pressure compliance, shaping evidence
   - Generates verification plots and 3D toolpath visualizations
   - Provides statistical analysis of pressure traces

3. **`generate_10_figures.py`** - Paper figure generation
   - Generates 23 publication-ready figures for research papers
   - Includes G-code analysis, survival curves, electrical yield plots
   - Uses experimental data from CSV files (print trials, electrical traces)
   - IEEE-compatible formatting with bold text and professional styling

4. **`stabilizer_gui.py`** - Graphical user interface
   - User-friendly GUI for the entire workflow
   - Tabs for stabilization, verification, visualization, and figure generation
   - File selection and data directory management
   - Interactive figure display and export

5. **`generate_comparison_plots.py`** - Comparison and analysis
   - Side-by-side comparison of original vs stabilized G-code
   - 3D toolpath visualization
   - Extrusion statistics and effectiveness metrics

### Key Innovation

Unlike hardware-based solutions (pressure sensors, closed-loop control), this approach requires **zero hardware modifications**. It works entirely through G-code transformation, making it applicable to any 3D printer that accepts standard G-code commands.

### Use Cases

- **Conductive paste printing** - Silver/carbon paste for electronics
- **Structural paste printing** - Cement, clay, ceramic pastes
- **Biomaterial printing** - Hydrogels, bioinks
- **Research applications** - Reproducible paste extrusion experiments

The goal is to make a low-cost paste extrusion printer usable and repeatable *without hardware modifications* by transforming slicer-generated G-code into paste-stable G-code.

---

## What the stabilizer does

### Core idea
Paste extrusion is not "instantaneous" like FDM. It behaves like a delayed, yield-stress flow with pressure accumulation and relaxation. Standard FDM G-code often contains:
- aggressive retractions (negative E)
- frequent stop/start behavior
- commands that collapse paste pressure and cause clogs

This stabilizer converts that into paste-aware execution by:

1. **Priming ramp + purge line**
   - builds stable pressure before printing real geometry
2. **Retraction suppression**
   - removes negative E moves and replaces them with:
     - a short dwell (pressure stabilization)
     - a micro-prime (restore forward flow)
   - **Preserves XY/Z geometry**: When retractions include position changes, the geometry is maintained in the replacement micro-prime move
3. **Pressure estimation + command shaping (v2)**
   - maintains an internal estimated pressure state `p_hat`
   - shapes extrusion to keep `p_hat` in a stable window:
     - if `p_hat < p_y`: inject low-priming
     - if `p_hat > p_max`: insert a dwell to relax pressure
   - reduces impulsive extrusion changes using a rate limiter (Δu limiting)
4. **Logging for scientific reporting**
   - produces a CSV `run_log.csv` with:
     - `t_s`, `p_hat`, `u_raw`, `u_shaped`, `feed_scale`, `action`
   - supports reproducible plots used in the paper

---

## Files

### Project Structure
```
code/
├── paste_stabilizer_v2.py    # Main stabilizer engine
├── verify_stabilizer.py      # Verification and QA tool
├── generate_10_figures.py    # Paper figure generator (23 figures)
├── generate_comparison_plots.py  # Comparison and analysis tool
├── stabilizer_gui.py         # Graphical user interface
├── paper_figs.py             # Alternative figure generator
├── plot_ut_phat_combined.py  # Combined u(t) and p̂(t) plots
├── 3d_map.py                 # 3D toolpath visualization
├── test.gcode                # Example input G-code
├── input/                    # Input data files directory
│   ├── print_trials.csv      # Print trial experimental data
│   ├── electrical_traces.csv # Electrical measurement data
│   ├── first_layer_sweep.csv # First-layer operating envelope data (optional)
│   └── README.md             # Input files documentation
├── results/                  # Output directory (created automatically)
│   ├── stabilized.gcode      # Stabilized output
│   ├── run_log.csv          # Pressure and action log
│   ├── changes.log          # Change log
│   └── figures/             # Generated plots
│       ├── comparison_*.png  # Comparison plots
│       └── verification_*.png # Verification plots
└── readme.md                 # This file
```

### Main Components

**Core Stabilization Engine**
- `paste_stabilizer_v2.py`  
  Reads input G-code, writes stabilized G-code, produces logs.
  - **Key feature**: Preserves XY/Z geometry when suppressing retractions
  - Implements pressure estimation model (`p_hat`)
  - Active command shaping (feed scaling, dwell insertion, priming)

**Verification and QA**
- `verify_stabilizer.py`  
  Checks that the stabilized output meets expected invariants:
  - Header inserted
  - Retractions removed
  - Modes set correctly (G90/M83)
  - Shaping occurred (feed scaling or pressure actions present)
  - Logs exist and contain plausible values
  - **Enhanced with 3D toolpath visualization and comprehensive plots**

**Figure Generation**
- `generate_10_figures.py`  
  Generates 23 publication-ready figures for research papers:
  - G-code analysis (delta, retractions, timelines)
  - Pressure estimates with bounds
  - Survival curves (extrusion continuity)
  - Operating envelopes and yield maps
  - Electrical trace analysis
  - Width uniformity plots
  - IEEE-compatible formatting with bold text

- `paper_figs.py`  
  Alternative figure generator with different styling options

**Comparison and Analysis**
- `generate_comparison_plots.py`  
  Comprehensive visual comparison tool between original and stabilized G-code:
  - Side-by-side 3D toolpath visualization
  - Retraction vs micro-prime analysis
  - Feed rate comparison
  - Geometry preservation analysis
  - Effectiveness metrics summary
  - Extrusion statistics and continuity analysis

**User Interface**
- `stabilizer_gui.py`  
  Graphical user interface for the entire workflow:
  - Tab 1: G-code stabilization
  - Tab 2: Verification
  - Tab 3: 3D visualization
  - Tab 4: Research plots generation
  - Tab 5: Paper figures (23 figures)
  - Tab 6: Help and documentation
  - Input file selection for CSV data files
  - Interactive figure display

**Visualization Tools**
- `3d_map.py`  
  3D toolpath visualization and mapping
- `plot_ut_phat_combined.py`  
  Combined extrusion rate u(t) and pressure p̂(t) plots

---

## Installation / Requirements

- Python 3.9+ recommended
- Packages:
  - `pandas` (verification + CSV analysis)
  - `matplotlib` (optional for plotting)

Install:
```bash
pip install pandas matplotlib
```

---

## Quick start

### 1) Run the stabilizer on your G-code
```bash
python3 paste_stabilizer_v2.py --in test.gcode --out stabilized.gcode --csv run_log.csv --log changes.log
```

**Outputs:**
- `stabilized.gcode`
- `run_log.csv`
- `changes.log`

### 2) Verify the output automatically
```bash
python3 verify_stabilizer.py --in test.gcode --out stabilized.gcode --csv run_log.csv
```

If all checks pass, you will see:
- PASS/FAIL results
- counts of suppressed retractions
- evidence of shaping interventions
- basic pressure-window compliance statistics

### 3) Compare original vs stabilized G-code (recommended)
```bash
python3 compare_gcode.py test.gcode results/stabilized.gcode
```

This generates comprehensive comparison plots in `results/figures/`:
- 3D toolpath visualization (side-by-side)
- Extrusion statistics
- Retraction vs micro-prime analysis
- Feed rate comparison
- Geometry preservation analysis
- Effectiveness metrics summary

### 4) Generate the paper figure for p_hat (optional)
```bash
python3 plot_phat.py
```

**Note**: All scripts now automatically look for files in the `results/` directory if not found in the current location.

---

## How to interpret verification results

### Expected PASS conditions

- Stabilization header inserted once
- Retractions suppressed (if input has negative E moves)
- Output contains no deposition moves with negative E
- If extrusion exists:
  - some evidence of shaping occurs, such as:
    - modified feed rates on extrusion moves
    - actions logged: `low_prime` and/or `relax_dwell`
  - CSV log is present and includes `p_hat`

### Common "expected warnings"

If your input file contains mostly negative E moves (FDM retractions everywhere), the stabilizer may remove most extrusion and the print may become mostly travel. This indicates the slicer output is incompatible with paste and must be reconfigured.

---

## How to interpret comparison plots

The `compare_gcode.py` tool generates 6 comprehensive plots:

### 1. 3D Toolpath Comparison
- **Left panel**: Original G-code with retractions shown in red (X markers)
- **Right panel**: Stabilized G-code with micro-primes shown in green (O markers)
- **Check**: Both toolpaths should have similar 3D geometry (geometry preservation)

### 2. Extrusion Statistics
- **Histograms**: Distribution of E values and extrusion rates
- **Cumulative plot**: Total extrusion over moves
- **Statistics box**: Detailed counts and metrics

### 3. Retraction vs Micro-prime Analysis
- **Top views**: Spatial distribution of retractions (original) vs micro-primes (stabilized)
- **E value comparison**: Shows retraction E values (negative) vs micro-prime E values (positive)
- **Spatial overlay**: Visual confirmation that micro-primes replace retractions at similar locations

### 4. Feed Rate Comparison
- **Distribution**: How feed rates changed
- **Over moves**: Feed rate progression
- **Statistics**: Mean, median, min, max, std dev
- **Scatter plot**: Direct comparison (points on diagonal = no change)

### 5. Geometry Preservation Analysis
- **Toolpath overlay**: 2D top view showing both paths overlaid (should match closely)
- **Z-height**: Z coordinate progression (should be identical)
- **Cumulative distance**: Total distance traveled (should be similar)
- **Move distances**: Distribution of individual move lengths

### 6. Effectiveness Metrics Summary
- **Move type comparison**: Bar chart showing counts before/after
- **Extrusion continuity**: Longer consecutive extrusion runs = better flow
- **Effectiveness summary**: Key metrics including:
  - Retraction removal rate (should be ~100%)
  - Geometry preservation (move count change should be small)
  - Extrusion continuity improvement (higher is better)

### Key Metrics to Check

✅ **Retraction removal rate**: Should be close to 100%  
✅ **Geometry preservation**: Move count change < 5%  
✅ **Micro-primes added**: Should match number of retractions removed  
✅ **Extrusion continuity**: Stabilized should have longer consecutive extrusion runs  
✅ **Toolpath overlay**: Original and stabilized paths should overlap closely

---

## Recommended slicer settings for paste (baseline)

- Use relative extrusion: `M83`
- Disable retractions (or set to 0)
- Avoid wipe/coast features that create negative E extrusion patterns
- Use larger first-layer height (paste requires clearance)
- Lower speeds and acceleration

---

## Safety notes (printer testing)

Before printing with paste:

- Dry run the stabilized file with empty syringe (motion check)
- Run only the stabilization header section to confirm prime/purge behavior
- Start with conservative paste and nozzle geometry

---

## Scientific usage (paper)

Use `run_log.csv` to plot:
- `p_hat(t)` with `p_y` and `p_max`
- counts of `low_prime` and `relax_dwell`

Use print trial CSV (separate experiment logs) for:
- onset time
- flow continuity duration
- first-layer success rate
- completion rate
- clog rate

---

## CLI reference

### Stabilizer
```bash
python3 paste_stabilizer_v2.py --in INPUT.gcode --out OUTPUT.gcode \
  --csv run_log.csv --log changes.log
```

**Options:**
- `--in`: Input G-code file (required)
- `--out`: Output stabilized G-code file (required)
- `--csv`: CSV log file path (required)
- `--log`: Changes log file path (optional)
- `--no-plots`: Disable plotting in verifier (if used together)

### Verifier
```bash
python3 verify_stabilizer.py --in INPUT.gcode --out OUTPUT.gcode --csv run_log.csv
```

**Options:**
- `--in`: Input G-code file (required)
- `--out`: Stabilized G-code file (required)
- `--csv`: CSV log file path (optional, for pressure analysis)
- `--p_y`: Yield threshold (default: 5.0)
- `--p_max`: Upper pressure bound (default: 14.0)
- `--no-plots`: Disable plot generation

**Outputs:**
- Verification results (PASS/FAIL)
- Plots in `results/figures/`:
  - `verification_3d_toolpath_comparison.png`
  - `verification_3d_toolpath_pressure.png`
  - `verification_pressure_trace.png`
  - `verification_extrusion_comparison.png`
  - `verification_feed_scaling.png`
  - `verification_action_summary.png`

### Comparison Tool
```bash
python3 compare_gcode.py ORIGINAL.gcode STABILIZED.gcode
```

**Outputs:**
- 6 comprehensive comparison plots in `results/figures/`:
  - `comparison_3d_toolpath.png` - 3D toolpath visualization
  - `comparison_statistics.png` - Extrusion statistics
  - `comparison_retractions.png` - Retraction vs micro-prime analysis
  - `comparison_feedrates.png` - Feed rate comparison
  - `comparison_geometry.png` - Geometry preservation analysis
  - `comparison_effectiveness.png` - Stabilizer effectiveness metrics

### Plot Generator
```bash
python3 plot_phat.py
```

Reads `results/run_log.csv` and generates `results/figures/phat_trace.pdf`

---

## Troubleshooting

### Output still contains negative E
Your slicer may emit negative E in ways not captured by simple parsing. Use verifier output to locate lines and extend suppression rules.

### No shaping happens
Your file might contain no positive extrusion moves. Or it uses absolute extrusion mode (`M82`). Convert to `M83` or update parsing.

### p_hat always stays low
Increase `prime_total_e`, `lowprime_e`, or `u_scale` in the config.

### Too many relax dwells
Decrease `u_scale` or increase `p_max` slightly. Reduce `du_max` to slow pressure ramping.

---

## Recent Updates and Improvements

### Geometry Preservation (v2.1)
- **Fixed**: Retraction suppression now preserves XY/Z coordinates when retractions include position changes
- **Impact**: Stabilized G-code maintains the same 3D geometry as the original, ensuring accurate prints
- **Example**: `G1 X10 Y20 E-4.0` → `G4 S0.35` + `G1 X10 Y20 E0.6` (geometry preserved)

### Enhanced Toolpath Extraction
- **Improved**: Better handling of moves with partial coordinates (Z-only or E-only moves)
- **Impact**: More accurate 3D visualization and analysis
- **Files**: `verify_stabilizer.py`, `compare_gcode.py`

### Comprehensive Comparison Tool
- **New**: `compare_gcode.py` provides detailed visual and statistical comparison
- **Features**:
  - 6 different comparison plots showing stabilizer effectiveness
  - Retraction vs micro-prime spatial analysis
  - Feed rate changes visualization
  - Geometry preservation verification
  - Extrusion continuity metrics
  - Effectiveness summary with key metrics

### Enhanced Verification Plots
- **Added**: 3D toolpath visualization in `verify_stabilizer.py`
- **Features**:
  - Side-by-side 3D toolpath comparison
  - Pressure-colored toolpath visualization
  - Retraction visualization in original G-code
  - Extrusion rate coloring

### Path Resolution Improvements
- **Enhanced**: All scripts now intelligently resolve file paths
- **Behavior**: Automatically checks `results/` directory for input/output files
- **Impact**: Easier workflow when files are organized in the `results/` folder

### Output Organization
- **Standardized**: All plots and figures are saved to `results/figures/`
- **Consistent**: CSV logs and output G-code go to `results/` directory
- **Benefit**: Cleaner project structure and easier file management
