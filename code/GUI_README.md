# Paste Stabilizer GUI - Quick Start Guide

## Overview

The GUI application provides an intuitive interface to guide you through the complete paste stabilization workflow.

## Launching the GUI

```bash
cd /Users/mpmp/Desktop/paste_paper/paste_paper_package/data/code
python3 stabilizer_gui.py
```

## Workflow Steps

### Step 1: Stabilize G-code
1. Click on tab **"1. Stabilize G-code"**
2. Click **"Browse..."** to select your input G-code file
3. Review output file paths (defaults are fine)
4. Click **"Run Stabilizer"**
5. Check the output log for success messages

**Outputs:**
- `results/stabilized.gcode` - Ready-to-print G-code
- `results/run_log.csv` - Pressure and action log
- `results/changes.log` - Change summary

### Step 2: Verify Output
1. Click on tab **"2. Verify"**
2. Verify file paths are correct
3. Click **"Run Verification"**
4. Review verification results in the log

**Checks:**
- Header inserted ✓
- Retractions suppressed ✓
- Pressure shaping occurred ✓
- CSV log valid ✓

### Step 3: Visualize
1. Click on tab **"3. Visualize"**
2. Select visualization options:
   - ✓ 3D Toolpath Comparison (recommended)
   - ✓ Detailed Comparison Plots
   - ✓ Statistics Summary
3. Click **"Generate Visualizations"**

**Outputs:**
- `results/figures/3d_comparison.pdf/png` - Beautiful before/after 3D map
- `results/figures/comparison_*.png` - Detailed comparison plots

### Step 4: Research Plots
1. Click on tab **"4. Research Plots"**
2. Click **"Generate Research Plots"**
3. Wait for all figures to be generated

**Outputs:**
All publication-ready figures in `figures/` directory:
- Extrusion onset boxplots
- Flow duration analysis
- Success rates
- Clog frequency
- Pressure simulations
- Electrical trace results
- Comprehensive summary

### Help Tab
- Complete workflow guide
- Key concepts explanation
- Output file descriptions

## Features

### 3D Visualization
The enhanced 3D comparison shows:
- **Left panel (BEFORE)**: Original G-code with retractions marked in red (X markers)
- **Right panel (AFTER)**: Stabilized G-code with micro-primes marked in green (O markers)
- **Color coding**: Extrusion rate (E/mm) - darker = higher rate
- **Geometry preservation**: Both toolpaths maintain the same 3D shape

### Command Line Alternative

If you prefer command line:

```bash
# 1. Stabilize
python3 paste_stabilizer_v2.py --in test.gcode --out results/stabilized.gcode --csv results/run_log.csv

# 2. Verify
python3 verify_stabilizer.py --in test.gcode --out results/stabilized.gcode --csv results/run_log.csv

# 3. Visualize
python3 visualize_3d_comparison.py --original test.gcode --stabilized results/stabilized.gcode --output results/figures/3d_comparison

# 4. Research plots
cd ..
python3 code/generate_research_plots.py
```

## Troubleshooting

### GUI won't start
- Make sure you have tkinter installed: `python3 -m tkinter` (should open a window)
- On macOS: tkinter is usually included with Python

### Files not found
- Make sure you're running from the `code/` directory
- Use absolute paths if needed

### Visualization fails
- Check that both G-code files exist
- Ensure matplotlib is installed: `pip install matplotlib`

## Tips

1. **Always verify** after stabilization to ensure quality
2. **Check the logs** for any warnings or issues
3. **Use 3D visualization** to visually confirm geometry preservation
4. **Generate research plots** last, after all data is ready

