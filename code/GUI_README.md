# Paste Extrusion Stabilizer - Quick Guide

## Code Structure

The project consists of **4 main Python files**:

1. **`paste_stabilizer_v2.py`** - Core stabilizer engine
   - Transforms FDM G-code into paste-stable G-code
   - Eliminates retractions, adds pressure stabilization
   - Outputs: stabilized G-code, CSV log, changes log

2. **`verify_stabilizer.py`** - Verification tool
   - Validates stabilization output
   - Checks retraction suppression, pressure shaping
   - Reports PASS/FAIL results

3. **`generate_10_figures.py`** - All plotting functionality
   - Generates 13 publication-ready figures (1-12 + effectiveness dashboard)
   - Includes effectiveness metrics and comparisons
   - Displays figures interactively for review before saving

4. **`stabilizer_gui.py`** - Enhanced GUI application
   - Unified interface for all tools
   - Workflow tabs: Stabilize → Verify → Generate Figures
   - Progress tracking and effectiveness metrics

## Quick Start

### Launch GUI
```bash
cd code
python3 stabilizer_gui.py
```

### Workflow (3 Steps)
1. **Stabilize**: Load G-code → Run stabilizer → Get stabilized output
2. **Verify**: Check that stabilization worked correctly
3. **Generate Figures**: Create publication-ready visualizations (Figures 1-13)

### Command Line Alternative
```bash
# Stabilize
python3 paste_stabilizer_v2.py --in test.gcode --out results/stabilized.gcode --csv results/run_log.csv

# Verify
python3 verify_stabilizer.py --in test.gcode --out results/stabilized.gcode --csv results/run_log.csv

# Generate figures
python3 generate_10_figures.py --baseline-gcode test.gcode --stabilized-gcode results/stabilized.gcode --figures all
```

## Output Files

- `results/stabilized.gcode` - Ready-to-print G-code
- `results/run_log.csv` - Pressure and action log
- `results/changes.log` - Change summary
- Figures displayed interactively (save using figure window controls)

## Key Features

- **Retraction Elimination**: Removes all negative E moves
- **Pressure Stabilization**: Maintains pressure within admissible window
- **Effectiveness Dashboard**: Quantitative metrics showing software impact
- **13 Publication Figures**: Comprehensive visualizations for research paper

## Troubleshooting

- **GUI won't start**: Check tkinter: `python3 -m tkinter`
- **Figures don't appear**: Run from terminal to see matplotlib backend messages
- **Files not found**: Ensure you're in the `code/` directory
