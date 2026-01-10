# Effectiveness Figures Guide

This document describes the 12 additional effectiveness figures (Figures 24-35) that highlight stabilizer performance and impact.

## Quick Start

```bash
cd code
python3 generate_effectiveness_figures.py \
  --baseline-gcode test.gcode \
  --stabilized-gcode results/stabilized.gcode \
  --run-log results/run_log.csv \
  --figures all
```

## Figure Descriptions

### Figure 24: Pressure Window Compliance Over Time
**Purpose**: Shows when pressure stays within admissible bounds (p_y to p_max)  
**Key Metric**: Percentage of time in compliance  
**Why Important**: Direct measure of stabilization effectiveness  
**Data Required**: `run_log.csv` with `p_hat` and `t_s` columns

### Figure 25: Action Frequency Analysis
**Purpose**: Shows when and how often the stabilizer intervenes  
**Key Metrics**: Counts of low_prime, relax_dwell, retract_suppressed actions  
**Why Important**: Demonstrates active stabilization throughout the print  
**Data Required**: `run_log.csv` with `action` column

### Figure 26: Feed Rate Scaling Distribution
**Purpose**: Shows how often feed rates are scaled and by how much  
**Key Metrics**: Distribution of feed_scale values, scaling rate percentage  
**Why Important**: Quantifies command shaping intensity  
**Data Required**: `run_log.csv` with `feed_scale` column

### Figure 27: Layer-by-Layer Analysis
**Purpose**: Shows how stabilization metrics vary across layers  
**Key Metrics**: Retractions per layer (baseline) vs actions per layer (stabilized)  
**Why Important**: Reveals if stabilization needs vary by layer  
**Data Required**: G-code files + `run_log.csv`

### Figure 28: Print Time Impact
**Purpose**: Comparison of total print time before/after stabilization  
**Key Metrics**: Time increase in seconds and percentage  
**Why Important**: Quantifies overhead from stabilization  
**Data Required**: G-code files (computed from move times)

### Figure 29: Pressure Recovery Time
**Purpose**: Time to recover pressure after low_prime interventions  
**Key Metrics**: Mean/median recovery time, percentage < 1s, < 2s  
**Why Important**: Shows how quickly the system stabilizes  
**Data Required**: `run_log.csv` with `p_hat` and `action` columns

### Figure 30: Extrusion Rate Variability (CV)
**Purpose**: Coefficient of variation comparison before/after  
**Key Metrics**: CV values, variability reduction percentage  
**Why Important**: Quantifies flow consistency improvement  
**Data Required**: G-code files (computed from u(t) timelines)

### Figure 31: Failure Mode Analysis
**Purpose**: Breakdown of failure types by condition  
**Key Metrics**: Clogs, incomplete prints, electrical failures  
**Why Important**: Identifies which failure modes stabilization addresses  
**Data Required**: `print_trials.csv` + `electrical_traces.csv`

### Figure 32: Micro-prime vs Retraction Magnitude
**Purpose**: Comparison of retraction magnitudes to micro-prime magnitudes  
**Key Metrics**: Correlation, trend line, distribution comparison  
**Why Important**: Shows replacement strategy effectiveness  
**Data Required**: G-code files (extracted from E values)

### Figure 33: Cumulative Extrusion Comparison
**Purpose**: Total material usage over time  
**Key Metrics**: Cumulative E curves for both conditions  
**Why Important**: Shows flow continuity and material usage  
**Data Required**: G-code files (computed from cumulative E)

### Figure 34: Resistance vs Print Length
**Purpose**: Resistance scaling with trace length  
**Key Metrics**: Linear fit coefficients, trend lines by condition  
**Why Important**: Shows if resistance scales linearly (good) or has issues  
**Data Required**: `electrical_traces.csv` with `length_mm` and `resistance_ohm`

### Figure 35: Stabilization Overhead Analysis
**Purpose**: Breakdown of added time from stabilization actions  
**Key Metrics**: Time from dwells, micro-primes, feed scaling  
**Why Important**: Quantifies computational/print time cost  
**Data Required**: G-code files + `run_log.csv`

## Most Important for Paper

**Top 5 Effectiveness Figures:**
1. **Figure 24** - Pressure Window Compliance (direct effectiveness)
2. **Figure 25** - Action Frequency (shows active intervention)
3. **Figure 30** - Extrusion Variability (quantifies consistency)
4. **Figure 31** - Failure Mode Analysis (identifies improvements)
5. **Figure 28** - Print Time Impact (cost analysis)

## Integration with GUI

These figures can be added to the GUI's "Paper Figures" tab. They complement the existing 23 figures by focusing specifically on effectiveness metrics.

## Styling

All figures use IEEE-compatible formatting:
- Bold text throughout (axes labels, tick labels, legends, titles)
- Serif fonts (Times/Times New Roman)
- Professional color palette
- Consistent figure sizing
- Enhanced grid and legend styling
