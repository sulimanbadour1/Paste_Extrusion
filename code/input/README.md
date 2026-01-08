# Input Data Files

This folder contains CSV data files used for generating paper figures.

## Required Files

### 1. `print_trials.csv`
Contains print trial data with the following columns:
- `trial`: Trial number
- `condition`: Condition type (baseline, partial, full)
- `onset_s`: Extrusion onset time in seconds
- `flow_duration_s`: Flow duration in seconds
- `first_layer_success`: First layer success (0 or 1)
- `completed`: Print completion status (0 or 1)
- `clogs`: Number of clogs

**Used for:**
- Figure 7: Extrusion continuity survival curve
- Figure 14: Print completion rate
- Figure 15: Extrusion onset time distribution
- Figure 16: Flow interruptions / clogs per print
- Figure 19: Ablation study
- Figure 20: Peak pressure vs failure probability

### 2. `electrical_traces.csv`
Contains electrical trace measurement data with the following columns:
- `trace_id`: Trace identifier
- `condition`: Condition type (baseline, partial, full)
- `length_mm`: Trace length in mm
- `resistance_ohm`: Electrical resistance in ohms (empty for open circuits)
- `open_circuit`: Open circuit indicator (0 or 1)

**Used for:**
- Figure 9: Electrical yield (open-circuit rate)
- Figure 10: Resistance stability (boxplot)
- Figure 17: Electrical resistance comparison
- Figure 19: Ablation study

### 3. `first_layer_sweep.csv` (Optional)
Contains first-layer operating envelope data with columns:
- `h_ratio`: First-layer height ratio
- `speed_mmps`: Speed in mm/s
- `success`: Success rate (0.0 to 1.0)

**Used for:**
- Figure 8: First-layer operating envelope heatmap

## File Location

The code looks for these files in:
1. `code/input/` (preferred location)
2. `code/data/` (fallback location)
3. Custom path specified via `--data-dir` argument

## Notes

- All CSV files should use comma (`,`) as the delimiter
- Missing files will result in warnings and skipped figures
- Files can be placed in either `input/` or `data/` folder
