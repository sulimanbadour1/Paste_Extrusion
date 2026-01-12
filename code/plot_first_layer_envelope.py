#!/usr/bin/env python3
"""
plot_first_layer_envelope.py

Generates the first-layer operating envelope heatmap showing success rate vs.
normalized height (h_1/d_n) and speed. Enhanced with IEEE-compatible styling.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt

# Set professional matplotlib style for IEEE publication
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 12,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
})

def plot_first_layer_envelope(first_layer_df: pd.DataFrame, 
                               baseline_df: pd.DataFrame = None,
                               output_path: Path = None,
                               dpi: int = 300,
                               width: float = 0.8):
    """
    Plot first-layer operating envelope heatmap.
    
    Parameters:
    -----------
    first_layer_df : pd.DataFrame
        DataFrame with columns: h_ratio, speed_mmps, success, [n]
        If 'n' column exists, it will be used to annotate cells
    baseline_df : pd.DataFrame, optional
        Baseline data for comparison (same format as first_layer_df)
    output_path : Path, optional
        Path to save the figure. If None, displays interactively.
    dpi : int
        Resolution for saved figure (default: 300)
    width : float
        Figure width relative to linewidth (default: 0.8 for 0.8\linewidth)
    """
    if first_layer_df is None or len(first_layer_df) == 0:
        print("ERROR: No first layer data provided", flush=True)
        return
    
    # Check required columns
    required_cols = ['h_ratio', 'speed_mmps', 'success']
    missing_cols = [col for col in required_cols if col not in first_layer_df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}", flush=True)
        print(f"Available columns: {list(first_layer_df.columns)}", flush=True)
        return
    
    # Check if n (number of trials) column exists
    has_n = 'n' in first_layer_df.columns
    
    # Create pivot tables for heatmap
    # Index: h_ratio (normalized height h_1/d_n)
    # Columns: speed_mmps (speed in mm/s)
    # Values: mean success rate
    pivot = first_layer_df.pivot_table(
        values='success',
        index='h_ratio',
        columns='speed_mmps',
        aggfunc='mean'
    )
    
    # Get n values if available
    pivot_n = None
    if has_n:
        pivot_n = first_layer_df.pivot_table(
            values='n',
            index='h_ratio',
            columns='speed_mmps',
            aggfunc='mean'  # Should be same for all cells, but use mean for safety
        )
    
    # Sort indices for proper display
    pivot = pivot.sort_index(axis=0)  # Sort h_ratio
    pivot = pivot.sort_index(axis=1)  # Sort speed
    if pivot_n is not None:
        pivot_n = pivot_n.sort_index(axis=0)
        pivot_n = pivot_n.sort_index(axis=1)
    
    # Determine figure layout
    if baseline_df is not None and len(baseline_df) > 0:
        # Two-panel comparison
        fig_width = 8.5 if width == 0.8 else 10.6 * width  # Even wider for bigger cells
        fig_height = fig_width * 0.55  # Better aspect ratio
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), 
                                       gridspec_kw={'wspace': 0.3})  # More space between panels
        axes = [ax1, ax2]
        
        # Create baseline pivot
        baseline_pivot = baseline_df.pivot_table(
            values='success',
            index='h_ratio',
            columns='speed_mmps',
            aggfunc='mean'
        )
        baseline_pivot = baseline_pivot.sort_index(axis=0)
        baseline_pivot = baseline_pivot.sort_index(axis=1)
        
        pivots = [baseline_pivot, pivot]
        titles = ['Baseline', 'Shaped']
    else:
        # Single panel - significantly increased size for bigger cells
        fig_width = 4.5 if width == 0.8 else 5.6 * width  # Increased from 3.5
        fig_height = fig_width * 0.9  # Taller for better cell proportions
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]
        pivots = [pivot]
        titles = [None]
    
    fig.patch.set_facecolor('white')
    
    # Plot each panel
    for idx, (ax, pivot_data, title) in enumerate(zip(axes, pivots, titles)):
        # Create heatmap with colorblind-safe sequential colormap (viridis)
        # Dark = low success (0), Bright = high success (1)
        im = ax.imshow(pivot_data.values, 
                       cmap='viridis', 
                       aspect='auto', 
                       vmin=0, 
                       vmax=1, 
                       origin='lower',
                       interpolation='nearest')  # Discrete blocks, no interpolation
        
        # Set ticks and labels
        n_speeds = len(pivot_data.columns)
        n_heights = len(pivot_data.index)
        
        # X-axis: Speed (mm/s) - larger font
        ax.set_xticks(np.arange(n_speeds))
        ax.set_xticklabels([f'{int(c)}' for c in pivot_data.columns], 
                           fontsize=12, fontweight='bold', fontfamily='serif')  # Increased from 10
        
        # Y-axis: Normalized height h_1/d_n - larger font
        ax.set_yticks(np.arange(n_heights))
        ax.set_yticklabels([f'{c:.2f}' for c in pivot_data.index], 
                           fontsize=12, fontweight='bold', fontfamily='serif')  # Increased from 10
        
        # Labels - larger font
        ax.set_xlabel('Speed (mm/s)', fontsize=13, fontweight='bold', fontfamily='serif')  # Increased from 12
        if title is None or idx == 0:  # Only label leftmost y-axis
            ax.set_ylabel(r'$h_1/d_n$', fontsize=13, fontweight='bold', fontfamily='serif')  # Increased from 12
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=13, fontweight='bold', fontfamily='serif', pad=8)  # Increased from 12
        
        # Annotate cells with success rate and n (if available) - larger text
        # Use pivot_n only for the shaped condition (last panel)
        current_pivot_n = pivot_n if (idx == len(axes) - 1) else None
        for i in range(n_heights):
            for j in range(n_speeds):
                val = pivot_data.iloc[i, j]
                if not np.isnan(val):
                    # Format success rate
                    text = f'{val:.2f}'
                    # Add n if available (only for shaped condition)
                    if current_pivot_n is not None and i < len(current_pivot_n.index) and j < len(current_pivot_n.columns):
                        try:
                            n_val = current_pivot_n.iloc[i, j]
                            if not np.isnan(n_val):
                                text += f'\n(n={int(n_val)})'
                        except:
                            pass
                    ax.text(j, i, text, 
                           ha='center', va='center',
                           fontsize=13, fontweight='bold', fontfamily='serif',  # Increased from 11
                           color='white' if val > 0.5 else 'black',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='none', edgecolor='none', alpha=0.7) if val > 0.5 else None)
        
        # Add grid lines for better readability (cell boundaries) - thicker and more visible
        ax.set_xticks(np.arange(n_speeds + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_heights + 1) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5, alpha=1.0)  # Thicker grid lines
        ax.set_axisbelow(False)
        
        # Bold axis spines - thicker for better visibility
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Improve spacing and padding
        ax.tick_params(axis='both', which='major', pad=8)  # More padding around ticks
    
    # Add shared colorbar (only for last axis)
    if len(axes) == 1:
        cbar_ax = axes[0]
    else:
        cbar_ax = axes[-1]
    
    cbar = plt.colorbar(im, ax=cbar_ax, pad=0.02, aspect=20)
    cbar_label = 'Success Rate'
    if has_n:
        # Get unique n values to show in label
        n_vals = first_layer_df['n'].unique() if has_n else []
        if len(n_vals) == 1:
            cbar_label += f' (n={int(n_vals[0])})'
        elif len(n_vals) > 1:
            cbar_label += f' (n varies)'
    cbar.set_label(cbar_label, fontsize=12, fontweight='bold', fontfamily='serif')  # Increased from 11
    cbar.ax.tick_params(labelsize=11)  # Increased from 9
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    # Tight layout with extra padding for better appearance
    plt.tight_layout(pad=2.0)  # Increased padding
    
    # Save or display
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"[OK] Saved: {output_path}", flush=True)
    else:
        plt.show()
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Generate first-layer operating envelope heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from CSV and save
  python3 plot_first_layer_envelope.py --input code/input/first_layer_sweep.csv --output first_layer_operating_envelope.png
  
  # Display interactively
  python3 plot_first_layer_envelope.py --input code/input/first_layer_sweep.csv
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to first_layer_sweep.csv file (shaped condition)')
    parser.add_argument('--baseline', '-b', type=str, default=None,
                       help='Path to baseline first_layer_sweep.csv file (optional, for comparison)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for saved figure (default: display interactively)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure (default: 300)')
    parser.add_argument('--width', type=float, default=0.8,
                       help='Figure width relative to linewidth (default: 0.8 for 0.8*linewidth)')
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = Path(args.input)
    if not input_path.is_absolute():
        script_dir = Path(__file__).parent
        input_path = script_dir / input_path
    
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", flush=True)
        sys.exit(1)
    
    # Read data
    print(f"Reading first-layer sweep data: {input_path}", flush=True)
    try:
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} data points", flush=True)
        print(f"Columns: {list(df.columns)}", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}", flush=True)
        sys.exit(1)
    
    # Read baseline data if provided
    baseline_df = None
    if args.baseline:
        baseline_path = Path(args.baseline)
        if not baseline_path.is_absolute():
            script_dir = Path(__file__).parent
            baseline_path = script_dir / baseline_path
        
        if baseline_path.exists():
            print(f"Reading baseline data: {baseline_path}", flush=True)
            try:
                baseline_df = pd.read_csv(baseline_path)
                print(f"Loaded {len(baseline_df)} baseline data points", flush=True)
            except Exception as e:
                print(f"WARNING: Failed to read baseline CSV: {e}", flush=True)
        else:
            print(f"WARNING: Baseline file not found: {baseline_path}", flush=True)
    
    # Output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent
            output_path = script_dir / output_path
    
    # Generate plot
    plot_first_layer_envelope(df, baseline_df, output_path, args.dpi, args.width)


if __name__ == '__main__':
    main()
