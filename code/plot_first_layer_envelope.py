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
                               output_path: Path = None,
                               dpi: int = 300,
                               width: float = 0.8):
    """
    Plot first-layer operating envelope heatmap.
    
    Parameters:
    -----------
    first_layer_df : pd.DataFrame
        DataFrame with columns: h_ratio, speed_mmps, success
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
    
    # Create pivot table for heatmap
    # Index: h_ratio (normalized height h_1/d_n)
    # Columns: speed_mmps (speed in mm/s)
    # Values: mean success rate
    pivot = first_layer_df.pivot_table(
        values='success',
        index='h_ratio',
        columns='speed_mmps',
        aggfunc='mean'
    )
    
    # Sort indices for proper display
    pivot = pivot.sort_index(axis=0)  # Sort h_ratio
    pivot = pivot.sort_index(axis=1)  # Sort speed
    
    # Create figure with appropriate size
    # Standard IEEE column width is about 3.5 inches
    # For 0.8\linewidth, use approximately 2.8 inches width
    fig_width = 2.8 if width == 0.8 else 3.5 * width
    fig_height = fig_width * 0.75  # Maintain aspect ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('white')
    
    # Create heatmap with RdYlGn colormap (red-yellow-green)
    # Red = low success (0), Green = high success (1)
    im = ax.imshow(pivot.values, 
                   cmap='RdYlGn', 
                   aspect='auto', 
                   vmin=0, 
                   vmax=1, 
                   origin='lower',
                   interpolation='bilinear')  # Smooth interpolation
    
    # Set ticks and labels
    n_speeds = len(pivot.columns)
    n_heights = len(pivot.index)
    
    # X-axis: Speed (mm/s)
    ax.set_xticks(np.arange(n_speeds))
    ax.set_xticklabels([f'{int(c)}' for c in pivot.columns], 
                       fontsize=10, fontweight='bold', fontfamily='serif')
    
    # Y-axis: Normalized height h_1/d_n
    ax.set_yticks(np.arange(n_heights))
    ax.set_yticklabels([f'{c:.2f}' for c in pivot.index], 
                       fontsize=10, fontweight='bold', fontfamily='serif')
    
    # Labels
    ax.set_xlabel('Speed (mm/s)', fontsize=12, fontweight='bold', fontfamily='serif')
    ax.set_ylabel(r'$h_1/d_n$', fontsize=12, fontweight='bold', fontfamily='serif')
    
    # Add colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax, pad=0.02, aspect=20)
    cbar.set_label('Success Rate', fontsize=11, fontweight='bold', fontfamily='serif')
    cbar.ax.tick_params(labelsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    # Add grid lines for better readability
    ax.set_xticks(np.arange(n_speeds + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_heights + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(False)
    
    # Bold axis spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    
    # Tight layout
    plt.tight_layout()
    
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
                       help='Path to first_layer_sweep.csv file')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output path for saved figure (default: display interactively)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved figure (default: 300)')
    parser.add_argument('--width', type=float, default=0.8,
                       help='Figure width relative to linewidth (default: 0.8)')
    
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
    
    # Output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            script_dir = Path(__file__).parent
            output_path = script_dir / output_path
    
    # Generate plot
    plot_first_layer_envelope(df, output_path, args.dpi, args.width)


if __name__ == '__main__':
    main()
