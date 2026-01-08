#!/usr/bin/env python3
"""
GUI Application for Paste Extrusion Stabilizer
Provides an intuitive interface to guide users through the stabilization workflow.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import subprocess
import sys
import os
import importlib.util

class StabilizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Paste Extrusion Stabilizer - Enhanced GUI")
        self.root.geometry("1100x800")
        
        # Configure style
        self.setup_style()
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar(value="results/stabilized.gcode")
        self.csv_file = tk.StringVar(value="results/run_log.csv")
        self.log_file = tk.StringVar(value="results/changes.log")
        
        # Input data files for figures
        self.print_trials_file = tk.StringVar(value="input/print_trials.csv")
        self.electrical_traces_file = tk.StringVar(value="input/electrical_traces.csv")
        self.first_layer_file = tk.StringVar(value="input/first_layer_sweep.csv")
        self.data_dir = tk.StringVar(value="input")  # Data directory path
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Stabilization
        self.create_stabilization_tab()
        
        # Tab 2: Verification
        self.create_verification_tab()
        
        # Tab 3: Visualization
        self.create_visualization_tab()
        
        # Tab 4: Research Plots
        self.create_research_plots_tab()
        
        # Tab 5: 10 Paper Figures
        self.create_10_figures_tab()
        
        # Tab 6: Help
        self.create_help_tab()
    
    def setup_style(self):
        """Configure modern ttk style."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='#2ca02c', font=('Arial', 10, 'bold'))
        style.configure('Warning.TLabel', foreground='#ff7f0e', font=('Arial', 10))
        style.configure('Accent.TButton', font=('Arial', 11, 'bold'))
    
    def create_stabilization_tab(self):
        """Tab 1: Run Stabilizer"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="1. Stabilize G-code")
        
        # Title
        title = ttk.Label(frame, text="Step 1: Stabilize Your G-code", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        # Description
        desc = ttk.Label(frame, 
                        text="Transform your slicer G-code into paste-stable G-code.\n"
                             "This adds priming, suppresses retractions, and shapes extrusion commands.",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # Input file selection
        input_frame = ttk.LabelFrame(frame, text="Input G-code File", padding=10)
        input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(input_frame, text="Select input G-code:").pack(anchor=tk.W)
        input_entry_frame = ttk.Frame(input_frame)
        input_entry_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(input_entry_frame, textvariable=self.input_file, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(input_entry_frame, text="Browse...", 
                  command=self.browse_input_file).pack(side=tk.LEFT, padx=5)
        
        # Output files
        output_frame = ttk.LabelFrame(frame, text="Output Files", padding=10)
        output_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(output_frame, text="Stabilized G-code:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.output_file, width=50).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(output_frame, text="CSV log:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.csv_file, width=50).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(output_frame, text="Changes log:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(output_frame, textvariable=self.log_file, width=50).grid(row=2, column=1, padx=5, pady=2)
        
        # Run button
        run_btn = ttk.Button(frame, text="Run Stabilizer", 
                            command=self.run_stabilizer, style="Accent.TButton")
        run_btn.pack(pady=20)
        
        # Output log
        log_frame = ttk.LabelFrame(frame, text="Output Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.stabilizer_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.stabilizer_log.pack(fill=tk.BOTH, expand=True)
    
    def create_verification_tab(self):
        """Tab 2: Verify Output"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="2. Verify")
        
        title = ttk.Label(frame, text="Step 2: Verify Stabilization", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        desc = ttk.Label(frame, 
                        text="Check that the stabilized G-code meets all requirements:\n"
                             "• Header inserted\n• Retractions suppressed\n• Pressure shaping occurred",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # File selection
        verify_frame = ttk.LabelFrame(frame, text="Files", padding=10)
        verify_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.verify_input = tk.StringVar(value="test.gcode")
        self.verify_output = tk.StringVar(value="results/stabilized.gcode")
        self.verify_csv = tk.StringVar(value="results/run_log.csv")
        
        ttk.Label(verify_frame, text="Original G-code:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(verify_frame, textvariable=self.verify_input, width=50).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(verify_frame, text="Stabilized G-code:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(verify_frame, textvariable=self.verify_output, width=50).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(verify_frame, text="CSV log:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(verify_frame, textvariable=self.verify_csv, width=50).grid(row=2, column=1, padx=5, pady=2)
        
        # Run verification
        verify_btn = ttk.Button(frame, text="Run Verification", 
                               command=self.run_verification)
        verify_btn.pack(pady=20)
        
        # Verification log
        verify_log_frame = ttk.LabelFrame(frame, text="Verification Results", padding=10)
        verify_log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.verification_log = scrolledtext.ScrolledText(verify_log_frame, height=10, wrap=tk.WORD)
        self.verification_log.pack(fill=tk.BOTH, expand=True)
    
    def create_visualization_tab(self):
        """Tab 3: 3D Visualization"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="3. Visualize")
        
        title = ttk.Label(frame, text="Step 3: Create 3D Comparison", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        desc = ttk.Label(frame, 
                        text="Generate beautiful 3D visualizations comparing original and stabilized toolpaths.\n"
                             "Shows retractions (red X) vs micro-primes (green O) and extrusion rates.",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # File selection
        viz_frame = ttk.LabelFrame(frame, text="Files", padding=10)
        viz_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.viz_original = tk.StringVar(value="test.gcode")
        self.viz_stabilized = tk.StringVar(value="results/stabilized.gcode")
        self.viz_output = tk.StringVar(value="results/figures/3d_comparison")
        
        ttk.Label(viz_frame, text="Original G-code:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(viz_frame, textvariable=self.viz_original, width=50).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(viz_frame, text="Stabilized G-code:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(viz_frame, textvariable=self.viz_stabilized, width=50).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(viz_frame, text="Output path:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(viz_frame, textvariable=self.viz_output, width=50).grid(row=2, column=1, padx=5, pady=2)
        
        # Visualization options
        options_frame = ttk.LabelFrame(frame, text="Visualization Options", padding=10)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.viz_3d = tk.BooleanVar(value=True)
        self.viz_comparison = tk.BooleanVar(value=True)
        self.viz_stats = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="3D Toolpath Comparison", 
                       variable=self.viz_3d).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Detailed Comparison Plots", 
                       variable=self.viz_comparison).pack(anchor=tk.W)
        ttk.Checkbutton(options_frame, text="Statistics Summary", 
                       variable=self.viz_stats).pack(anchor=tk.W)
        
        # Run visualization
        viz_btn = ttk.Button(frame, text="Generate Visualizations", 
                            command=self.run_visualization)
        viz_btn.pack(pady=20)
        
        # Visualization log
        viz_log_frame = ttk.LabelFrame(frame, text="Visualization Output", padding=10)
        viz_log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.visualization_log = scrolledtext.ScrolledText(viz_log_frame, height=8, wrap=tk.WORD)
        self.visualization_log.pack(fill=tk.BOTH, expand=True)
    
    def create_research_plots_tab(self):
        """Tab 4: Research Plots"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="4. Research Plots")
        
        title = ttk.Label(frame, text="Step 4: Generate Research Figures", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        desc = ttk.Label(frame, 
                        text="Generate reviewer-proof figures for your research paper.\n"
                             "Figures are displayed interactively - save them manually using the figure window controls.",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # Input data files selection
        input_files_frame = ttk.LabelFrame(frame, text="Input Data Files", padding=10)
        input_files_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Print trials CSV
        ttk.Label(input_files_frame, text="Print Trials CSV:").grid(row=0, column=0, sticky=tk.W, pady=2)
        print_trials_entry = ttk.Entry(input_files_frame, textvariable=self.print_trials_file, width=50)
        print_trials_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(input_files_frame, text="Browse...", 
                  command=lambda: self.browse_file(self.print_trials_file, "Select Print Trials CSV", [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=0, column=2, padx=5, pady=2)
        
        # Electrical traces CSV
        ttk.Label(input_files_frame, text="Electrical Traces CSV:").grid(row=1, column=0, sticky=tk.W, pady=2)
        electrical_entry = ttk.Entry(input_files_frame, textvariable=self.electrical_traces_file, width=50)
        electrical_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(input_files_frame, text="Browse...", 
                  command=lambda: self.browse_file(self.electrical_traces_file, "Select Electrical Traces CSV", [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=1, column=2, padx=5, pady=2)
        
        # First layer sweep CSV (optional)
        ttk.Label(input_files_frame, text="First Layer Sweep CSV (optional):").grid(row=2, column=0, sticky=tk.W, pady=2)
        first_layer_entry = ttk.Entry(input_files_frame, textvariable=self.first_layer_file, width=50)
        first_layer_entry.grid(row=2, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(input_files_frame, text="Browse...", 
                  command=lambda: self.browse_file(self.first_layer_file, "Select First Layer Sweep CSV", [("CSV files", "*.csv"), ("All files", "*.*")])).grid(row=2, column=2, padx=5, pady=2)
        
        # Data directory (alternative to individual files)
        ttk.Label(input_files_frame, text="Data Directory (or use files above):").grid(row=3, column=0, sticky=tk.W, pady=2)
        data_dir_entry = ttk.Entry(input_files_frame, textvariable=self.data_dir, width=50)
        data_dir_entry.grid(row=3, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(input_files_frame, text="Browse...", 
                  command=lambda: self.browse_directory(self.data_dir)).grid(row=3, column=2, padx=5, pady=2)
        
        input_files_frame.columnconfigure(1, weight=1)
        
        # Plot options
        options_frame = ttk.LabelFrame(frame, text="Figure Sets", padding=10)
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.plot_basic = tk.BooleanVar(value=True)
        self.plot_advanced = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(options_frame, text="Figures 1-3: G-code Analysis (G-code delta, command timeline, pressure estimate)", 
                       variable=self.plot_basic).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(options_frame, text="Figures 4-7: Experimental Data (survival curve, operability map, clog correlation, electrical yield)", 
                       variable=self.plot_advanced).pack(anchor=tk.W, pady=2)
        
        # Info
        info_frame = ttk.LabelFrame(frame, text="Figure Descriptions", padding=10)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_text = ("Figures 1-3 (G-code Analysis):\n"
                    "• Figure 1: G-code delta (retractions, dwells, E deltas)\n"
                    "• Figure 2: Command timeline u(t) before vs after\n"
                    "• Figure 3: Model-based pressure estimate p̂(t) with bounds\n\n"
                    "Figures 4-7 (Experimental Data - requires print_trials.csv):\n"
                    "• Figure 4: Extrusion continuity survival curve\n"
                    "• Figure 5: First-layer operability map\n"
                    "• Figure 6: Clog events vs retraction count\n"
                    "• Figure 7: Conductive trace yield + resistance stability\n\n"
                    "Note: All figures are displayed on screen. Use figure window controls to save manually.")
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, font=("Courier", 9)).pack(anchor=tk.W)
        
        # Buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(pady=10)
        
        plots_btn = ttk.Button(buttons_frame, text="Generate All Selected Plots", 
                              command=self.run_all_research_plots, style="Accent.TButton")
        plots_btn.pack(side=tk.LEFT, padx=5)
        
        basic_btn = ttk.Button(buttons_frame, text="Basic Only", 
                              command=self.run_research_plots)
        basic_btn.pack(side=tk.LEFT, padx=5)
        
        advanced_btn = ttk.Button(buttons_frame, text="Advanced Only", 
                                 command=self.run_advanced_plots)
        advanced_btn.pack(side=tk.LEFT, padx=5)
        
        # Research plots log
        plots_log_frame = ttk.LabelFrame(frame, text="Output", padding=10)
        plots_log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.research_log = scrolledtext.ScrolledText(plots_log_frame, height=10, wrap=tk.WORD)
        self.research_log.pack(fill=tk.BOTH, expand=True)
    
    def create_10_figures_tab(self):
        """Tab 5: Generate Paper Figures"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="5. Paper Figures")
        
        title = ttk.Label(frame, text="Generate Paper Figures", 
                         font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        desc = ttk.Label(frame, 
                        text="Generate figures for your research paper (23 available).\n"
                             "Figures are displayed interactively - save them manually using figure window controls.",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # Input data files selection (shared with Research Plots tab)
        input_files_frame = ttk.LabelFrame(frame, text="Input Data Files (Optional - uses same as Research Plots tab)", padding=10)
        input_files_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_label = ttk.Label(input_files_frame, 
                              text="Input files are configured in the 'Research Plots' tab.\n"
                                   "The data directory or individual CSV files can be set there.",
                              justify=tk.LEFT, font=("Arial", 9))
        info_label.pack(anchor=tk.W, pady=5)
        
        # Figure selection
        selection_frame = ttk.LabelFrame(frame, text="Select Figures to Generate", padding=10)
        selection_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Create checkboxes for each figure
        self.fig_vars = {}
        fig_descriptions = {
            '1': 'Fig. 1 — G-code modification summary (delta)',
            '2': 'Fig. 2 — Retraction suppression histogram',
            '3': 'Fig. 3 — Extrusion-rate proxy timeline u(t) (baseline)',
            '4': 'Fig. 4 — Extrusion-rate proxy timeline u(t) (stabilized)',
            '5': 'Fig. 5 — Pressure estimate p̂(t) with bounds (baseline)',
            '6': 'Fig. 6 — Pressure estimate p̂(t) with bounds (stabilized)',
            '7': 'Fig. 7 — Extrusion continuity survival curve',
            '8': 'Fig. 8 — First-layer operating envelope heatmap',
            '9': 'Fig. 9 — Electrical yield (open-circuit rate)',
            '10': 'Fig. 10 — Resistance stability (boxplot)',
            '11': 'Fig. 11 — 3D Toolpath Comparison (Before/After)',
            '12': 'Fig. 12 — 3D Extrusion Rate Map',
            '13': 'Fig. 13 — Effectiveness Dashboard',
            '14': 'Fig. 14 — Print Completion Rate (Executive KPI)',
            '15': 'Fig. 15 — Extrusion Onset Time Distribution',
            '16': 'Fig. 16 — Flow Interruptions / Clogs per Print',
            '17': 'Fig. 17 — Electrical Resistance Comparison',
            '18': 'Fig. 18 — Middleware Pipeline Diagram',
            '19': 'Fig. 19 — Ablation Study',
            '20': 'Fig. 20 — Peak Pressure vs Failure Probability',
            '21': 'Fig. 21 — Extrusion Width Uniformity',
            '22': 'Fig. 22 — Energy / Motor Load Proxy',
            '23': 'Fig. 23 — Time-Lapse Frame with Flow Annotation'
        }
        
        # Three columns of checkboxes
        checkbox_frame = ttk.Frame(selection_frame)
        checkbox_frame.pack(fill=tk.X)
        
        col1_frame = ttk.Frame(checkbox_frame)
        col1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        col2_frame = ttk.Frame(checkbox_frame)
        col2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        col3_frame = ttk.Frame(checkbox_frame)
        col3_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        for i, (fig_num, desc) in enumerate(fig_descriptions.items()):
            var = tk.BooleanVar(value=True)
            self.fig_vars[fig_num] = var
            # Split into 3 columns: 1-8, 9-16, 17-23
            if int(fig_num) <= 8:
                parent = col1_frame
            elif int(fig_num) <= 16:
                parent = col2_frame
            else:
                parent = col3_frame
            ttk.Checkbutton(parent, text=f"{desc}", variable=var).pack(anchor=tk.W, pady=2)
        
        # Select all / Deselect all buttons
        select_buttons_frame = ttk.Frame(selection_frame)
        select_buttons_frame.pack(pady=5)
        ttk.Button(select_buttons_frame, text="Select All", 
                  command=lambda: [v.set(True) for v in self.fig_vars.values()]).pack(side=tk.LEFT, padx=5)
        ttk.Button(select_buttons_frame, text="Deselect All", 
                  command=lambda: [v.set(False) for v in self.fig_vars.values()]).pack(side=tk.LEFT, padx=5)
        
        # Generate button
        generate_btn = ttk.Button(frame, text="Generate Selected Figures", 
                                 command=self.run_10_figures, style="Accent.TButton")
        generate_btn.pack(pady=20)
        
        # Output log
        log_frame = ttk.LabelFrame(frame, text="Output", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.figures_10_log = scrolledtext.ScrolledText(log_frame, height=10, wrap=tk.WORD)
        self.figures_10_log.pack(fill=tk.BOTH, expand=True)
    
    def create_help_tab(self):
        """Tab 6: Help and Documentation"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Help")
        
        title = ttk.Label(frame, text="Workflow Guide", font=("Arial", 14, "bold"))
        title.pack(pady=10)
        
        help_text = """
WORKFLOW ORDER:

1. STABILIZE G-CODE
   • Select your input G-code file (from slicer)
   • Click "Run Stabilizer"
   • Output: stabilized.gcode, run_log.csv, changes.log

2. VERIFY OUTPUT
   • Verify that stabilization worked correctly
   • Checks: header inserted, retractions suppressed, shaping occurred
   • Generates verification plots

3. VISUALIZE
   • Create 3D comparison maps (before/after)
   • Generate detailed comparison plots
   • View statistics and effectiveness metrics

4. RESEARCH PLOTS
   • Generate all figures for your paper
   • Includes print trials, electrical results, pressure analysis
   • All figures formatted for IEEE paper format

KEY CONCEPTS:

• Pressure Estimation: The stabilizer maintains an internal pressure estimate (p_hat)
• Command Shaping: Extrusion commands are modified to keep pressure in safe window
• Retraction Suppression: Negative E moves are replaced with dwell + micro-prime
• Geometry Preservation: XY/Z coordinates are preserved when suppressing retractions

OUTPUT FILES:

• results/stabilized.gcode - Stabilized G-code ready for printing
• results/run_log.csv - Pressure and action log (for analysis)
• results/changes.log - Human-readable change log
• results/figures/ - All generated plots and visualizations

For more information, see readme.md
        """
        
        help_label = ttk.Label(frame, text=help_text, justify=tk.LEFT, font=("Courier", 10))
        help_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    def browse_input_file(self):
        filename = filedialog.askopenfilename(
            title="Select Input G-code File",
            filetypes=[("G-code files", "*.gcode"), ("All files", "*.*")]
        )
        if filename:
            self.input_file.set(filename)
    
    def browse_file(self, var, title, filetypes):
        """Browse for a file and set the variable."""
        filename = filedialog.askopenfilename(
            title=title,
            filetypes=filetypes
        )
        if filename:
            # Convert to relative path if in code directory
            code_dir = Path(__file__).parent
            try:
                rel_path = Path(filename).relative_to(code_dir)
                var.set(str(rel_path))
            except ValueError:
                var.set(filename)
    
    def browse_directory(self, var):
        """Browse for a directory and set the variable."""
        dirname = filedialog.askdirectory(title="Select Data Directory")
        if dirname:
            # Convert to relative path if in code directory
            code_dir = Path(__file__).parent
            try:
                rel_path = Path(dirname).relative_to(code_dir)
                var.set(str(rel_path))
            except ValueError:
                var.set(dirname)
    
    def run_stabilizer(self):
        """Run the stabilizer script."""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input G-code file.")
            return
        
        input_path = Path(self.input_file.get())
        if not input_path.exists():
            messagebox.showerror("Error", f"Input file not found: {input_path}")
            return
        
        self.stabilizer_log.delete(1.0, tk.END)
        self.stabilizer_log.insert(tk.END, "Running stabilizer...\n\n")
        self.root.update()
        
        try:
            cmd = [
                sys.executable, "paste_stabilizer_v2.py",
                "--in", str(input_path),
                "--out", self.output_file.get(),
                "--csv", self.csv_file.get(),
                "--log", self.log_file.get()
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            
            self.stabilizer_log.insert(tk.END, result.stdout)
            if result.stderr:
                self.stabilizer_log.insert(tk.END, "\n\nSTDERR:\n" + result.stderr)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "Stabilization completed successfully!")
            else:
                messagebox.showerror("Error", f"Stabilization failed with return code {result.returncode}")
        
        except Exception as e:
            self.stabilizer_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to run stabilizer: {str(e)}")
    
    def run_verification(self):
        """Run verification script."""
        self.verification_log.delete(1.0, tk.END)
        self.verification_log.insert(tk.END, "Running verification...\n\n")
        self.root.update()
        
        try:
            cmd = [
                sys.executable, "verify_stabilizer.py",
                "--in", self.verify_input.get(),
                "--out", self.verify_output.get(),
                "--csv", self.verify_csv.get()
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
            
            self.verification_log.insert(tk.END, result.stdout)
            if result.stderr:
                self.verification_log.insert(tk.END, "\n\nSTDERR:\n" + result.stderr)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "Verification completed!")
            else:
                messagebox.showwarning("Warning", "Verification found issues. Check the log.")
        
        except Exception as e:
            self.verification_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to run verification: {str(e)}")
    
    def run_visualization(self):
        """Run visualization scripts."""
        self.visualization_log.delete(1.0, tk.END)
        self.visualization_log.insert(tk.END, "Generating visualizations...\n\n")
        self.root.update()
        
        try:
            # Get file paths and resolve them
            code_dir = Path(__file__).parent
            original_gcode = self.viz_original.get() or "test.gcode"
            stabilized_gcode = self.viz_stabilized.get() or "results/stabilized.gcode"
            
            original_path = code_dir / original_gcode
            if not original_path.exists() and original_gcode == "test.gcode":
                original_path = code_dir / "test_gcode" / "test.gcode"
            stabilized_path = code_dir / stabilized_gcode
            
            if not original_path.exists():
                self.visualization_log.insert(tk.END, f"ERROR: Original G-code not found: {original_gcode}\n")
                messagebox.showerror("Error", f"Original G-code not found: {original_gcode}")
                return
            
            if not stabilized_path.exists():
                self.visualization_log.insert(tk.END, f"ERROR: Stabilized G-code not found: {stabilized_gcode}\n")
                messagebox.showerror("Error", f"Stabilized G-code not found: {stabilized_gcode}")
                return
            
            figures_to_generate = []
            
            # 3D comparison
            if self.viz_3d.get():
                self.visualization_log.insert(tk.END, "Creating 3D comparison...\n")
                figures_to_generate.append("11")
            
            # Detailed comparison (use 3D extrusion rate map)
            if self.viz_comparison.get():
                self.visualization_log.insert(tk.END, "Creating detailed comparison...\n")
                figures_to_generate.append("12")
            
            if not figures_to_generate:
                messagebox.showwarning("No Selection", "Please select at least one visualization option.")
                return
            
            # Use generate_10_figures.py for 3D visualizations
            cmd = [
                sys.executable, "generate_10_figures.py",
                "--baseline-gcode", str(original_path),
                "--stabilized-gcode", str(stabilized_path),
                "--data-dir", str(code_dir / "data"),
                "--figures"
            ] + figures_to_generate
            
            self.visualization_log.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
            self.visualization_log.insert(tk.END, "NOTE: Figures will open in separate windows.\n")
            self.visualization_log.insert(tk.END, "Close the figure windows to continue.\n\n")
            self.root.update()
            
            # CRITICAL: Don't capture stdout/stderr - matplotlib needs direct access to display
            # Run the process without capturing output so figure windows can appear
            self.visualization_log.insert(tk.END, "Starting figure generation...\n")
            self.visualization_log.insert(tk.END, "Figure windows will appear on your screen.\n")
            self.visualization_log.insert(tk.END, "Close each window to proceed to the next figure.\n\n")
            self.root.update()
            
            # Run without capturing output - this is essential for matplotlib to display windows
            import os
            env = os.environ.copy()
            
            # Run the process - matplotlib will display windows directly
            process = subprocess.Popen(cmd, cwd=code_dir, env=env)
            
            # Wait for completion
            returncode = process.wait()
            
            if returncode == 0:
                self.visualization_log.insert(tk.END, "\n✓ Figure generation completed!\n")
                messagebox.showinfo("Success", 
                                  f"Visualizations generated!\n\n"
                                  "Figures should have been displayed in separate windows.\n"
                                  "If figures didn't appear, try running from terminal:\n"
                                  f"{' '.join(cmd)}\n\n"
                                  "Use the figure window controls to save them manually.")
            else:
                self.visualization_log.insert(tk.END, f"\n⚠ Process exited with code {returncode}\n")
                messagebox.showwarning("Warning", 
                                     f"Process exited with code {returncode}.\n"
                                     "Figures may still have been displayed.\n"
                                     "Check terminal output for details.")
            
            # Statistics summary
            if self.viz_stats.get():
                self.visualization_log.insert(tk.END, "\nGenerating statistics summary...\n")
                self.visualization_log.insert(tk.END, "Statistics available in verification tab.\n")
        
        except Exception as e:
            self.visualization_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate visualizations: {str(e)}")
    
    def run_research_plots(self):
        """Run paper figures script (Figures 1-3: G-code analysis)."""
        self.research_log.delete(1.0, tk.END)
        self.research_log.insert(tk.END, "Generating paper figures (Figures 1-3)...\n\n")
        self.root.update()
        
        try:
            # Get file paths from GUI
            baseline_gcode = self.verify_input.get() or self.viz_original.get() or "test.gcode"
            stabilized_gcode = self.verify_output.get() or self.viz_stabilized.get() or "results/stabilized.gcode"
            stabilized_csv = self.verify_csv.get() or self.csv_file.get() or "results/run_log.csv"
            
            # Resolve paths relative to code directory
            code_dir = Path(__file__).parent
            baseline_path = code_dir / baseline_gcode
            if not baseline_path.exists() and baseline_gcode == "test.gcode":
                # Try alternative location
                baseline_path = code_dir / "test_gcode" / "test.gcode"
            stabilized_path = code_dir / stabilized_gcode
            
            if not baseline_path.exists():
                self.research_log.insert(tk.END, f"ERROR: Baseline G-code not found: {baseline_gcode}\n")
                messagebox.showerror("Error", f"Baseline G-code not found: {baseline_gcode}")
                return
            
            if not stabilized_path.exists():
                self.research_log.insert(tk.END, f"ERROR: Stabilized G-code not found: {stabilized_gcode}\n")
                messagebox.showerror("Error", f"Stabilized G-code not found: {stabilized_gcode}")
                return
            
            # Build command - only include CSV if file exists
            cmd = [
                sys.executable, "generate_paper_figures.py",
                "--baseline-gcode", str(baseline_path),
                "--stabilized-gcode", str(stabilized_path),
            ]
            
            # Add CSV argument only if file exists
            stabilized_csv_path = code_dir / stabilized_csv
            if stabilized_csv_path.exists():
                cmd.extend(["--stabilized-csv", str(stabilized_csv_path)])
            
            cmd.extend(["--figures", "1", "2", "3"])
            
            self.research_log.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
            self.root.update()
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=Path(__file__).parent)
            
            self.research_log.insert(tk.END, result.stdout)
            if result.stderr:
                self.research_log.insert(tk.END, "\n\nSTDERR:\n" + result.stderr)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", 
                                  "Paper figures (1-3) displayed!\n\n"
                                  "Use the figure window controls to save them manually.")
            else:
                messagebox.showwarning("Warning", "Some figures may have failed. Check the log.")
        
        except Exception as e:
            self.research_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate figures: {str(e)}")
    
    def run_advanced_plots(self):
        """Run paper figures script (Figures 4-7: Experimental data analysis)."""
        self.research_log.delete(1.0, tk.END)
        self.research_log.insert(tk.END, "Generating paper figures (Figures 4-7)...\n\n")
        self.root.update()
        
        try:
            # Get file paths from GUI
            baseline_gcode = self.verify_input.get() or self.viz_original.get() or "test.gcode"
            stabilized_gcode = self.verify_output.get() or self.viz_stabilized.get() or "results/stabilized.gcode"
            
            # Resolve paths relative to code directory
            code_dir = Path(__file__).parent
            baseline_path = code_dir / baseline_gcode
            if not baseline_path.exists() and baseline_gcode == "test.gcode":
                baseline_path = code_dir / "test_gcode" / "test.gcode"
            stabilized_path = code_dir / stabilized_gcode
            
            # Get data directory from GUI
            data_dir_path = code_dir / self.data_dir.get() if self.data_dir.get() else code_dir / "input"
            if not data_dir_path.exists():
                data_dir_path = code_dir / "data"  # Fallback
            
            print_trials = data_dir_path / "print_trials.csv"
            
            if not print_trials.exists():
                self.research_log.insert(tk.END, f"ERROR: print_trials.csv not found at {print_trials}\n")
                self.research_log.insert(tk.END, "Figures 4-7 require experimental data from print_trials.csv\n")
                messagebox.showerror("Error", f"print_trials.csv not found at {print_trials}\n\n"
                                            "Figures 4-7 require experimental data.\n"
                                            "Please set the data directory or file path in the Input Data Files section.")
                return
            
            cmd = [
                sys.executable, "generate_10_figures.py",
                "--baseline-gcode", str(baseline_path),
                "--stabilized-gcode", str(stabilized_path),
                "--data-dir", str(data_dir_path),
                "--figures", "4", "5", "6", "7"
            ]
            
            self.research_log.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
            self.root.update()
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=Path(__file__).parent)
            
            self.research_log.insert(tk.END, result.stdout)
            if result.stderr:
                self.research_log.insert(tk.END, "\n\nSTDERR:\n" + result.stderr)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", 
                                  "Paper figures (4-7) displayed!\n\n"
                                  "Use the figure window controls to save them manually.")
            else:
                messagebox.showwarning("Warning", "Some figures may have failed. Check the log.")
        
        except Exception as e:
            self.research_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate figures: {str(e)}")
    
    def run_all_research_plots(self):
        """Run all selected research plots."""
        self.research_log.delete(1.0, tk.END)
        self.research_log.insert(tk.END, "Generating all selected research plots...\n\n")
        self.root.update()
        
        results = []
        errors = []
        
        # Get file paths from GUI
        baseline_gcode = self.verify_input.get() or self.viz_original.get() or "test.gcode"
        stabilized_gcode = self.verify_output.get() or self.viz_stabilized.get() or "results/stabilized.gcode"
        stabilized_csv = self.verify_csv.get() or self.csv_file.get() or "results/run_log.csv"
        
        # Get data directory from GUI
        code_dir = Path(__file__).parent
        data_dir_path = code_dir / self.data_dir.get() if self.data_dir.get() else code_dir / "input"
        if not data_dir_path.exists():
            data_dir_path = code_dir / "data"  # Fallback
        print_trials = data_dir_path / "print_trials.csv"
        
        if self.plot_basic.get():
            self.research_log.insert(tk.END, "=== Generating Basic Research Plots (Figures 1-3) ===\n")
            self.root.update()
            try:
                # Resolve paths relative to code directory
                code_dir = Path(__file__).parent
                baseline_path = code_dir / baseline_gcode
                if not baseline_path.exists() and baseline_gcode == "test.gcode":
                    baseline_path = code_dir / "test_gcode" / "test.gcode"
                stabilized_path = code_dir / stabilized_gcode
                
                if not baseline_path.exists():
                    self.research_log.insert(tk.END, f"ERROR: Baseline G-code not found: {baseline_gcode}\n")
                    errors.append("Basic plots")
                elif not stabilized_path.exists():
                    self.research_log.insert(tk.END, f"ERROR: Stabilized G-code not found: {stabilized_gcode}\n")
                    errors.append("Basic plots")
                else:
                    # Get data directory from GUI
                    data_dir_path = code_dir / self.data_dir.get() if self.data_dir.get() else code_dir / "input"
                    if not data_dir_path.exists():
                        data_dir_path = code_dir / "data"  # Fallback
                    
                    # Build command - only include CSV if file exists
                    cmd = [
                        sys.executable, "generate_10_figures.py",
                        "--baseline-gcode", str(baseline_path),
                        "--stabilized-gcode", str(stabilized_path),
                        "--data-dir", str(data_dir_path),
                        "--figures", "1", "2", "3"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          cwd=Path(__file__).parent)
                    self.research_log.insert(tk.END, result.stdout)
                    if result.stderr:
                        self.research_log.insert(tk.END, "\nSTDERR:\n" + result.stderr)
                    if result.returncode == 0:
                        results.append("Basic plots (Figures 1-3)")
                    else:
                        errors.append("Basic plots")
            except Exception as e:
                self.research_log.insert(tk.END, f"\nERROR: {str(e)}\n")
                errors.append("Basic plots")
            self.research_log.insert(tk.END, "\n")
        
        if self.plot_advanced.get():
            self.research_log.insert(tk.END, "=== Generating Advanced Research Plots (Figures 4-7) ===\n")
            self.root.update()
            try:
                if not print_trials.exists():
                    self.research_log.insert(tk.END, f"ERROR: print_trials.csv not found at {print_trials}\n")
                    self.research_log.insert(tk.END, "Figures 4-7 require experimental data.\n")
                    errors.append("Advanced plots")
                else:
                    # Get data directory from GUI
                    data_dir_path = code_dir / self.data_dir.get() if self.data_dir.get() else code_dir / "input"
                    if not data_dir_path.exists():
                        data_dir_path = code_dir / "data"  # Fallback
                    
                    cmd = [
                        sys.executable, "generate_10_figures.py",
                        "--baseline-gcode", str(baseline_path),
                        "--stabilized-gcode", str(stabilized_path),
                        "--data-dir", str(data_dir_path),
                        "--figures", "4", "5", "6", "7"
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          cwd=Path(__file__).parent)
                    self.research_log.insert(tk.END, result.stdout)
                    if result.stderr:
                        self.research_log.insert(tk.END, "\nSTDERR:\n" + result.stderr)
                    if result.returncode == 0:
                        results.append("Advanced plots (Figures 4-7)")
                    else:
                        errors.append("Advanced plots")
            except Exception as e:
                self.research_log.insert(tk.END, f"\nERROR: {str(e)}\n")
                errors.append("Advanced plots")
            self.research_log.insert(tk.END, "\n")
        
        # Summary
        self.research_log.insert(tk.END, "="*60 + "\n")
        self.research_log.insert(tk.END, "SUMMARY:\n")
        if results:
            self.research_log.insert(tk.END, f"✓ Successfully generated: {', '.join(results)}\n")
        if errors:
            self.research_log.insert(tk.END, f"⚠ Errors in: {', '.join(errors)}\n")
        
        if errors:
            messagebox.showwarning("Partial Success", 
                                  f"Generated: {', '.join(results)}\n\n"
                                  f"Errors: {', '.join(errors)}\n\n"
                                  "Check the log for details.")
        elif results:
            messagebox.showinfo("Success", f"All selected plots generated successfully!\n\n"
                                         f"Generated: {', '.join(results)}")
        else:
            messagebox.showwarning("No Selection", "Please select at least one plot type.")
    
    def run_10_figures(self):
        """Run generate_10_figures.py with selected figures."""
        self.figures_10_log.delete(1.0, tk.END)
        self.figures_10_log.insert(tk.END, "Generating selected figures...\n\n")
        self.root.update()
        
        try:
            # Get file paths from GUI
            baseline_gcode = self.verify_input.get() or self.viz_original.get() or "test.gcode"
            stabilized_gcode = self.verify_output.get() or self.viz_stabilized.get() or "results/stabilized.gcode"
            
            # Resolve paths relative to code directory
            code_dir = Path(__file__).parent
            baseline_path = code_dir / baseline_gcode
            if not baseline_path.exists() and baseline_gcode == "test.gcode":
                baseline_path = code_dir / "test_gcode" / "test.gcode"
            stabilized_path = code_dir / stabilized_gcode
            
            if not baseline_path.exists():
                self.figures_10_log.insert(tk.END, f"ERROR: Baseline G-code not found: {baseline_gcode}\n")
                messagebox.showerror("Error", f"Baseline G-code not found: {baseline_gcode}")
                return
            
            if not stabilized_path.exists():
                self.figures_10_log.insert(tk.END, f"ERROR: Stabilized G-code not found: {stabilized_gcode}\n")
                messagebox.showerror("Error", f"Stabilized G-code not found: {stabilized_gcode}")
                return
            
            # Get selected figures
            selected_figures = [num for num, var in self.fig_vars.items() if var.get()]
            if not selected_figures:
                messagebox.showwarning("No Selection", "Please select at least one figure to generate.")
                return
            
            # Get data directory from GUI
            data_dir_path = code_dir / self.data_dir.get() if self.data_dir.get() else code_dir / "input"
            if not data_dir_path.exists():
                data_dir_path = code_dir / "data"  # Fallback
            
            # Build command
            cmd = [
                sys.executable, "generate_10_figures.py",
                "--baseline-gcode", str(baseline_path),
                "--stabilized-gcode", str(stabilized_path),
                "--data-dir", str(data_dir_path),
                "--figures"
            ] + selected_figures
            
            self.figures_10_log.insert(tk.END, f"Running: {' '.join(cmd)}\n\n")
            self.figures_10_log.insert(tk.END, "IMPORTANT: Figures will open in separate windows.\n")
            self.figures_10_log.insert(tk.END, "Close each figure window to proceed to the next one.\n\n")
            self.root.update()
            
            # Run without capturing output so matplotlib can display windows
            import os
            env = os.environ.copy()
            
            process = subprocess.Popen(cmd, cwd=code_dir, env=env)
            returncode = process.wait()
            
            if returncode == 0:
                self.figures_10_log.insert(tk.END, "\n✓ Figure generation completed!\n")
                messagebox.showinfo("Success", 
                                  f"Generated {len(selected_figures)} figure(s)!\n\n"
                                  "Figures should have been displayed in separate windows.\n"
                                  "If figures didn't appear, try running from terminal:\n"
                                  f"{' '.join(cmd)}\n\n"
                                  "Use the figure window controls to save them manually.")
            else:
                self.figures_10_log.insert(tk.END, f"\n⚠ Process exited with code {returncode}\n")
                messagebox.showwarning("Warning", 
                                     f"Process exited with code {returncode}.\n"
                                     "Figures may still have been displayed.\n"
                                     "Check terminal output for details.")
        
        except Exception as e:
            self.figures_10_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate figures: {str(e)}")

def main():
    root = tk.Tk()
    app = StabilizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

