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

class StabilizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Paste Extrusion Stabilizer - GUI")
        self.root.geometry("900x700")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar(value="results/stabilized.gcode")
        self.csv_file = tk.StringVar(value="results/run_log.csv")
        self.log_file = tk.StringVar(value="results/changes.log")
        
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
        
        # Tab 5: Help
        self.create_help_tab()
    
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
                        text="Generate all figures for your research paper:\n"
                             "• Print trials analysis\n• Electrical trace results\n• Pressure simulations\n• Stabilization analysis",
                        justify=tk.CENTER)
        desc.pack(pady=5)
        
        # Info
        info_frame = ttk.LabelFrame(frame, text="Information", padding=10)
        info_frame.pack(fill=tk.X, padx=20, pady=10)
        
        info_text = ("This will generate all publication-ready figures:\n\n"
                    "• Extrusion onset and flow duration boxplots\n"
                    "• Success rates (first-layer, completion)\n"
                    "• Clog frequency analysis\n"
                    "• Pressure simulation comparisons\n"
                    "• Electrical trace results\n"
                    "• Comprehensive summary figure\n\n"
                    "All figures are saved to the 'figures/' directory.")
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # Run research plots
        plots_btn = ttk.Button(frame, text="Generate Research Plots", 
                              command=self.run_research_plots)
        plots_btn.pack(pady=20)
        
        # Research plots log
        plots_log_frame = ttk.LabelFrame(frame, text="Output", padding=10)
        plots_log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.research_log = scrolledtext.ScrolledText(plots_log_frame, height=10, wrap=tk.WORD)
        self.research_log.pack(fill=tk.BOTH, expand=True)
    
    def create_help_tab(self):
        """Tab 5: Help and Documentation"""
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
            results = []
            
            # 3D comparison
            if self.viz_3d.get():
                self.visualization_log.insert(tk.END, "Creating 3D comparison...\n")
                cmd = [
                    sys.executable, "visualize_3d_comparison.py",
                    "--original", self.viz_original.get(),
                    "--stabilized", self.viz_stabilized.get(),
                    "--output", self.viz_output.get()
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
                results.append(("3D Comparison", result))
            
            # Detailed comparison
            if self.viz_comparison.get():
                self.visualization_log.insert(tk.END, "Creating detailed comparison...\n")
                cmd = [
                    sys.executable, "compare_gcode.py",
                    self.viz_original.get(), self.viz_stabilized.get()
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
                results.append(("Detailed Comparison", result))
            
            # Output results
            for name, result in results:
                self.visualization_log.insert(tk.END, f"\n{name}:\n")
                self.visualization_log.insert(tk.END, result.stdout)
                if result.stderr:
                    self.visualization_log.insert(tk.END, "\nSTDERR:\n" + result.stderr)
            
            messagebox.showinfo("Success", "Visualizations generated successfully!")
        
        except Exception as e:
            self.visualization_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate visualizations: {str(e)}")
    
    def run_research_plots(self):
        """Run research plots script."""
        self.research_log.delete(1.0, tk.END)
        self.research_log.insert(tk.END, "Generating research plots...\n\n")
        self.root.update()
        
        try:
            cmd = [sys.executable, "generate_research_plots.py"]
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=Path(__file__).parent.parent)
            
            self.research_log.insert(tk.END, result.stdout)
            if result.stderr:
                self.research_log.insert(tk.END, "\n\nSTDERR:\n" + result.stderr)
            
            if result.returncode == 0:
                messagebox.showinfo("Success", "Research plots generated successfully!")
            else:
                messagebox.showwarning("Warning", "Some plots may have failed. Check the log.")
        
        except Exception as e:
            self.research_log.insert(tk.END, f"\n\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate research plots: {str(e)}")

def main():
    root = tk.Tk()
    app = StabilizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

