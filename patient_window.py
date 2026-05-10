"""
Patient Window - For stress detection from single EEG file
Uses trained CNN model to predict stress levels
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import sys
import numpy as np
# optional shared UI helpers (no hard dependency)
try:
    from ui_style import apply_theme, Tooltip
except Exception:
    apply_theme = None
    Tooltip = None

# Import our modules
from preprocessor import EEGPreprocessor

class LoadingOverlay:
    """A pop-up loading window for Mac reliability"""
    def __init__(self, parent, cancel_callback):
        self.win = tk.Toplevel(parent)
        self.win.title("AI EEG Scanner")
        self.win.geometry("500x400")
        self.win.configure(bg="#f8f9fa")
        
        # Center in screen
        self.win.transient(parent)
        self.win.grab_set()
        
        # UI Elements
        tk.Label(self.win, text="NEURAL PROCESSING", font=("Helvetica", 12, "bold"), bg="#f8f9fa", fg="#2c3e50").pack(pady=(50, 20))
        
        self.spinner = tk.Label(self.win, text="◐", font=("Helvetica", 60), bg="#f8f9fa", fg="#2b8cff")
        self.spinner.pack()
        
        self.pct_label = tk.Label(self.win, text="AI SCANNING: 0%", font=("Helvetica", 20, "bold"), bg="#f8f9fa", fg="#2c3e50")
        self.pct_label.pack(pady=10)
        
        self.status_label = tk.Label(self.win, text="Initializing...", font=("Helvetica", 11), bg="#f8f9fa", fg="#7f8c8d")
        self.status_label.pack()
        
        self.cancel_btn = tk.Button(
            self.win, text="CANCEL ANALYSIS", 
            command=cancel_callback,
            font=("Helvetica", 11, "bold"),
            highlightbackground="#e74c3c", pady=10
        )
        self.cancel_btn.pack(pady=40)
        
        self.rotation = 0
        self._animate()

    def _animate(self):
        if not self.win.winfo_exists(): return
        chars = ["◐", "◓", "◑", "◒"]
        self.spinner.config(text=chars[self.rotation % 4])
        self.rotation += 1
        self.win.after(200, self._animate)

    def update(self, pct, status):
        if not self.win.winfo_exists(): return
        self.pct_label.config(text=f"AI SCANNING: {int(pct)}%")
        self.status_label.config(text=status)
        self.win.update()

    def destroy(self):
        if self.win.winfo_exists():
            self.win.destroy()

class PatientWindow:
    def __init__(self, master):
        self.master = master
        self.master.title("Stress Detection - Patient Mode")
        self.master.geometry("1000x700")
        # apply shared theme if available
        try:
            apply_theme(self.master)
        except Exception:
            pass
        
        self.model = None
        self.model_path = 'stress_detection_model.h5'
        self.edf_file_path = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Ultra-Stable Mac UI - Vertical Flow"""
        self.master.configure(bg="#f0f2f5")
        
        # Main Scrollable Frame (or just fixed size if preferred)
        container = tk.Frame(self.master, bg="#f0f2f5")
        container.pack(fill=tk.BOTH, expand=True, padx=40, pady=20)

        # Header
        tk.Label(container, text="EEG DIAGNOSTIC DASHBOARD", 
                 font=("Helvetica", 24, "bold"), bg="#f0f2f5", fg="#1a252f").pack(pady=(0, 20))

        # --- STEP 1: UPLOAD ---
        tk.Label(container, text="STEP 1: SELECT DATA", font=("Helvetica", 10, "bold"), bg="#f0f2f5", fg="#7f8c8d").pack(anchor=tk.W)
        self.upload_button = tk.Button(
            container, text="[+] OPEN EEG RECORDING (.EDF)", 
            command=self.upload_file, font=("Helvetica", 13, "bold"),
            height=2, highlightbackground="#3498db"
        )
        self.upload_button.pack(fill=tk.X, pady=(5, 5))

        # FILENAME DISPLAY (Directly below button)
        self.file_label = tk.Label(
            container, text="[ NO FILE SELECTED ]", 
            font=("Helvetica", 12, "bold"), bg="#f0f2f5", fg="#e74c3c",
            pady=10
        )
        self.file_label.pack()

        # --- STEP 2: ANALYZE ---
        tk.Label(container, text="STEP 2: RUN AI SCAN", font=("Helvetica", 10, "bold"), bg="#f0f2f5", fg="#7f8c8d").pack(anchor=tk.W, pady=(20, 0))
        self.analyze_button = tk.Button(
            container, text="🚀 START STRESS DIAGNOSIS", 
            command=self.start_analysis_thread, font=("Helvetica", 16, "bold"), 
            state=tk.DISABLED, height=3, highlightbackground="#2ecc71"
        )
        self.analyze_button.pack(fill=tk.X, pady=5)

        # Result display
        tk.Label(container, text="FINAL DIAGNOSTIC REPORT", font=("Helvetica", 10, "bold"), bg="#f0f2f5", fg="#7f8c8d").pack(anchor=tk.W, pady=(20, 0))
        self.results_text = tk.Text(
            container, height=10, font=("Helvetica", 12),
            bg="white", relief=tk.FLAT, padx=20, pady=20
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.insert(tk.END, "Results will appear here after analysis...")
        self.results_text.config(state=tk.DISABLED)

        # Footer Actions
        self.gradcam_button = tk.Button(
            container, text="👁 VIEW BRAIN ACTIVITY MAP (GRAD-CAM)", 
            command=self.show_gradcam, font=("Helvetica", 12),
            state=tk.DISABLED, highlightbackground="#9b59b6"
        )
        self.gradcam_button.pack(fill=tk.X, pady=10)

        # Status Bar
        self.status_label = tk.Label(self.master, text="System Ready", font=("Helvetica", 9), bg="#2c3e50", fg="white", pady=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def start_analysis_thread(self):
        """Start analysis - UI Pop-up version for Mac"""
        if not self.edf_file_path:
            messagebox.showwarning("File Missing", "Please select an EEG recording first.")
            return
            
        self.overlay = LoadingOverlay(self.master, cancel_callback=self.cancel_analysis)
        self.analyze_button.config(state=tk.DISABLED)
        self._analysis_cancelled = False
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def cancel_analysis(self):
        """Abort analysis and close scanner window"""
        self._analysis_cancelled = True
        if hasattr(self, 'overlay'):
            self.overlay.destroy()
        self.analyze_button.config(state=tk.NORMAL)
        self.update_status("Analysis Aborted", "orange")

    def _run_analysis(self):
        """Background analysis with explicit pop-up updates"""
        try:
            self._update_progress(5, "Calibrating Neural Sensors...")
            import time; time.sleep(0.5)
            
            self._update_progress(20, "Filtering EEG Signal Noise...")
            preprocessor = EEGPreprocessor(sampling_rate=500)
            segments, channel_names = preprocessor.preprocess_pipeline(
                self.edf_file_path,
                apply_ica=False,
                segment_length=2.0,
                overlap=1.0
            )
            
            if self._analysis_cancelled: return
            self._update_progress(45, "Feature Extraction Phase...")
            time.sleep(0.5)

            segments = preprocessor.extract_first_20_channels(segments)
            segments = np.array(segments)
            
            if self._analysis_cancelled: return
            self._update_progress(70, "CNN Pattern Matching...")

            from tensorflow import keras
            if not self.model: self.load_model()
            
            predictions = self.model.predict(segments, verbose=0)
            segment_predictions = np.argmax(predictions, axis=1)
            segment_confidences = np.max(predictions, axis=1)

            stressed_count = np.sum(segment_predictions == 1)
            relaxed_count = np.sum(segment_predictions == 0)
            overall_prediction = 1 if stressed_count > relaxed_count else 0
            overall_confidence = np.mean(segment_confidences) * 100
            stress_percentage = (stressed_count / len(segment_predictions)) * 100

            self.segments = segments
            self.segment_predictions = segment_predictions
            self.segment_confidences = segment_confidences
            self.channel_names = channel_names[:20]

            self._update_progress(100, "Generating Final Report...")
            time.sleep(0.5)

            # Close popup and show results
            self.master.after(0, self.overlay.destroy)
            self.master.after(0, lambda: self.display_results(
                overall_prediction,
                overall_confidence,
                stress_percentage,
                len(segment_predictions),
                stressed_count,
                relaxed_count
            ))
            self._finish_analysis()
        except Exception as e:
            err = f"Error during analysis: {str(e)}"
            self.master.after(0, lambda: self._handle_analysis_error(err))

    def _update_step(self, step_idx, state):
        """Update the visual step indicator"""
        colors = {"active": "#2b8cff", "done": "#27ae60", "pending": "#95a5a6"}
        icons = {"active": "▶", "done": "✔", "pending": "○"}
        
        label = self.step_labels[step_idx]
        text = label.cget("text")[2:] # Strip icon
        label.config(text=f"{icons[state]} {text}", fg=colors[state])
        if state == "active":
            label.config(font=("Helvetica", 10, "bold"))
        else:
            label.config(font=("Helvetica", 10))

    def _update_progress(self, pct, status="Scanning..."):
        try:
            self.progress_var.set(pct)
            if hasattr(self, 'overlay'):
                self.master.after(0, lambda: self.overlay.update(pct, status))
        except Exception:
            pass

    def _finish_analysis(self, cancelled=False):
        if cancelled:
            self.update_status("Analysis cancelled.", "orange")
            self.master.after(0, lambda: self.results_text.config(state=tk.NORMAL))
            self.master.after(0, lambda: self.results_text.insert(tk.END, "\nAnalysis cancelled by user.\n"))
            self.master.after(0, lambda: self.results_text.config(state=tk.DISABLED))
        else:
            self.update_status("Analysis completed successfully!", "green")
            self.master.after(0, lambda: self.gradcam_button.config(state=tk.NORMAL))
        self.master.after(0, lambda: self.analyze_button.config(state=tk.NORMAL))
        self.master.after(0, lambda: self.cancel_button.config(state=tk.DISABLED))

    def _handle_analysis_error(self, error_msg):
        self.update_status(error_msg, "red")
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR:\n{error_msg}\n")
        self.results_text.config(state=tk.DISABLED)
        self.analyze_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
    
    def load_model(self):
        """Load the trained model"""
        from tensorflow import keras
        try:
            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                self.update_status(f"Model loaded: {self.model_path}", "green")
            else:
                self.update_status("Warning: Model file not found! Please train the model first.", "orange")
                messagebox.showwarning(
                    "Model Not Found",
                    f"The trained model file '{self.model_path}' was not found.\n\n"
                    "Please run 'train_model.py' first to train the model."
                )
        except Exception as e:
            self.update_status(f"Error loading model: {str(e)}", "red")
            messagebox.showerror("Model Loading Error", str(e))
    
    def upload_file(self):
        """Handle file upload"""
        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[("EDF Files", "*.edf"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.edf_file_path = file_path
            filename = os.path.basename(file_path)
            
            # Update DISPLAY text below button
            self.file_label.config(
                text=f"📄 LOADED: {filename}", 
                fg="#1e8449"
            )
            
            # Ensure model is ready
            if self.model is None:
                self.load_model()
            
            if self.model is not None:
                self.analyze_button.config(state=tk.NORMAL)
                self.update_status(f"File Ready: {filename}", "green")
            else:
                self.update_status("AI Model missing - Please train first!", "red")
            
            # Force Mac UI sync
            self.master.update()
    
    def analyze_stress(self):
        """Analyze the uploaded EEG file for stress"""
        if not self.edf_file_path or self.model is None:
            return
        
        self.update_status("Analyzing EEG data... Please wait.", "blue")
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Processing EEG data...\n\n")
        self.results_text.config(state=tk.DISABLED)
        self.master.update()
        
        try:
            # Preprocess
            self.update_status("Preprocessing EEG signals...", "blue")
            preprocessor = EEGPreprocessor(sampling_rate=500)
            
            segments, channel_names = preprocessor.preprocess_pipeline(
                self.edf_file_path,
                apply_ica=False,
                segment_length=2.0,
                overlap=1.0
            )
            
            segments = preprocessor.extract_first_20_channels(segments)
            segments = np.array(segments)
            
            # Predict
            self.update_status("Running stress detection model...", "blue")
            predictions = self.model.predict(segments, verbose=0)
            segment_predictions = np.argmax(predictions, axis=1)
            segment_confidences = np.max(predictions, axis=1)
            
            # Calculate results
            stressed_count = np.sum(segment_predictions == 1)
            relaxed_count = np.sum(segment_predictions == 0)
            overall_prediction = 1 if stressed_count > relaxed_count else 0
            overall_confidence = np.mean(segment_confidences) * 100
            stress_percentage = (stressed_count / len(segment_predictions)) * 100
            
            # Store for Grad-CAM
            self.segments = segments
            self.segment_predictions = segment_predictions
            self.segment_confidences = segment_confidences
            self.channel_names = channel_names[:20]
            
            # Display results
            self.display_results(
                overall_prediction,
                overall_confidence,
                stress_percentage,
                len(segment_predictions),
                stressed_count,
                relaxed_count
            )
            
            self.gradcam_button.config(state=tk.NORMAL)
            self.update_status("Analysis completed successfully!", "green")
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            self.update_status(error_msg, "red")
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"ERROR:\n{error_msg}\n")
            self.results_text.config(state=tk.DISABLED)
            messagebox.showerror("Analysis Error", error_msg)
    
    def display_results(self, prediction, confidence, stress_pct, total_segs, 
                       stressed_segs, relaxed_segs):
        """Display analysis results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        # Header
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "     STRESS DETECTION ANALYSIS RESULTS\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")
        
        # Main result
        prediction_text = "STRESSED [!]" if prediction == 1 else "RELAXED [OK]"
        color_tag = "stressed" if prediction == 1 else "relaxed"
        
        self.results_text.insert(tk.END, "OVERALL STATUS: ")
        self.results_text.insert(tk.END, prediction_text + "\n", color_tag)
        self.results_text.insert(tk.END, f"Confidence Level: {confidence:.1f}%\n\n")
        
        # Detailed analysis
        self.results_text.insert(tk.END, "-" * 70 + "\n")
        self.results_text.insert(tk.END, "DETAILED ANALYSIS:\n")
        self.results_text.insert(tk.END, "-" * 70 + "\n\n")
        
        self.results_text.insert(tk.END, f"Total EEG segments analyzed: {total_segs}\n")
        self.results_text.insert(tk.END, f"Stressed segments: {stressed_segs} ({stress_pct:.1f}%)\n")
        self.results_text.insert(tk.END, f"Relaxed segments: {relaxed_segs} ({100-stress_pct:.1f}%)\n\n")
        
        # Interpretation
        self.results_text.insert(tk.END, "-" * 70 + "\n")
        self.results_text.insert(tk.END, "INTERPRETATION:\n")
        self.results_text.insert(tk.END, "-" * 70 + "\n\n")
        
        if prediction == 1:
            if stress_pct > 80:
                interpretation = "High stress levels detected. Consider stress management\ntechniques or consultation with a healthcare professional."
            elif stress_pct > 60:
                interpretation = "Moderate to high stress detected. Monitoring and stress\nreduction activities are recommended."
            else:
                interpretation = "Mild stress detected. Some segments show stress patterns."
        else:
            interpretation = "Low stress levels. EEG patterns indicate a relaxed state."
        
        self.results_text.insert(tk.END, interpretation + "\n\n")
        
        # Footer
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        self.results_text.insert(tk.END, "Click 'View Grad-CAM Visualization' to see which brain regions\n")
        self.results_text.insert(tk.END, "contributed to this classification.\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n")
        
        # Configure tags for coloring
        self.results_text.tag_config("stressed", foreground="red", font=("Arial", 14, "bold"))
        self.results_text.tag_config("relaxed", foreground="green", font=("Arial", 14, "bold"))
        
        self.results_text.config(state=tk.DISABLED)
    
    def show_gradcam(self):
        """Show Grad-CAM visualization in a new window"""
        from gradcam import GradCAM
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'segments'):
            return
        
        # Create new window
        gradcam_window = tk.Toplevel(self.master)
        gradcam_window.title("Grad-CAM Visualization")
        gradcam_window.geometry("1000x800")
        
        # Find a strongly stressed segment
        stressed_idx = np.where(self.segment_predictions == 1)[0]
        
        if len(stressed_idx) > 0:
            best_idx = stressed_idx[np.argmax(self.segment_confidences[stressed_idx])]
        else:
            best_idx = 0
        
        # Create Grad-CAM
        gradcam = GradCAM(self.model)
        
        # Create visualization
        fig = gradcam.visualize_multi_channel(
            self.segments[best_idx],
            channel_indices=[0, 1, 16],  # Fp1, Fp2, Fz
            channel_names=self.channel_names,
            save_path=None
        )
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=gradcam_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Close button
        close_button = tk.Button(
            gradcam_window,
            text="Close",
            command=gradcam_window.destroy,
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            width=15
        )
        close_button.pack(pady=10)
    
    def update_status(self, message, color="black"):
        """Update status bar"""
        colors = {
            "green": "#27ae60",
            "red": "#e74c3c",
            "blue": "#3498db",
            "orange": "#e67e22",
            "black": "#34495e"
        }
        self.status_label.config(text=message, bg=colors.get(color, "#34495e"))
        self.master.update()

def open_patient_window():
    """Open the patient window"""
    root = tk.Tk()
    # macOS rendering fix
    root.update()
    root.lift()
    root.attributes('-topmost', 1)
    root.attributes('-topmost', 0)
    
    root.mainloop()

if __name__ == "__main__":
    open_patient_window()
