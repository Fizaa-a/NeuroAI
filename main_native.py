import os
# Force CPU mode and silence TensorFlow warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import sys
import numpy as np
import time

# Force non-interactive Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class EEGEngineSafe:
    def __init__(self):
        self.model = None
        self.gradcam = None
        self.segments = None
        self.predictions = None
        self.confidences = None
        self.channel_names = None
        self.model_path = 'best_model.h5' if os.path.exists('best_model.h5') else 'stress_detection_model.h5'

    def run_scan(self, file_path, log_cb, progress_cb):
        """Step-by-step diagnostic scan with verbose printing"""
        try:
            # Step 1: Loading Libraries (Internal)
            print("--- BACKEND: Initializing Signal Processing ---")
            log_cb("Initializing Neural Sensors...")
            from preprocessor import EEGPreprocessor
            progress_cb(10)
            
            # Step 2: EEG Data Intake
            print(f"--- BACKEND: Opening EDF File: {os.path.basename(file_path)} ---")
            log_cb(f"Reading: {os.path.basename(file_path)}")
            prep = EEGPreprocessor(500)
            
            # This is often where it hangs on Mac
            segments, channels = prep.preprocess_pipeline(file_path, apply_ica=False)
            
            # Step 3: Feature Extraction
            print(f"--- BACKEND: Extracted {len(segments)} segments ---")
            log_cb("Cleaning Brain Signals...")
            progress_cb(40)
            
            eeg_segs = prep.extract_first_20_channels(segments)
            self.segments = np.array(eeg_segs)
            self.channel_names = channels[:20]
            
            # Step 4: AI Model Loading
            log_cb("Waking AI Neural Network...")
            print("--- BACKEND: Loading AI Model (TensorFlow) ---")
            if not self.model:
                from tensorflow import keras
                from gradcam import GradCAM
                self.model = keras.models.load_model(self.model_path)
                self.gradcam = GradCAM(self.model)
            
            # Step 5: AI Inference
            print("--- BACKEND: Running Classification ---")
            log_cb("AI Classification (CNN)...")
            progress_cb(70)
            
            preds = self.model.predict(self.segments, verbose=0)
            self.predictions = np.argmax(preds, axis=1)
            self.confidences = np.max(preds, axis=1)
            
            # Results
            total = len(self.predictions)
            stressed = int(np.sum(self.predictions == 1))
            overall = "STRESSED" if stressed > (total/2) else "RELAXED"
            
            result = {
                "overall": overall,
                "confidence": float(np.mean(self.confidences) * 100),
                "stress_pct": float((stressed/total) * 100),
                "total": total
            }
            print(f"--- BACKEND: Scan Complete. Result: {overall} ---")
            log_cb("Diagnosis Data Compiled.")
            progress_cb(100)
            return result

        except Exception as e:
            print(f"!!! BACKEND CRASH: {str(e)} !!!")
            log_cb(f"CRITICAL ERROR: {str(e)}")
            raise e

class UltimateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG SYSTEM: STANDBY")
        self.root.geometry("800x800")
        self.root.configure(bg="#f0f0f0")
        
        self.engine = EEGEngineSafe()
        self.q = queue.Queue()
        self.edf_path = None
        
        # UI BUILD - Simplified for absolute Mac Rendering
        self.setup_ui()
        self.poll()

    def setup_ui(self):
        # 1. Main Header
        tk.Label(self.root, text="NEURAL SCAN DASHBOARD", font=("Arial", 20, "bold"), bg="#1a1a1a", fg="white", pady=15).pack(fill=tk.X)

        # 2. File Selection (Simple Flat Frame)
        self.f_box = tk.Frame(self.root, bg="#ffffff", bd=1, relief="solid", padx=20, pady=20)
        self.f_box.pack(fill=tk.X, padx=30, pady=20)

        self.btn_open = tk.Button(self.f_box, text="[ SELECT EEG FILE ]", command=self.do_open, width=20, height=2, bg="#3498db")
        self.btn_open.pack()

        self.lbl_file = tk.Label(self.f_box, text="--- NO DATA LOADED ---", font=("Courier", 14, "bold"), fg="red", bg="#ffffff")
        self.lbl_file.pack(pady=10)

        # 3. AI Console (Basic Text)
        tk.Label(self.root, text="AI REAL-TIME FEEDBACK", font=("Arial", 10, "bold"), bg="#f0f0f0").pack(anchor=tk.W, padx=30)
        self.console = tk.Text(self.root, height=15, bg="#000000", fg="#00ff00", font=("Menlo", 12), padx=10, pady=10)
        self.console.pack(fill=tk.BOTH, expand=True, padx=30, pady=5)
        self.console.insert(tk.END, ">>> READY. PLEASE LOAD FILE.\n")

        # 4. Action Button
        self.btn_run = tk.Button(self.root, text="[ START AI STRESS SCAN ]", command=self.do_run, state="disabled", width=30, height=3, bg="#2ecc71")
        self.btn_run.pack(padx=30, pady=10, fill=tk.X)

        # 5. Interpret (Grad-CAM)
        self.btn_map = tk.Button(self.root, text="[ VIEW BRAIN HEATMAP ]", command=self.do_map, state="disabled", width=30, height=2)
        self.btn_map.pack(padx=30, pady=(0, 20), fill=tk.X)

    def log(self, text):
        """Updates UI console and Window Title Bar for maximum visibility"""
        self.console.insert(tk.END, f">>> {text}\n")
        self.console.see(tk.END)
        self.root.title(f"EEG SYSTEM: {text[:20]}...")
        # Force EVERYTHING to update
        self.lbl_file.update()
        self.console.update()
        self.root.update_idletasks()
        self.root.update()
        print(f"UI_LOG: {text}")

    def do_open(self):
        f = filedialog.askopenfilename(filetypes=[("EDF", "*.edf")])
        if f:
            self.edf_path = f
            name = os.path.basename(f)
            # Immediate feedback inside the colored label
            self.lbl_file.config(text=f"📂 LOADED: {name}", fg="green")
            self.btn_run.config(state="normal")
            self.log(f"Brain signals loaded: {name}")

    def do_run(self):
        if not self.edf_path: return
        self.btn_run.config(state="disabled")
        self.log("Starting full system scan...")
        
        def work():
            try:
                def p_cb(v): self.q.put(('p', v))
                def l_cb(t): self.q.put(('l', t))
                res = self.engine.run_scan(self.edf_path, l_cb, p_cb)
                self.q.put(('f', res))
            except Exception as e:
                self.q.put(('e', str(e)))
        
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.q.empty():
            msg = self.q.get_nowait()
            if msg[0] == 'l':
                self.log(msg[1])
            elif msg[0] == 'p':
                # No progress bar for safety, just update text
                pass
            elif msg[0] == 'e':
                self.log(f"CRITICAL ERROR: {msg[1]}")
                messagebox.showerror("Scan Crash", msg[1])
                self.btn_run.config(state="normal")
            elif msg[0] == 'f':
                self.finish(msg[1])
        self.root.after(50, self.poll)

    def finish(self, res):
        self.btn_run.config(state="normal")
        self.btn_map.config(state="normal")
        self.log(f"SCAN FINISHED! RESULT: {res['overall']}")
        
        self.console.insert(tk.END, "\n" + "="*40 + "\n")
        self.console.insert(tk.END, f" DIAGNOSIS: {res['overall']}\n")
        self.console.insert(tk.END, f" CONFIDENCE: {res['confidence']:.1f}%\n")
        self.console.insert(tk.END, f" STRESS RATIO: {res['stress_pct']:.2f}%\n")
        self.console.insert(tk.END, "="*40 + "\n")
        self.console.see(tk.END)

    def do_map(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        try:
            self.log("Loading Interpretation Map...")
            idx = 0
            if self.engine.predictions is not None:
                st = np.where(self.engine.predictions == 1)[0]
                if len(st) > 0: idx = st[0]
            
            fig = self.engine.gradcam.visualize_multi_channel(
                self.engine.segments[idx],
                channel_indices=[0, 1, 16],
                channel_names=self.engine.channel_names
            )
            
            w = tk.Toplevel(self.root)
            w.title("Neural Interpretation Map")
            canvas = FigureCanvasTkAgg(fig, master=w)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            plt.close(fig)
        except Exception as e:
            self.log(f"Visualization Error: {str(e)}")
            messagebox.showerror("Interpretation Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateApp(root)
    root.mainloop()
