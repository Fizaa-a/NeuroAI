import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import threading
import time
import queue

# Simple Colors for absolute visibility
BG_COLOR = "#f0f0f0"
TEXT_COLOR = "#000000"
GREEN_COLOR = "#008000"
BLUE_COLOR = "#0000FF"

class PatientApp:
    def __init__(self, master, engine):
        self.master = master
        self.engine = engine
        self.master.title("Patient Mode - Stable")
        self.master.geometry("800x600")
        self.master.configure(bg=BG_COLOR)
        
        self.edf_file = None
        self.msg_queue = queue.Queue()
        
        self.setup_ui()
        self.check_queue() # Start queue poller

    def setup_ui(self):
        # 1. Selection Area
        f1 = tk.Frame(self.master, bg=BG_COLOR, pady=20)
        f1.pack(fill=tk.X, padx=20)
        
        tk.Label(f1, text="1. ACCESS EEG DATA", font=("Arial", 12, "bold"), bg=BG_COLOR).pack(anchor=tk.W)
        
        row = tk.Frame(f1, bg="white", bd=1, relief="solid")
        row.pack(fill=tk.X, pady=5)
        
        self.btn_load = tk.Button(row, text="SELECT FILE (.EDF)", command=self.load_file)
        self.btn_load.pack(side=tk.TOP, padx=10, pady=10)
        
        self.lbl_file = tk.Label(row, text="NO FILE LOADED", fg="red", bg="white", font=("Arial", 11))
        self.lbl_file.pack(side=tk.TOP, padx=10, pady=(0, 10))

        # 2. Analysis Area
        f2 = tk.Frame(self.master, bg=BG_COLOR, pady=20)
        f2.pack(fill=tk.BOTH, expand=True, padx=20)
        
        tk.Label(f2, text="2. AI DIAGNOSIS", font=("Arial", 12, "bold"), bg=BG_COLOR).pack(anchor=tk.W)
        
        self.btn_start = tk.Button(f2, text="START AI STRESS SCAN", command=self.start_scan, state="disabled", height=2)
        self.btn_start.pack(fill=tk.X, pady=5)
        
        # Inline Progress Bar instead of Popup
        self.lbl_progress = tk.Label(f2, text="", font=("Arial", 16, "bold"), fg=BLUE_COLOR, bg=BG_COLOR)
        self.lbl_progress.pack(pady=5)
        
        self.txt_out = tk.Text(f2, height=10, bg="white", font=("Courier", 12))
        self.txt_out.pack(fill=tk.BOTH, expand=True, pady=10)
        self.txt_out.insert("1.0", "System ready. Please load a file to begin.")
        self.txt_out.config(state="disabled")

        # 3. Visualization Toggle
        self.btn_grad = tk.Button(self.master, text="VIEW BRAIN MAP (GRAD-CAM)", state="disabled", command=self.show_gradcam)
        self.btn_grad.pack(fill=tk.X, padx=20, pady=10)

    def log(self, text):
        self.txt_out.config(state="normal")
        self.txt_out.delete("1.0", tk.END)
        self.txt_out.insert("1.0", f"{text}\n")
        self.txt_out.config(state="disabled")
        print(f"APP: {text}")

    def load_file(self):
        file = filedialog.askopenfilename(filetypes=[("EDF", "*.edf")])
        if file:
            self.edf_file = file
            fname = os.path.basename(file)
            self.lbl_file.config(text=f"LOADED: {fname}", fg=GREEN_COLOR)
            self.master.update()
            
            # Load model safely here instead of background thread to avoid crash
            self.log("Initializing AI Model...")
            self.master.config(cursor="wait")
            self.master.update()
            
            if not self.engine.is_model_loaded():
                self.lbl_progress.config(text="Loading Model (Takes a moment)...", fg="orange")
                self.master.update()
                success = self.engine.load_model()
                if success:
                    self.lbl_progress.config(text="Model Loaded Successfully.", fg="green")
                else:
                    self.lbl_progress.config(text="Failed to load Model!", fg="red")
            
            self.master.config(cursor="")
            self.btn_start.config(state="normal")
            self.log(f"File Ready: {fname}")
            self.master.update()

    def start_scan(self):
        self.btn_start.config(state="disabled")
        self.lbl_progress.config(text="0% - Starting Scan...", fg=BLUE_COLOR)
        self.master.update()
        
        def run():
            try:
                def cb(p, s):
                    self.msg_queue.put(('p', p, s))
                res = self.engine.process_and_analyze(self.edf_file, on_progress_callback=cb)
                self.msg_queue.put(('f', res))
            except Exception as e:
                self.msg_queue.put(('e', str(e)))

        threading.Thread(target=run, daemon=True).start()

    def check_queue(self):
        """Poll the queue for background updates"""
        try:
            while not self.msg_queue.empty():
                msg = self.msg_queue.get_nowait()
                if msg[0] == 'p':
                    # Progress update
                    pct, st = msg[1], msg[2]
                    self.lbl_progress.config(text=f"{int(pct)}% - {st}", fg=BLUE_COLOR)
                    self.log(f"[IN PROGRESS] {st} ({int(pct)}%)")
                elif msg[0] == 'f':
                    # Finished successfully
                    self.lbl_progress.config(text="100% - Diagnosis Complete", fg=GREEN_COLOR)
                    self.btn_start.config(state="normal")
                    self.show_result(msg[1])
                    print(f"DEBUG: Analysis Finished: {msg[1]['overall']}")
                elif msg[0] == 'e':
                    # Error occurred
                    self.lbl_progress.config(text="ERROR DURING ANALYSIS", fg="red")
                    self.btn_start.config(state="normal")
                    messagebox.showerror("AI Error", msg[1])
                    self.log(f"CRITICAL ERROR: {msg[1]}")
        except Exception as e:
            print(f"DEBUG: Queue Error: {e}")
            
        self.master.after(50, self.check_queue)

    def show_result(self, res):
        prediction_text = "STRESSED [!]" if res['overall'] == "STRESSED" else "RELAXED [OK]"
        report = (
            "====================================================\n"
            "         STRESS DETECTION ANALYSIS RESULTS\n"
            "====================================================\n\n"
            f"OVERALL STATUS: {prediction_text}\n"
            f"Confidence Level: {res['confidence']:.1f}%\n\n"
            "----------------------------------------------------\n"
            "DETAILED ANALYSIS:\n"
            "----------------------------------------------------\n"
            f"Total EEG segments analyzed: {res['total_segments']}\n"
            f"Stress Level (Stressed Segments): {res['stress_percentage']:.1f}%\n\n"
            "====================================================\n"
            "Click 'VIEW BRAIN MAP' to see visual explanation.\n"
        )
        self.txt_out.config(state="normal")
        self.txt_out.delete("1.0", tk.END)
        self.txt_out.insert(tk.END, report)
        self.txt_out.config(state="disabled")
        self.btn_grad.config(state="normal")

    def show_gradcam(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = self.engine.get_gradcam_figure()
        if fig:
            win = tk.Toplevel(self.master)
            win.title("Brain Heatmap")
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            
            win.update()
            win.lift()
            win.attributes('-topmost', 1)
            win.attributes('-topmost', 0)
