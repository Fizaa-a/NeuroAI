import tkinter as tk
from tkinter import filedialog, scrolledtext
import subprocess
import threading
import sys

def open_upload_window(user_type):
    # Create the main application window
    app = tk.Tk()
    app.title("File Upload & Feature Extraction")
    
    # Set the initial size of the window (responsive default)
    app.geometry("1100x720")
    try:
        from ui_style import apply_theme, Tooltip
        apply_theme(app)
    except Exception:
        pass

    # Create a Canvas widget
    canvas = tk.Canvas(app)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Add a scrollbar to the canvas
    scrollbar = tk.Scrollbar(app, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Configure the canvas to use the scrollbar
    canvas.configure(yscrollcommand=scrollbar.set)

    # Create a frame to contain the widgets
    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor=tk.NW)

    def on_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_configure)

    def upload_files():
        file_paths = filedialog.askopenfilenames()
        if file_paths:
            label.config(text="Files uploaded:\n" + "\n".join(file_paths))
            build_model_button.config(state="normal")  # Enable the build model button
            # Store the file paths for processing when building the model
            app.uploaded_files = file_paths
        else:
            label.config(text="No files selected.")
            build_model_button.config(state="disabled")  # Disable the build model button if no files are selected

    def build_model():
        # Update status label to indicate calculation is in progress
        status_label.config(text="Model building initiated. Please wait...")
        # Start the streaming subprocess runner (defined below)
        threading.Thread(target=run_subprocesses, daemon=True).start()
    
    def view_results():
        import view_results  # Import the module where open_view_results_window() is defined
        view_results.open_view_results_window()

        
    # Create a button to upload files
    from tkinter import ttk
    upload_button = ttk.Button(frame, text="Upload Files  (Alt+U)", command=upload_files)
    upload_button.pack(pady=10, fill=tk.X)
    try:
        Tooltip(upload_button, 'Select EDF files to upload')
    except Exception:
        pass

    # Create a label to display file paths
    label = tk.Label(frame, text="", anchor=tk.W, justify=tk.LEFT)
    label.pack(fill=tk.X)

    # Create a button to build the model (initially disabled)
    build_model_button = ttk.Button(frame, text="Build Model  (Alt+B)", command=build_model, state="disabled")
    build_model_button.pack(pady=10, fill=tk.X)

    # Create a button to view results (initially disabled)
    view_results_button = ttk.Button(frame, text="View Results  (Alt+V)", command=view_results, state="disabled")
    view_results_button.pack(pady=10, fill=tk.X)

    # Create a label to show status
    status_label = tk.Label(frame, text="", anchor=tk.W)
    status_label.pack(fill=tk.X)

    # Keyboard shortcuts
    app.bind_all('<Alt-u>', lambda e: upload_files())
    app.bind_all('<Alt-b>', lambda e: build_model())
    app.bind_all('<Alt-v>', lambda e: view_results())
    # Log output area (captures feature-extraction subprocess output)
    log_text = scrolledtext.ScrolledText(frame, height=12, width=100, state=tk.DISABLED, bg="#0b0b0b", fg="#dff0d8")
    log_text.pack(pady=10)
    log_text.config(state=tk.DISABLED)

    def append_log(line: str):
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, line + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

    def run_subprocesses():
        try:
            # Disable buttons in UI thread
            app.after(0, lambda: build_model_button.config(state=tk.DISABLED))
            app.after(0, lambda: status_label.config(text="Model building initiated. Please wait..."))

            scripts = ['hfd_relaxed.py', 'hfd_stressed.py', 'psd_calc.py', 'relative_theta.py', 'relative_alpha.py']
            for py_file in scripts:
                if hasattr(app, 'uploaded_files') and app.uploaded_files:
                    cmd = [sys.executable, py_file] + list(app.uploaded_files)
                else:
                    cmd = [sys.executable, py_file]

                app.after(0, lambda p=py_file: append_log(f"Starting: {p} ..."))
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

                # Stream output line-by-line
                for out_line in proc.stdout:
                    stripped = out_line.rstrip('\n')
                    app.after(0, lambda s=stripped: append_log(s))

                proc.wait()
                app.after(0, lambda p=py_file: append_log(f"Finished: {p} (rc={proc.returncode})"))

            app.after(0, lambda: status_label.config(text="Model building finished!"))
            # enable View Results only if required files were created and are non-empty
            def _enable_if_ready():
                import os
                required = [
                    'Features/stressed_hfd_values.csv', 'Features/relaxed_hfd_values.csv',
                    'Features/stressed_theta_power.csv', 'Features/relaxed_theta_power.csv',
                    'Features/stressed_alpha_power.csv', 'Features/relaxed_alpha_power.csv'
                ]
                missing = [p for p in required if not os.path.exists(p) or os.path.getsize(p) == 0]
                if missing:
                    append_log('WARNING: Some feature files are missing or empty: ' + ', '.join(missing))
                    append_log('Tip: inspect the build log or re-run feature extraction on the EDF files')
                    app.after(0, lambda: view_results_button.config(state=tk.DISABLED))
                else:
                    app.after(0, lambda: view_results_button.config(state=tk.NORMAL))

            app.after(0, _enable_if_ready)

        except Exception as exc:
            app.after(0, lambda: append_log(f"ERROR: {str(exc)}"))
            app.after(0, lambda: status_label.config(text="Error during model build"))
        finally:
            app.after(0, lambda: build_model_button.config(state=tk.NORMAL))

    # macOS rendering fix
    app.update()
    app.lift()
    app.attributes('-topmost', 1)
    app.attributes('-topmost', 0)
    
    # If this function is used standalone, start the Tk mainloop so the window remains responsive
    app.mainloop()


# --- CLI / module entrypoint --------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        prog='upload_window',
        description='Open Upload / Feature-extraction UI (standalone)'
    )
    parser.add_argument('--user-type', choices=['Academician', 'Patient'], default='Academician',
                        help='Which UI mode to open (default: Academician)')
    args = parser.parse_args()

    # Launch the requested UI
    open_upload_window(args.user_type)


