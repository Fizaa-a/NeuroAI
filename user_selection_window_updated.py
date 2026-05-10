"""
Updated User Selection Window
Allows selection between Patient and Academician modes
"""

import tkinter as tk
from tkinter import messagebox
import os

def open_selection_window():
    """Open the main user selection window"""
    
    app = tk.Tk()
    app.title("Stress Detection System - User Selection")
    app.geometry("900x560")
    # apply shared theme
    try:
        from ui_style import apply_theme, Tooltip
        PALETTE = apply_theme(app)
    except Exception:
        PALETTE = None

    # Main title
    FONT = "Helvetica"
    title_label = tk.Label(
        app,
        text="EEG Stress Detection System",
        font=(FONT, 22, "bold"),
        bg=PALETTE['bg'] if PALETTE else "#ecf0f1",
        fg=PALETTE['text'] if PALETTE else "#2c3e50"
    )
    title_label.pack(pady=(30, 6))
    
    # Subtitle
    subtitle_label = tk.Label(
        app,
        text="Advanced AI‑Powered Stress Analysis from EEG signals",
        font=(FONT, 11),
        bg=PALETTE['bg'] if PALETTE else "#ecf0f1",
        fg=PALETTE['muted'] if PALETTE else "#7f8c8d"
    )
    subtitle_label.pack(pady=(0, 12))
    
    # Separator (subtle)
    separator = tk.Frame(app, height=2, bg="#e6e9ee")
    separator.pack(fill=tk.X, padx=70, pady=(6, 18))
    
    # Instructions
    instruction_label = tk.Label(
        app,
        text="Choose a mode to get started",
        font=(FONT, 12),
        bg=PALETTE['bg'] if PALETTE else "#ecf0f1",
        fg=PALETTE['text'] if PALETTE else "#34495e"
    )
    instruction_label.pack(pady=(6, 18))
    
    # Button frame
    button_frame = tk.Frame(app, bg=PALETTE['bg'] if PALETTE else "#ecf0f1")
    button_frame.pack(pady=6)
    
    def open_patient_mode():
        """Open patient window"""
        # Check if model exists
        if not os.path.exists('stress_detection_model.h5'):
            response = messagebox.askyesno(
                "Model Not Found",
                "The trained model was not found in the current directory.\n\n"
                "Patient mode requires a trained model to make predictions.\n\n"
                "Would you like to:\n"
                "- Click 'Yes' to continue anyway (you can specify model location later)\n"
                "- Click 'No' to cancel and train a model first"
            )
            if not response:
                return
        
        app.destroy()
        from patient_window import open_patient_window
        open_patient_window()
    
    def open_academician_mode():
        """Open academician/researcher window"""
        app.destroy()
        from upload_window import open_upload_window
        open_upload_window("Academician")
    
    # Primary action buttons (ttk for consistent theming & accessibility)
    from tkinter import ttk

    patient_button = ttk.Button(
        button_frame,
        text="Patient Mode  (Alt+P)",
        command=open_patient_mode,
        style="Primary.TButton"
    )
    patient_button.grid(row=0, column=0, padx=20, pady=10, ipadx=18, ipady=12)
    try:
        Tooltip(patient_button, "Run single-subject analysis and visual explanations")
    except Exception:
        pass

    patient_desc = tk.Label(
        button_frame,
        text="For individuals seeking\nstress-level analysis from\ntheir EEG recordings",
        font=(FONT, 9),
        bg=PALETTE['bg'] if PALETTE else "#ecf0f1",
        fg=PALETTE['muted'] if PALETTE else "#7f8c8d",
        justify=tk.CENTER
    )
    patient_desc.grid(row=1, column=0, padx=20, pady=5)

    academician_button = ttk.Button(
        button_frame,
        text="Academician / Researcher  (Alt+R)",
        command=open_academician_mode,
        style="Accent.TButton"
    )
    academician_button.grid(row=0, column=1, padx=20, pady=10, ipadx=18, ipady=12)
    try:
        Tooltip(academician_button, "Batch processing, feature extraction and training")
    except Exception:
        pass

    academician_desc = tk.Label(
        button_frame,
        text="For researchers analyzing\nfeatures and conducting\nstudies on EEG data",
        font=(FONT, 9),
        bg=PALETTE['bg'] if PALETTE else "#ecf0f1",
        fg=PALETTE['muted'] if PALETTE else "#7f8c8d",
        justify=tk.CENTER
    )
    academician_desc.grid(row=1, column=1, padx=20, pady=5)

    # Quick-access bindings for keyboard users
    app.bind_all('<Alt-p>', lambda e: open_patient_mode())
    app.bind_all('<Alt-r>', lambda e: open_academician_mode())
    app.bind_all('<Alt-x>', lambda e: app.destroy())
    patient_button.focus_set()
    
    # Information box (card style)
    from tkinter import ttk
    info_frame = ttk.Frame(app, style='Card.TFrame')
    info_frame.pack(pady=26, padx=50, fill=tk.X)

    info_title = tk.Label(
        info_frame,
        text="[i] System Requirements",
        font=(FONT, 11, "bold"),
        bg=PALETTE['card'] if PALETTE else "#ffffff",
        fg=PALETTE['text'] if PALETTE else "#2c3e50"
    )
    info_title.pack(pady=(8, 2))

    info_text = tk.Label(
        info_frame,
        text="• Patient Mode: Requires trained model (stress_detection_model.h5)\n"
             "• Academician Mode: For feature extraction and model training\n"
             "• Supported format: EDF (European Data Format) files\n"
             "• Minimum Python 3.7 with TensorFlow, MNE, and required packages",
        font=(FONT, 9),
        bg=PALETTE['card'] if PALETTE else "#ffffff",
        fg=PALETTE['muted'] if PALETTE else "#555",
        justify=tk.LEFT
    )
    info_text.pack(pady=(0, 12), padx=10)

    # Footer (subtle)
    footer_label = tk.Label(
        app,
        text="Powered by Deep Learning & Grad-CAM | Version 1.0",
        font=(FONT, 9),
        bg=PALETTE.get('accent_bg', '#34495e') if PALETTE else "#34495e",
        fg=PALETTE.get('accent_fg', '#ffffff') if PALETTE else "#ffffff"
    )
    footer_label.pack(side=tk.BOTTOM, fill=tk.X, pady=0)
    
    # Center the window
    app.update_idletasks()
    width = 900
    height = 560
    x = (app.winfo_screenwidth() // 2) - (width // 2)
    y = (app.winfo_screenheight() // 2) - (height // 2)
    app.geometry(f'{width}x{height}+{x}+{y}')
    
    # macOS rendering fix
    app.update()
    app.lift()
    app.attributes('-topmost', 1)
    app.attributes('-topmost', 0)
    
    app.mainloop()

if __name__ == "__main__":
    open_selection_window()
