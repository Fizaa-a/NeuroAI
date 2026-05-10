import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# apply shared theme if available
try:
    from ui_style import apply_theme, Tooltip
except Exception:
    apply_theme = None
    Tooltip = None

def generate_chart(electrodes, label, stressed, relaxed):
    # Electrode mapping dictionary
    electrode_mapping = {
        1: "EEG Fp1",
        2: "EEG Fp2",
        3: "EEG F3",
        4: "EEG F4",
        5: "EEG F7",
        6: "EEG F8",
        7: "EEG T3",
        8: "EEG T4",
        9: "EEG C3",
        10: "EEG C4",
        11: "EEG T5",
        12: "EEG T6",
        13: "EEG P3",
        14: "EEG P4",
        15: "EEG O1",
        16: "EEG O2",
        17: "EEG Fz",
        18: "EEG Cz",
        19: "EEG Pz",
        20: "EEG A2-A1"
    }

    # Create a new window for the chart
    chart_window = tk.Toplevel()
    chart_window.title(f"Charts for {label}")
    if apply_theme:
        apply_theme(chart_window)

    # Set the size of the window (responsive)
    chart_window.geometry("900x640")

    # Create a figure for the chart (figure size scales with window)
    fig = Figure(figsize=(9, 6), dpi=100)

    nrows = len(electrodes)
    for i, electrode in enumerate(electrodes):
        # Create a subplot for each electrode
        ax = fig.add_subplot(nrows, 1, i+1)

        # Extract values for the current label and electrode from both stressed and relaxed dataframes
        stressed_values = stressed.iloc[:, electrode]
        relaxed_values = relaxed.iloc[:, electrode]

        # Plot stressed state data in red
        ax.plot(stressed_values, color='#e74c3c', label='Stressed')

        # Plot relaxed state data in blue
        ax.plot(relaxed_values, color='#3498db', label='Relaxed')

        # Set plot title and labels
        ax.set_title(f'{label} — {electrode_mapping.get(electrode, electrode)}')
        ax.set_xlabel('Subjects')
        ax.set_ylabel(label)

        # Add legend
        ax.legend(frameon=False)

    # Tidy layout
    fig.tight_layout(pad=2.0)

    # Add the figure to the chart window
    canvas = FigureCanvasTkAgg(fig, master=chart_window)
    canvas.draw()
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Add toolbar + save button
    controls = ttk.Frame(chart_window)
    controls.pack(fill=tk.X, padx=8, pady=(6, 10))

    def _save_fig():
        path = tk.filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png'), ('PDF', '*.pdf')])
        if path:
            fig.savefig(path, dpi=200)

    save_btn = ttk.Button(controls, text='Save chart (Ctrl+S)', command=_save_fig)
    save_btn.pack(side=tk.RIGHT)
    if Tooltip:
        Tooltip(save_btn, 'Save this figure to disk')

    # keyboard shortcut for saving
    chart_window.bind_all('<Control-s>', lambda e: _save_fig())
    chart_window.focus_set()

# Function to open the view results window
def open_view_results_window():
    # Create the view results window
    view_results_window = tk.Toplevel()
    view_results_window.title("View Results")
    if apply_theme:
        apply_theme(view_results_window)

    # Set the initial size of the window
    view_results_window.geometry("880x640")

    # Header
    header = ttk.Frame(view_results_window)
    header.pack(fill=tk.X, pady=8)
    title_label = tk.Label(header, text="Available Results", font=("Segoe UI", 14, "bold"))
    title_label.pack(side=tk.LEFT, padx=12)

    # Read the data from CSV files
    stressed_hfd_df = pd.read_csv("Features/stressed_hfd_values.csv")
    relaxed_hfd_df = pd.read_csv("Features/relaxed_hfd_values.csv")
    stressed_rtp_df = pd.read_csv("Features/stressed_theta_power.csv")
    relaxed_rtp_df = pd.read_csv("Features/relaxed_theta_power.csv")
    stressed_rap_df = pd.read_csv("Features/stressed_alpha_power.csv")
    relaxed_rap_df = pd.read_csv("Features/relaxed_alpha_power.csv")

    # Define electrodes for different features
    hfd_electrodes = [1, 3, 6]
    rel_theta_electrodes = [17, 1, 2]
    rel_alpha_electrodes = [15, 13, 19]

    # Feature buttons (ttk)
    content = ttk.Frame(view_results_window, padding=(12, 6))
    content.pack(fill=tk.BOTH, expand=1)

    btn1 = ttk.Button(content, text="Higuchi Fractal Dimension  (Alt+1)", command=lambda: generate_chart(hfd_electrodes, "Higuchi Fractal Dimension", stressed_hfd_df, relaxed_hfd_df))
    btn1.pack(fill=tk.X, pady=8)

    btn2 = ttk.Button(content, text="Relative Theta Power  (Alt+2)", command=lambda: generate_chart(rel_theta_electrodes, "Relative Theta Power", stressed_rtp_df, relaxed_rtp_df))
    btn2.pack(fill=tk.X, pady=8)

    btn3 = ttk.Button(content, text="Relative Alpha Power  (Alt+3)", command=lambda: generate_chart(rel_alpha_electrodes, "Relative Alpha Power", stressed_rap_df, relaxed_rap_df))
    btn3.pack(fill=tk.X, pady=8)

    # Keyboard shortcuts
    view_results_window.bind_all('<Alt-1>', lambda e: generate_chart(hfd_electrodes, "Higuchi Fractal Dimension", stressed_hfd_df, relaxed_hfd_df))
    view_results_window.bind_all('<Alt-2>', lambda e: generate_chart(rel_theta_electrodes, "Relative Theta Power", stressed_rtp_df, relaxed_rtp_df))
    view_results_window.bind_all('<Alt-3>', lambda e: generate_chart(rel_alpha_electrodes, "Relative Alpha Power", stressed_rap_df, relaxed_rap_df))

    view_results_window.focus_set()
