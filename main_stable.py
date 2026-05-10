"""
STRESS DETECTION FROM EEG SIGNALS
Consolidated macOS Optimized Application
"""
import tkinter as tk
from main_native import UltimateApp
import sys

def main():
    root = tk.Tk()
    root.title("EEG Stress Analysis System")
    
    # Final Mac-specific event loop handling
    app = UltimateApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
