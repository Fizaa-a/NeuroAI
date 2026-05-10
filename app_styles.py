import tkinter as tk
from tkinter import font as tkfont
import sys

# --- COLORS ---
BG_MAIN = "#f0f2f5"
BG_CARD = "#ffffff"
PRIMARY = "#3498db"      # Blue
SECONDARY = "#2ecc71"    # Green
ACCENT = "#9b59b6"       # Purple
DANGER = "#e74c3c"       # Red
TEXT_DARK = "#2c3e50"
TEXT_LIGHT = "#7f8c8d"
HEADER_BG = "#1a252f"

# --- FONTS ---
# Helvetica is the most stable native font for macOS
FONT_H1 = ("Helvetica", 28, "bold")
FONT_H2 = ("Helvetica", 20, "bold")
FONT_H3 = ("Helvetica", 16, "bold")
FONT_BODY = ("Helvetica", 12)
FONT_BOLD = ("Helvetica", 12, "bold")
FONT_SMALL = ("Helvetica", 10)
FONT_TINY = ("Helvetica", 9)

def get_font(size, bold=False):
    weight = "bold" if bold else "normal"
    return ("Helvetica", size, weight)

def apply_mac_fixes(root):
    """Specific fixes for macOS Tkinter stability"""
    if sys.platform == "darwin":
        # Allow window to receive focus correctly
        root.update_idletasks()
        # Prevent some flickering by not using aggressive lift()
        pass

def center_window(window, width, height):
    """Center a window on the screen"""
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
