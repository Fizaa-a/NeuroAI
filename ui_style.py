"""Shared UI theme and helper utilities for the GUI windows.

Provides:
- apply_theme(root): configures ttk styles and returns a palette dict
- Tooltip: small tooltip helper class (non-blocking)

Keep this file minimal and dependency-free (Tkinter only).
"""
from tkinter import ttk

PALETTE = {
    "bg": "#f4f6f8",
    "card": "#ffffff",
    "primary": "#2b8cff",
    "accent": "#2ecc71",
    "muted": "#7f8c8d",
    "text": "#243044",
    "danger": "#e74c3c",
    # used by several windows for footer / accent backgrounds
    "accent_bg": "#34495e",
    "accent_fg": "#ffffff"
}


def apply_theme(root):
    """Apply a consistent ttk theme and styles to the given Tk root."""
    style = ttk.Style(root)

    # Use Helvetica for Mac compatibility, fallback to Arial/Segoe
    FONT_FAMILY = "Helvetica"

    # Prefer a clean base theme
    try:
        style.theme_use('clam')
    except Exception:
        pass

    # General button styles
    style.configure('Primary.TButton',
                    background=PALETTE['primary'],
                    foreground='white',
                    font=(FONT_FAMILY, 11, 'bold'),
                    padding=8)
    style.map('Primary.TButton', background=[('active', '#1f6fe6')])

    style.configure('Accent.TButton',
                    background=PALETTE['accent'],
                    foreground='white',
                    font=(FONT_FAMILY, 11, 'bold'),
                    padding=8)
    style.map('Accent.TButton', background=[('active', '#28b463')])

    style.configure('Card.TFrame', background=PALETTE['card'], relief='flat')
    style.configure('Muted.TLabel', foreground=PALETTE['muted'], background=PALETTE['bg'], font=(FONT_FAMILY, 10))
    style.configure('Title.TLabel', foreground=PALETTE['text'], background=PALETTE['bg'], font=(FONT_FAMILY, 20, 'bold'))
    style.configure('Small.TLabel', foreground=PALETTE['muted'], background=PALETTE['bg'], font=(FONT_FAMILY, 9))

    # Progressbar style
    style.configure('verb.Progressbar', troughcolor=PALETTE['card'], bordercolor=PALETTE['bg'], background=PALETTE['primary'])

    # Apply root background and force focus
    try:
        root.configure(bg=PALETTE['bg'])
        # Use update_idletasks instead of update for safer rendering on Mac
        root.update_idletasks()
    except Exception:
        pass

    return PALETTE


class Tooltip:
    """Very small tooltip helper for Tkinter widgets.

    Usage:
        t = Tooltip(widget, text="...")
    """
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self.tipwindow = None
        widget.bind('<Enter>', self._schedule)
        widget.bind('<Leave>', self._unschedule)
        widget.bind('<ButtonPress>', self._unschedule)

    def _schedule(self, _ev=None):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show)

    def _unschedule(self, _ev=None):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None
        self._hide()

    def _show(self):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tipwindow = tw = ttk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        label = ttk.Label(tw, text=self.text, style='Small.TLabel', background='#333', foreground='white', padding=(6,4))
        label.pack()

    def _hide(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()
