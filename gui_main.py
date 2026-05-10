import tkinter as tk
from tkinter import messagebox
from app_styles import *
from gui_patient import PatientApp
from gui_researcher import ResearcherApp 
from eeg_engine import EEGEngine

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Stress Detection - Role Selection")
        self.root.geometry("800x600")
        self.root.configure(bg=BG_MAIN)
        center_window(self.root, 800, 600)
        
        # Shared AI Engine
        self.engine = EEGEngine()
        
        self.setup_ui()
        apply_mac_fixes(self.root)

    def setup_ui(self):
        # Header Container
        header = tk.Frame(self.root, bg=HEADER_BG, height=120)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text="EEG STRESS DETECTION SYSTEM",
            font=FONT_H1, bg=HEADER_BG, fg="white"
        ).pack(expand=True)

        # Body Container
        body = tk.Frame(self.root, bg=BG_MAIN)
        body.pack(fill=tk.BOTH, expand=True, padx=50, pady=50)

        # Welcome Label
        tk.Label(
            body, text="PLEASE CHOOSE YOUR ACCESS LEVEL",
            font=FONT_H3, bg=BG_MAIN, fg=TEXT_LIGHT
        ).pack(pady=(0, 40))

        # Selection Cards (Frames with labels)
        card_container = tk.Frame(body, bg=BG_MAIN)
        card_container.pack(fill=tk.X, expand=True)

        # 1. Patient Card
        p_frame = tk.Frame(card_container, bg=BG_CARD, padx=30, pady=30, bd=1, relief="solid")
        p_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(p_frame, text="👤 PATIENT MODE", font=FONT_H2, bg=BG_CARD, fg=TEXT_DARK).pack(pady=(10, 5))
        tk.Label(p_frame, text="Simple EEG analysis\nQuick AI result reporting", font=FONT_SMALL, bg=BG_CARD, fg=TEXT_LIGHT).pack(pady=10)
        tk.Button(p_frame, text="ACCESS PATIENT", command=self.open_patient, bg=PRIMARY, font=FONT_BOLD).pack(fill=tk.X, pady=20)

        # 2. Researcher Card
        r_frame = tk.Frame(card_container, bg=BG_CARD, padx=30, pady=30, bd=1, relief="solid")
        r_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(r_frame, text="🔬 ACADEMIC MODE", font=FONT_H2, bg=BG_CARD, fg=TEXT_DARK).pack(pady=(10, 5))
        tk.Label(r_frame, text="Spectral data maps\nDetailed feature logs", font=FONT_SMALL, bg=BG_CARD, fg=TEXT_LIGHT).pack(pady=10)
        tk.Button(r_frame, text="ACCESS LAB", command=self.open_researcher, bg=ACCENT, font=FONT_BOLD).pack(fill=tk.X, pady=20)

    def open_patient(self):
        new_win = tk.Toplevel(self.root)
        PatientApp(new_win, self.engine)

    def open_researcher(self):
        new_win = tk.Toplevel(self.root)
        ResearcherApp(new_win, self.engine)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
