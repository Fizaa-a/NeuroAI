#!/usr/bin/env python
"""
Interactive Execution Guide for EEG Stress Detection System
Shows menu options and guides user through the process
"""

import os
import sys
import subprocess

def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_menu(options):
    """Print menu options"""
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print()

def main():
    while True:
        print_header("EEG STRESS DETECTION SYSTEM - EXECUTION GUIDE")
        
        print("Welcome! Choose what you want to do:\n")
        
        options = [
            "Start the GUI Application (Recommended)",
            "Run Patient Mode Analysis",
            "Run Academician Mode",
            "Train Model with Real EEG Data",
            "Generate Demo Model",
            "View System Status",
            "Exit"
        ]
        
        print_menu(options)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == "1":
            start_gui()
        elif choice == "2":
            patient_mode_guide()
        elif choice == "3":
            academician_mode_guide()
        elif choice == "4":
            train_model_guide()
        elif choice == "5":
            generate_demo_model()
        elif choice == "6":
            system_status()
        elif choice == "7":
            print_header("Thank you for using EEG Stress Detection System!")
            print("Goodbye!\n")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.\n")

def start_gui():
    """Start the GUI application"""
    print_header("Starting GUI Application")
    print("Command: python user_selection_window_updated.py\n")
    print("A window will open with two options:")
    print("  [P] Patient Mode - Analyze individual EEG files")
    print("  [A] Academician Mode - Extract features and train models\n")
    
    input("Press Enter to start the GUI...")
    
    try:
        subprocess.Popen([sys.executable, "user_selection_window_updated.py"])
        print("\nGUI application started successfully!")
        print("The window should appear on your screen.\n")
    except Exception as e:
        print(f"\nError starting GUI: {e}\n")

def patient_mode_guide():
    """Guide for patient mode"""
    print_header("PATIENT MODE - Step by Step Guide")
    
    print("Step 1: Start the GUI")
    print("  Command: python user_selection_window_updated.py\n")
    
    print("Step 2: Click '[P] Patient Mode'\n")
    
    print("Step 3: Upload an EEG File")
    print("  - Click '[+] Upload EEG File (.edf)'")
    print("  - Browse to eeg_data/ folder")
    print("  - Select any .edf file (e.g., Subject00_1.edf)\n")
    
    print("Step 4: Analyze")
    print("  - Click '[*] Analyze Stress Level'")
    print("  - Wait 30-60 seconds for processing\n")
    
    print("Step 5: View Results")
    print("  - See stress/relaxed classification")
    print("  - View confidence score and segment analysis")
    print("  - Click '[V] View Grad-CAM' to see brain regions\n")
    
    input("Press Enter to go back to menu...")

def academician_mode_guide():
    """Guide for academician mode"""
    print_header("ACADEMICIAN MODE - Step by Step Guide")
    
    print("Step 1: Start the GUI")
    print("  Command: python user_selection_window_updated.py\n")
    
    print("Step 2: Click '[A] Academician/Researcher Mode'\n")
    
    print("Step 3: Upload Files")
    print("  - Click '[+] Upload Files'")
    print("  - Select multiple EDF files")
    print("  - Include both relaxed (_1.edf) and stressed (_2.edf) files")
    print("  - Recommended: 10-20 file pairs for good features\n")
    
    print("Step 4: Build Model (Feature Extraction)")
    print("  - Click '[*] Build Model'")
    print("  - Runs 5 parallel feature extraction processes:")
    print("    * hfd_relaxed.py - Fractal dimension (relaxed)")
    print("    * hfd_stressed.py - Fractal dimension (stressed)")
    print("    * psd_calc.py - Power spectral density")
    print("    * relative_theta.py - Theta power analysis")
    print("    * relative_alpha.py - Alpha power analysis")
    print("  - Wait for: 'Model building finished!'\n")
    
    print("Step 5: View Results")
    print("  - Click '[o] View Results'")
    print("  - See extracted features and analysis plots")
    print("  - Check Features/ folder for CSV files\n")
    
    input("Press Enter to go back to menu...")

def train_model_guide():
    """Guide for training real model"""
    print_header("TRAIN MODEL WITH REAL EEG DATA")
    
    print("This creates an accurate model trained on your EEG data.")
    print("(Takes 1-3 hours on CPU, 30-60 minutes on GPU)\n")
    
    print("Step 1: Prepare Data")
    print("  - Place EDF files in eeg_data/ folder")
    print("  - Naming: SubjectXX_1.edf (relaxed), SubjectXX_2.edf (stressed)")
    print("  - Recommended: 20+ subject pairs\n")
    
    print("Step 2: Run Training")
    print("  Command: python train_model.py eeg_data/\n")
    
    print("Step 3: Monitor Progress")
    print("  - [1/6] Loading files")
    print("  - [2/6] Initializing preprocessor")
    print("  - [3/6] Preprocessing files (takes longest)")
    print("  - [4/6] Preparing data")
    print("  - [5/6] Training CNN model (epochs with early stopping)")
    print("  - [6/6] Evaluating on test set\n")
    
    print("Step 4: Use Trained Model")
    print("  - stress_detection_model.h5 is created")
    print("  - Use immediately with GUI (Patient Mode)")
    print("  - Better accuracy than demo model\n")
    
    input("Press Enter to go back to menu...")

def generate_demo_model():
    """Generate demo model"""
    print_header("Generate Demo Model")
    
    print("This creates a quick demo model for testing.")
    print("(Not for production - trained on random data)\n")
    
    print("Command: python generate_demo_model.py\n")
    
    confirm = input("Generate demo model now? (y/n): ").lower()
    
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, "generate_demo_model.py"], check=True)
            print("\nDemo model generated successfully!")
        except Exception as e:
            print(f"\nError: {e}")
    
    input("\nPress Enter to go back to menu...")

def system_status():
    """Check system status"""
    print_header("System Status Check")
    
    # Check model
    model_exists = os.path.exists("stress_detection_model.h5")
    model_size = os.path.getsize("stress_detection_model.h5") if model_exists else 0
    
    print("Model Status:")
    if model_exists:
        print(f"  [OK] stress_detection_model.h5 exists ({model_size / 1024 / 1024:.1f} MB)")
    else:
        print("  [MISSING] stress_detection_model.h5 not found")
        print("  Run: python generate_demo_model.py\n")
    
    # Check data
    eeg_data_exists = os.path.isdir("eeg_data")
    print("\nEEG Data Status:")
    if eeg_data_exists:
        edf_files = len([f for f in os.listdir("eeg_data") if f.endswith('.edf')])
        print(f"  [OK] eeg_data/ folder found with {edf_files} EDF files")
    else:
        print("  [MISSING] eeg_data/ folder not found")
    
    # Check features
    features_exists = os.path.isdir("Features")
    print("\nFeatures Status:")
    if features_exists:
        csv_files = len([f for f in os.listdir("Features") if f.endswith('.csv')])
        print(f"  [OK] Features/ folder with {csv_files} CSV files")
    else:
        print("  [EMPTY] Features/ folder - not yet created")
    
    # Check key files
    print("\nKey Files Status:")
    files_to_check = [
        "user_selection_window_updated.py",
        "patient_window.py",
        "upload_window.py",
        "cnn_model.py",
        "preprocessor.py",
        "train_model.py"
    ]
    
    for file in files_to_check:
        exists = "[OK]" if os.path.exists(file) else "[MISSING]"
        print(f"  {exists} {file}")
    
    input("\nPress Enter to go back to menu...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!\n")
        sys.exit(0)
