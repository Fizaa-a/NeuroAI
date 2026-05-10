# EEG Stress Detection System - Complete Execution Guide

## Overview
This system can:
1. **Analyze single EEG files** (Patient Mode)
2. **Extract features and train models** (Academician Mode)
3. **Visualize stress detection results** with Grad-CAM

---

## STEP 1: Start the Main Application

### Command:
```bash
python user_selection_window_updated.py
```

### What happens:
- A window opens with two buttons: `[P] Patient Mode` and `[A] Academician/Researcher Mode`
- Select your mode based on your need

---

## OPTION A: PATIENT MODE (Single File Analysis)

### Step 1: Click `[P] Patient Mode`
- The selection window closes
- A new "Stress Detection - Patient Mode" window opens

### Step 2: Upload an EEG File
- Click `[+] Upload EEG File (.edf)`
- Select an EDF file from your computer
- Example: `eeg_data/Subject00_1.edf`

### Step 3: Analyze the File
- Click `[*] Analyze Stress Level`
- Wait for processing (takes 30-60 seconds depending on file size)
- The system will:
  1. Load the trained model
  2. Preprocess the EEG signal
  3. Segment the data
  4. Run stress detection
  5. Display results

### Step 4: View Results
You'll see:
```
================================================================
     STRESS DETECTION ANALYSIS RESULTS
================================================================

OVERALL STATUS: STRESSED [!] or RELAXED [OK]
Confidence Level: XX.X%

DETAILED ANALYSIS:
- Total EEG segments analyzed: XX
- Stressed segments: XX (XX.X%)
- Relaxed segments: XX (XX.X%)

INTERPRETATION:
[Explanation of the stress level]
```

### Step 5: View Grad-CAM Visualization (Optional)
- Click `[V] View Grad-CAM Visualization`
- Shows which brain regions contributed to the prediction
- Displays 3 frontal channels (Fp1, Fp2, Fz)
- Red areas = stressed signal, Blue areas = relaxed signal

---

## OPTION B: ACADEMICIAN MODE (Feature Extraction & Training)

### Step 1: Click `[A] Academician/Researcher Mode`
- The selection window closes
- An upload window opens with file browser

### Step 2: Upload EEG Files
- Click `[+] Upload Files`
- Select multiple EDF files from `eeg_data/`
- Select both relaxed (_1.edf) and stressed (_2.edf) files
- Example:
  ```
  Subject00_1.edf (relaxed)
  Subject00_2.edf (stressed)
  Subject01_1.edf (relaxed)
  Subject01_2.edf (stressed)
  ...
  ```

### Step 3: Build Model (Feature Extraction)
- Click `[*] Build Model`
- This runs parallel feature extraction processes:
  - `hfd_relaxed.py` - Higuchi Fractal Dimension for relaxed state
  - `hfd_stressed.py` - Higuchi Fractal Dimension for stressed state
  - `psd_calc.py` - Power Spectral Density
  - `relative_theta.py` - Relative Theta Power
  - `relative_alpha.py` - Relative Alpha Power

- Status shows: "Model building initiated. Please wait..."
- Wait for completion: "Model building finished!"
- Generated files appear in `Features/` folder

### Step 4: View Results
- Click `[o] View Results`
- Opens results visualization window
- Shows extracted features and analysis plots

---

## OPTION C: TRAIN A REAL MODEL (Advanced)

### For Accurate Predictions:
The demo model uses random data. Train a real model with actual EEG data:

### Command:
```bash
python train_model.py eeg_data/
```

### Process:
```
[1/6] Loading file paths...
      Found 36 relaxed state files
      Found 36 stressed state files

[2/6] Initializing preprocessor...

[3/6] Preprocessing files...
      Processing Subject00_1.edf...
      Processing Subject01_1.edf...
      ... (takes 1-3 hours on CPU, 30-60 min on GPU)

[4/6] Preparing data for training...

[5/6] Building and training CNN model...
      Epoch 1/100...
      Epoch 2/100...
      ... (with early stopping)

[6/6] Evaluating model...
      Test Accuracy: XX.X%
      Sensitivity: XX.X%
      Specificity: XX.X%
      Precision: XX.X%
```

### Output Files:
- `stress_detection_model.h5` - Trained model
- `model_metrics.csv` - Performance metrics
- `best_model.h5` - Checkpoint

---

## FILE FORMAT REQUIREMENTS

### EEG Data (.edf files):
- **Format**: EDF (European Data Format)
- **Channels**: Should have 20+ EEG channels
- **Sampling Rate**: 500 Hz
- **Duration**: 2-5 minutes recommended

### Naming Convention:
```
Subject00_1.edf  → Relaxed state (baseline)
Subject00_2.edf  → Stressed state (e.g., mental arithmetic)
Subject01_1.edf  → Relaxed state
Subject01_2.edf  → Stressed state
...
```

---

## TROUBLESHOOTING

### Problem: "Model not found" error
**Solution**: Run the demo model generator:
```bash
python generate_demo_model.py
```
Or train with real data:
```bash
python train_model.py eeg_data/
```

### Problem: "Import error: No module named X"
**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Problem: Analysis takes very long
**Possible causes**:
- Large EEG file (>100 MB)
- CPU processing (slower than GPU)
- System resources limited

**Solution**:
- Use smaller files
- Close other applications
- Consider using GPU if available

### Problem: GUI doesn't respond
**Solution**:
- Close the window
- Run again:
```bash
python user_selection_window_updated.py
```

---

## EXAMPLE WORKFLOW

### Complete Workflow:

**1. First Time Setup:**
```bash
# Install packages
pip install -r requirements.txt

# Generate demo model (already done)
python generate_demo_model.py
```

**2. Test with Demo Model:**
```bash
# Start GUI
python user_selection_window_updated.py

# Select Patient Mode
# Upload any EEG file from eeg_data/
# Click Analyze
# View results
```

**3. Train Real Model (Optional - for better accuracy):**
```bash
# This takes 1-3 hours
python train_model.py eeg_data/

# Now the real model is ready for production use
```

**4. Use in Production:**
```bash
# Run GUI with trained model
python user_selection_window_updated.py

# Patient Mode: Analyze patient EEG files
# Academician Mode: Extract features for research
```

---

## SYSTEM ARCHITECTURE

```
User Interface Layer
    ↓
[user_selection_window_updated.py]
    ↓
    ├─→ Patient Mode [patient_window.py]
    │       ├─→ Load Model
    │       ├─→ Preprocess EEG [preprocessor.py]
    │       ├─→ CNN Prediction [cnn_model.py]
    │       └─→ Grad-CAM Visualization [gradcam.py]
    │
    └─→ Academician Mode [upload_window.py]
            └─→ Feature Extraction
                ├─→ HFD [hfd_*.py]
                ├─→ PSD [psd_calc.py]
                ├─→ Theta Power [relative_theta.py]
                └─→ Alpha Power [relative_alpha.py]
                        ↓
                    Results Visualization [view_results.py]
```

---

## KEY FILES

| File | Purpose |
|------|---------|
| `user_selection_window_updated.py` | Main entry point |
| `patient_window.py` | Patient analysis GUI |
| `upload_window.py` | Feature extraction GUI |
| `cnn_model.py` | CNN architecture |
| `preprocessor.py` | EEG signal preprocessing |
| `gradcam.py` | Explainability visualization |
| `train_model.py` | Model training script |
| `predict.py` | Command-line prediction |
| `stress_detection_model.h5` | Trained model (binary format) |

---

## TIPS FOR BEST RESULTS

1. **Clean EEG Data**: Remove artifacts before analysis
2. **Consistent Sampling**: Use 500 Hz sampling rate
3. **Adequate Duration**: 2+ minutes of EEG data per file
4. **Train with Diverse Data**: 20+ subjects for good generalization
5. **Use GPU**: ~10x faster training than CPU
6. **Monitor Metrics**: Check accuracy, sensitivity, specificity

---

## GETTING HELP

- Read: `README_COMPLETE.md` - Full documentation
- Read: `QUICKSTART.md` - Quick reference
- Check logs for error messages
- Verify file formats and paths
- Ensure all dependencies are installed

---

**Good luck with your EEG stress detection analysis!**
