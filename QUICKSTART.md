# 🚀 Quick Start Guide - EEG Stress Detection System

## For Complete Beginners - Get Started in 5 Steps

### Step 1: Install Python (if not already installed)

**Windows/Mac/Linux**:
1. Download Python 3.8+ from [python.org](https://www.python.org/downloads/)
2. During installation, CHECK "Add Python to PATH"
3. Verify installation:
```bash
python --version
```

### Step 2: Install Required Packages

Open terminal/command prompt and run:

```bash
# Install all dependencies at once
pip install numpy pandas scipy mne tensorflow keras matplotlib scikit-learn pywavelets pyedflib hfda
```

**Note**: This may take 5-10 minutes depending on your internet speed.

### Step 3: Get EEG Data

You need EEG files in `.edf` format. Options:

**Option A: Use PhysioNet Dataset** (Recommended for testing)
1. Go to: https://physionet.org/content/eegmat/1.0.0/
2. Download sample files
3. Name them as: `subject01_1.edf` (relaxed), `subject01_2.edf` (stressed)

**Option B: Use Your Own Data**
- Must be in EDF format
- Follow naming: `*_1.edf` = relaxed, `*_2.edf` = stressed

### Step 4: Train the Model

Put all your EEG files in a folder (e.g., `eeg_data/`) and run:

```bash
python train_model.py eeg_data/
```

**What happens**:
- Files are preprocessed
- Model trains for ~100 epochs
- Takes 1-3 hours (CPU) or 20-60 mins (GPU)
- Creates `stress_detection_model.h5`

**You'll see**:
```
[1/6] Loading file paths...
[2/6] Initializing preprocessor...
[3/6] Preprocessing files...
[4/6] Preparing data for training...
[5/6] Building and training CNN model...
[6/6] Evaluating model...
```

### Step 5: Use the System

#### For Patients (Single File Analysis):

```bash
python user_selection_window_updated.py
```

Then:
1. Click "Patient Mode"
2. Upload your EEG file
3. Click "Analyze"
4. See results!

#### For Researchers (Feature Extraction):

```bash
python user_selection_window_updated.py
```

Then:
1. Click "Academician Mode"
2. Upload multiple files
3. Click "Build Model"
4. View extracted features

## 📊 Understanding Your Results

### Patient Mode Output:

```
OVERALL STATUS: STRESSED ⚠️
Confidence Level: 87.5%

Stressed segments: 45 (75.0%)
Relaxed segments: 15 (25.0%)
```

**What this means**:
- **Status**: Overall classification
- **Confidence**: How sure the model is
- **Segments**: Breakdown of 2-second windows

**Confidence Guide**:
- 90-100%: Very confident
- 80-89%: Confident
- 70-79%: Moderately confident
- <70%: Low confidence (interpret cautiously)

### Grad-CAM Visualization:

Red/yellow areas = Brain regions most important for the prediction
Blue areas = Less important regions

**Typical patterns**:
- **Stress**: Red in frontal regions (Fp1, Fp2, Fz)
- **Relaxed**: Red in occipital regions (O1, O2)

## ⚡ Command Line Quick Commands

### Train Model:
```bash
python train_model.py /path/to/data/
```

### Predict Single File:
```bash
python predict.py patient_eeg.edf
```

### Batch Predict:
```bash
python predict.py /path/to/directory/ --batch
```

### Extract Features Only:
```bash
python relative_theta.py file1.edf file2.edf
python relative_alpha.py file1.edf file2.edf
python hfd_stressed.py file_2.edf
python hfd_relaxed.py file_1.edf
```

## 🔥 Common First-Time Issues & Solutions

### Issue 1: "No module named 'tensorflow'"
**Solution**:
```bash
pip install tensorflow
```

### Issue 2: "Model file not found"
**Solution**: You need to train first!
```bash
python train_model.py your_data_folder/
```

### Issue 3: "Permission denied" on Mac/Linux
**Solution**:
```bash
chmod +x *.py
```

### Issue 4: Python command not found
**Solution**: 
- Windows: Use `python` or `py`
- Mac/Linux: Use `python3`

### Issue 5: Import errors with MNE
**Solution**:
```bash
pip install mne --upgrade
# If still fails:
pip uninstall mne
pip install mne
```

## 📁 File Organization Tips

**Good folder structure**:
```
my_project/
├── eeg_data/
│   ├── patient01_1.edf
│   ├── patient01_2.edf
│   ├── patient02_1.edf
│   └── patient02_2.edf
├── [all .py files here]
└── stress_detection_model.h5 (after training)
```

## 💡 Tips for Best Results

### For Training:
1. ✅ Use at least 20 subjects (10 relaxed + 10 stressed files)
2. ✅ Ensure files are properly labeled (_1.edf vs _2.edf)
3. ✅ Use good quality EEG data (minimal artifacts)
4. ✅ Let training complete (don't interrupt)

### For Prediction:
1. ✅ Use the same recording conditions as training data
2. ✅ Ensure file has at least 60 seconds of data
3. ✅ Check Grad-CAM to understand predictions
4. ✅ Consider multiple recordings for better assessment

## 🎯 What Each File Does

| File | Purpose |
|------|---------|
| `train_model.py` | Trains the CNN model |
| `predict.py` | Makes predictions on new files |
| `patient_window.py` | GUI for patients |
| `user_selection_window_updated.py` | Main menu |
| `preprocessor.py` | Cleans EEG signals |
| `cnn_model.py` | Neural network definition |
| `gradcam.py` | Visualization/explainability |

## 🚦 Workflow Summary

```
1. Get EEG data (.edf files)
         ↓
2. Train model (train_model.py)
         ↓
3. Get stress_detection_model.h5
         ↓
4. Use patient_window.py or predict.py
         ↓
5. Get stress classification + visualizations
```

## 📞 Need More Help?

1. **Read full README**: `README_COMPLETE.md`
2. **Check your data**: Make sure EDF files are valid
3. **Verify installation**: Run `pip list` to see installed packages
4. **Test with sample data**: Use PhysioNet dataset first

## ✅ Success Checklist

Before using the system, make sure:
- [ ] Python 3.7+ installed
- [ ] All packages installed (no import errors)
- [ ] Have EEG data in .edf format
- [ ] Files properly named (*_1.edf and *_2.edf for training)
- [ ] Model trained successfully (stress_detection_model.h5 exists)

If all checked, you're ready to go! 🎉

---

**Next Steps**: 
- Try the GUI: `python user_selection_window_updated.py`
- Read the full documentation: `README_COMPLETE.md`
- Experiment with your own data!

Good luck with your stress detection system! 🧠✨
