# EEG-Based Stress Detection System

A complete deep learning system for detecting stress from EEG (electroencephalogram) signals using Convolutional Neural Networks (CNN) and Gradient-weighted Class Activation Mapping (Grad-CAM) for interpretability.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

This system analyzes EEG brain signals to classify mental states as either **stressed** or **relaxed**. It uses:
- Advanced signal preprocessing (filtering, artifact removal)
- CNN-based deep learning for classification
- Grad-CAM for explainable AI visualization
- Feature extraction (HFD, relative theta/alpha power, MSC, PSD)

**Key Achievement**: 84.6% accuracy in stress detection

## ✨ Features

### Core Capabilities
- ✅ **Automated Stress Detection**: Classify EEG signals as stressed or relaxed
- ✅ **Explainable AI**: Visualize which brain regions contribute to predictions
- ✅ **Feature Extraction**: Extract multiple EEG features for analysis
- ✅ **User-Friendly GUI**: Separate interfaces for patients and researchers
- ✅ **Batch Processing**: Analyze multiple files at once
- ✅ **Comprehensive Reports**: Detailed analysis with confidence scores

### Two User Modes
1. **Patient Mode**: Upload a single EEG file for stress analysis
2. **Academician Mode**: Extract features, train models, and conduct research

## 💻 System Requirements

### Hardware
- CPU: Multi-core processor (Intel i5 or better recommended)
- RAM: Minimum 8GB (16GB recommended for training)
- Storage: 2GB free space
- GPU: Optional but recommended for faster training (NVIDIA with CUDA support)

### Software
- Python 3.7 or higher
- Operating System: Windows 10/11, macOS, or Linux

## 📦 Installation

### Step 1: Clone or Download the Project

```bash
git clone <repository-url>
cd eeg-stress-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; import mne; print('Installation successful!')"
```

## 📁 Project Structure

```
eeg-stress-detection/
├── preprocessor.py          # EEG signal preprocessing module
├── cnn_model.py            # CNN model architecture
├── gradcam.py              # Grad-CAM implementation
├── train_model.py          # Model training script
├── predict.py              # Prediction script
├── patient_window.py       # GUI for patient mode
├── user_selection_window_updated.py  # Main GUI entry
├── upload_window.py        # Feature extraction GUI
├── view_results.py         # Results visualization
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── Features/              # Generated feature CSVs (created during use)
├── *.h5                   # Trained model files (created after training)
└── *.png                  # Generated visualizations
```

## 🚀 Usage

### For First-Time Users: Training the Model

Before using the system, you need to train the model on EEG data.

#### Step 1: Prepare Your Data

Organize your EEG files in a directory:
```
data/
├── subject01_1.edf  (relaxed state)
├── subject01_2.edf  (stressed state)
├── subject02_1.edf
├── subject02_2.edf
...
```

**File naming convention**:
- `*_1.edf`: Relaxed/baseline state
- `*_2.edf`: Stressed state (e.g., during mental arithmetic)

#### Step 2: Train the Model

```bash
python train_model.py /path/to/data/directory
```

**Training Parameters** (can be modified in script):
- Epochs: 100 (with early stopping)
- Batch size: 32
- Validation split: 10%
- Test split: 20%

**Expected Training Time**:
- CPU: 2-4 hours (36 subjects, ~5000 segments)
- GPU: 30-60 minutes

**Outputs**:
- `stress_detection_model.h5`: Trained model
- `best_model.h5`: Best model during training
- `model_metrics.csv`: Performance metrics
- `training_history.png`: Loss/accuracy curves
- `confusion_matrix.png`: Confusion matrix
- `roc_curve.png`: ROC curve

### For Patients: Stress Detection

#### Using the GUI

```bash
python user_selection_window_updated.py
```

1. Click **"Patient Mode"**
2. Click **"Upload EEG File"** and select your `.edf` file
3. Click **"Analyze Stress Level"**
4. View results and confidence scores
5. Click **"View Grad-CAM Visualization"** to see brain activity patterns

#### Using Command Line

```bash
python predict.py path/to/your/file.edf
```

**Output**:
- Overall prediction (Stressed/Relaxed)
- Confidence percentage
- Segment-by-segment analysis
- Grad-CAM visualizations (saved as PNG files)

### For Researchers: Feature Extraction & Analysis

#### Using the GUI

```bash
python user_selection_window_updated.py
```

1. Click **"Academician Mode"**
2. Upload multiple EDF files
3. Click **"Build Model"** to extract features
4. Click **"View Results"** to see comparative analyses

**Generated Features**:
- Higuchi Fractal Dimension (HFD)
- Relative Theta Power
- Relative Alpha Power
- Power Spectral Density (Delta, Theta, Alpha, Beta)
- Magnitude Square Coherence (MSC)

#### Using Command Line

Extract individual features:

```bash
# Higuchi Fractal Dimension
python hfd_stressed.py file1_2.edf file2_2.edf ...
python hfd_relaxed.py file1_1.edf file2_1.edf ...

# Relative Theta Power
python relative_theta.py file1.edf file2.edf ...

# Relative Alpha Power
python relative_alpha.py file1.edf file2.edf ...

# Power Spectral Density
python psd_calc.py file1.edf file2.edf ...
```

### Batch Processing

Process multiple files at once:

```bash
python predict.py /path/to/directory --batch
```

Outputs: `batch_predictions.csv` with results for all files

## 🧠 Model Architecture

### CNN Structure

```
Input: (20 channels, 1000 time points)
│
├─ Conv1D (256 filters, kernel=9) → ReLU → MaxPool → Dropout(0.3)
├─ Conv1D (512 filters, kernel=9) → ReLU → MaxPool → Dropout(0.3)
├─ Flatten
├─ Dense (512 units) → ReLU → Dropout(0.5)
└─ Dense (2 units) → Softmax

Total Parameters: ~5M
Optimizer: Adam
Loss: Categorical Crossentropy
```

### Preprocessing Pipeline

1. **Filtering**:
   - High-pass: 0.5 Hz (removes drift)
   - Low-pass: 60 Hz (removes noise)
   - Notch: 50 Hz (removes power line interference)

2. **Artifact Removal** (Optional):
   - Independent Component Analysis (ICA)
   - Removes eye blinks, muscle artifacts

3. **Segmentation**:
   - Window length: 2 seconds
   - Overlap: 1 second

4. **Standardization**:
   - Z-score normalization per channel

## 📊 Results

### Model Performance (from report)

| Metric | Value |
|--------|-------|
| Test Accuracy | 84.6% |
| Sensitivity (Recall) | 83.7% |
| Specificity | 85.5% |
| Precision | 84.3% |
| F1 Score | 84.0% |

### Key Findings

**Most Discriminative Electrodes**:
1. **Theta Power**: Fz, Fp1, Fp2 (frontal regions)
2. **Alpha Power**: O1, P3, Pz (occipital/parietal regions)
3. **HFD**: Fp1, F8, Fp2 (frontal regions)

**Stress Indicators**:
- ⬆️ Theta power in frontal regions during stress
- ⬇️ Alpha power in occipital regions during stress
- Changes in fractal dimension reflecting complexity

## 🔧 Troubleshooting

### Common Issues

#### 1. "Model file not found"
**Solution**: Train the model first using `train_model.py`

#### 2. "Memory Error during training"
**Solutions**:
- Reduce batch size in `train_model.py` (try 16 or 8)
- Process fewer files at once
- Close other applications

#### 3. "Import Error: No module named X"
**Solution**: 
```bash
pip install -r requirements.txt --upgrade
```

#### 4. "Unable to read EDF file"
**Solutions**:
- Ensure file is valid EDF format
- Check file permissions
- Try opening file in EDFbrowser to verify

#### 5. Grad-CAM visualization not showing
**Solutions**:
- Ensure matplotlib backend is configured: `export MPLBACKEND=TkAgg`
- Try running from terminal instead of IDE
- Update matplotlib: `pip install matplotlib --upgrade`

### Performance Optimization

#### For Faster Training:
1. Use GPU (install `tensorflow-gpu`)
2. Reduce number of training epochs
3. Disable ICA artifact removal (set `apply_ica=False`)

#### For Better Accuracy:
1. Enable ICA artifact removal
2. Increase training epochs
3. Use more training data
4. Adjust learning rate

## 📚 Dataset Information

This system uses the **PhysioNet EEG dataset**:
- 36 subjects
- Mental arithmetic task for stress induction
- 21 EEG channels (10-20 system)
- Sampling rate: 500 Hz

**Citation**: 
Zyma, I., et al. (2019). "Electroencephalograms during Mental Arithmetic Task Performance" (version 1.0.0). PhysioNet.

## 🔬 Technical Details

### Feature Extraction Algorithms

**Higuchi Fractal Dimension (HFD)**:
- Measures complexity of EEG signals
- k_max = 250
- Higher values = more complex patterns

**Relative Power**:
- Theta (4-8 Hz): `theta / (theta + alpha + beta + gamma)`
- Alpha (8-13 Hz): Similar calculation
- Uses Welch's method for PSD estimation

**Magnitude Square Coherence (MSC)**:
- Measures synchronization between channels
- Computed using cross-spectral density

### Grad-CAM Implementation

- Uses last convolutional layer activations
- Computes gradients w.r.t. target class
- Generates heatmap showing important time points
- Overlays on original EEG signal

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional feature extraction methods
- Real-time stress monitoring
- Mobile app integration
- Multi-modal stress detection (EEG + other biosignals)

## 📄 License

This project is for educational and research purposes. Please cite appropriately if used in academic work.

## 📧 Support

For issues and questions:
1. Check the Troubleshooting section
2. Open an issue on GitHub
3. Contact the development team

## 🙏 Acknowledgments

- PhysioNet for providing the EEG dataset
- MNE-Python team for EEG processing tools
- TensorFlow team for deep learning framework
- Original research team from the project report

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Status**: Complete & Functional
