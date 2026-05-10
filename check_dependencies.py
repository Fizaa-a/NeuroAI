#!/usr/bin/env python3
"""
Dependency Checker for EEG Stress Detection System
Run this to verify all required packages are installed
"""

import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    
    print("=" * 70)
    print("EEG STRESS DETECTION SYSTEM - DEPENDENCY CHECK")
    print("=" * 70)
    print()
    
    required_packages = {
        'Core ML/DL': [
            ('tensorflow', 'TensorFlow'),
            ('keras', 'Keras (via TensorFlow)'),
            ('sklearn', 'Scikit-learn'),
        ],
        'EEG Processing': [
            ('mne', 'MNE-Python'),
            ('pyedflib', 'PyEDFlib (optional, MNE can read EDF)'),
        ],
        'Scientific Computing': [
            ('numpy', 'NumPy'),
            ('scipy', 'SciPy'),
            ('pandas', 'Pandas'),
        ],
        'Visualization': [
            ('matplotlib', 'Matplotlib'),
            ('seaborn', 'Seaborn (optional)'),
        ],
        'Signal Processing': [
            ('pywt', 'PyWavelets'),
            ('hfda', 'Higuchi Fractal Dimension'),
        ],
    }
    
    all_ok = True
    missing_packages = []
    
    for category, packages in required_packages.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for module_name, display_name in packages:
            try:
                __import__(module_name)
                # Try to get version if possible
                try:
                    mod = sys.modules[module_name]
                    version = getattr(mod, '__version__', 'unknown')
                    print(f"  ✓ {display_name:30s} v{version}")
                except:
                    print(f"  ✓ {display_name:30s} (installed)")
            except ImportError:
                print(f"  ✗ {display_name:30s} NOT INSTALLED")
                missing_packages.append(module_name)
                all_ok = False
    
    print()
    print("=" * 70)
    
    if all_ok:
        print("✓ ALL DEPENDENCIES INSTALLED!")
        print()
        print("Your system is ready to run the EEG Stress Detection application.")
    else:
        print("✗ MISSING DEPENDENCIES DETECTED")
        print()
        print("Install missing packages with:")
        print()
        print(f"  pip install {' '.join(missing_packages)}")
    
    print("=" * 70)
    print()
    
    return all_ok


def check_python_version():
    """Check if Python version is compatible"""
    print("\nPython Version Check:")
    print("-" * 70)
    
    major, minor = sys.version_info[:2]
    version_str = f"{major}.{minor}"
    
    if major == 3 and minor >= 7:
        print(f"  ✓ Python {version_str} (Compatible)")
        return True
    else:
        print(f"  ✗ Python {version_str} (Incompatible - need 3.7+)")
        return False


def check_files():
    """Check if critical project files exist"""
    import os
    
    print("\nProject Files Check:")
    print("-" * 70)
    
    required_files = [
        'main.py',
        'cnn_model.py',
        'gradcam.py',
        'preprocessor.py',
        'train_model.py',
        'predict.py',
        'patient_window.py',
        'upload_window.py',
        'user_selection_window_updated.py',
        'view_results.py',
        'ui_style.py',
    ]
    
    optional_files = [
        'stress_detection_model.h5',
        'requirements.txt',
    ]
    
    all_ok = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - MISSING")
            all_ok = False
    
    print()
    print("Optional files:")
    for filename in optional_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.1f} KB"
            print(f"  ✓ {filename} ({size_str})")
        else:
            print(f"  - {filename} - not found (will need to train model)")
    
    return all_ok


def check_model():
    """Check if the trained model can be loaded"""
    import os
    
    print("\nModel Check:")
    print("-" * 70)
    
    if not os.path.exists('stress_detection_model.h5'):
        print("  - Model file not found (you'll need to train one)")
        return False
    
    try:
        from tensorflow import keras
        model = keras.models.load_model('stress_detection_model.h5')
        print(f"  ✓ Model loaded successfully")
        print(f"    Input shape: {model.input_shape}")
        print(f"    Output shape: {model.output_shape}")
        print(f"    Trainable params: {model.count_params():,}")
        return True
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        return False


def main():
    """Run all checks"""
    
    # Python version
    python_ok = check_python_version()
    
    # Dependencies
    deps_ok = check_dependencies()
    
    # Files
    files_ok = check_files()
    
    # Model
    model_ok = check_model()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    checks = [
        ("Python Version", python_ok),
        ("Dependencies", deps_ok),
        ("Project Files", files_ok),
        ("Trained Model", model_ok),
    ]
    
    for check_name, status in checks:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {check_name}")
    
    print()
    
    if all(status for _, status in checks):
        print("🎉 SYSTEM READY! You can run: python main.py")
    elif deps_ok and files_ok and not model_ok:
        print("⚠️  System ready but model not found.")
        print("   Train a model with: python train_model.py /path/to/eeg/data")
    else:
        print("❌ Please fix the issues above before running the system.")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
