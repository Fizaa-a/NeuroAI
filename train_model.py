"""
Training Script for Stress Detection CNN Model
Loads preprocessed data, trains the model, and evaluates performance
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from preprocessor import EEGPreprocessor
from cnn_model import StressDetectionCNN

def load_files_by_subject(directory_path):
    """
    Load all EDF files and group them by subject
    
    Returns:
    - subjects_data: Dictionary mapping subject_id to {'relaxed': path, 'stressed': path}
    """
    subjects_data = {}
    
    for filename in os.listdir(directory_path):
        if not filename.endswith('.edf'):
            continue
            
        # Extract subject ID (e.g., 'Subject00' from 'Subject00_1.edf')
        subject_id = filename.split('_')[0]
        
        if subject_id not in subjects_data:
            subjects_data[subject_id] = {'relaxed': None, 'stressed': None}
            
        if filename.endswith('_1.edf'):
            subjects_data[subject_id]['relaxed'] = os.path.join(directory_path, filename)
        elif filename.endswith('_2.edf'):
            subjects_data[subject_id]['stressed'] = os.path.join(directory_path, filename)
    
    # Filter out subjects that don't have both files
    complete_subjects = {k: v for k, v in subjects_data.items() if v['relaxed'] and v['stressed']}
    
    print(f"Found {len(complete_subjects)} subjects with both relaxed and stressed state files")
    return complete_subjects

def preprocess_all_files(file_list, preprocessor, label, max_segments_per_file=None):
    """
    Preprocess all files in the list
    
    Parameters:
    - file_list: List of file paths
    - preprocessor: EEGPreprocessor object
    - label: Label for these files (0=relaxed, 1=stressed)
    - max_segments_per_file: (optional) cap number of segments taken from each file (debug)
    
    Returns:
    - all_segments: List of all preprocessed segments
    - all_labels: List of labels
    """
    all_segments = []
    all_labels = []
    
    for idx, file_path in enumerate(file_list):
        try:
            print(f"Processing {os.path.basename(file_path)} ({idx+1}/{len(file_list)})...")
            
            # Preprocess the file
            segments, channel_names = preprocessor.preprocess_pipeline(
                file_path, 
                apply_ica=False,  # ICA is slow; keep False by default
                segment_length=2.0,
                overlap=1.0
            )
            
            # Extract first 20 channels (remove ECG)
            segments = preprocessor.extract_first_20_channels(segments)
            
            # Add segments and labels (respect per-file cap if provided)
            if max_segments_per_file is None:
                for segment in segments:
                    all_segments.append(segment)
                    all_labels.append(label)
            else:
                count = 0
                for segment in segments:
                    if count >= max_segments_per_file:
                        break
                    all_segments.append(segment)
                    all_labels.append(label)
                    count += 1
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return all_segments, all_labels

def prepare_subject_independent_split(subjects_dict, preprocessor, test_size=0.2, val_size=0.1, max_segments_per_file=None):
    """
    Split subjects into train/val/test and then collect segments.
    This prevents data leakage from the same subject being in multiple sets.
    """
    subject_ids = list(subjects_dict.keys())
    
    # Split subject IDs
    train_val_ids, test_ids = train_test_split(subject_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size / (1 - test_size), random_state=42)
    
    print(f"Subject Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    def collect_segments(ids):
        X, y = [], []
        for sid in ids:
            # Process relaxed
            rel_segments, _ = preprocessor.preprocess_pipeline(subjects_dict[sid]['relaxed'], apply_ica=False)
            rel_segments = preprocessor.extract_first_20_channels(rel_segments)
            
            # Process stressed
            str_segments, _ = preprocessor.preprocess_pipeline(subjects_dict[sid]['stressed'], apply_ica=False)
            str_segments = preprocessor.extract_first_20_channels(str_segments)
            
            # Apply cap if needed
            if max_segments_per_file:
                rel_segments = rel_segments[:max_segments_per_file]
                str_segments = str_segments[:max_segments_per_file]
                
            X.extend(rel_segments)
            y.extend([0] * len(rel_segments))
            X.extend(str_segments)
            y.extend([1] * len(str_segments))
        return np.array(X), to_categorical(np.array(y), num_classes=2)

    print("Collecting training data...")
    X_train, y_train = collect_segments(train_ids)
    print("Collecting validation data...")
    X_val, y_val = collect_segments(val_ids)
    print("Collecting test data...")
    X_test, y_test = collect_segments(test_ids)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(data_directory, epochs=100, batch_size=32, debug=False, max_segments_per_file=None):
    """
    Main training function
    
    Parameters:
    - data_directory: Directory containing EDF files
    - epochs: Number of training epochs
    - batch_size: Batch size for training
    - debug: If True, enables fast debug behavior (fewer segments, no ICA)
    - max_segments_per_file: If set, limits segments taken from each file (useful for debug)
    """
    print("=" * 60)
    print("STRESS DETECTION CNN - TRAINING PIPELINE")
    print("=" * 60)
    
    # If debug, enforce conservative defaults
    if debug:
        print("DEBUG MODE: limiting work for a fast smoke-test")
        if max_segments_per_file is None:
            max_segments_per_file = 10
        if epochs > 10:
            epochs = min(epochs, 5)
        if batch_size > 16:
            batch_size = min(batch_size, 8)
    
    # Step 1: Load file paths and group by subject
    print("\n[1/4] Loading file paths...")
    subjects_data = load_files_by_subject(data_directory)
    
    if len(subjects_data) == 0:
        print("Error: No complete subjects found in the directory!")
        return
    
    # Step 2: Initialize preprocessor
    print("\n[2/4] Initializing preprocessor...")
    preprocessor = EEGPreprocessor(sampling_rate=500)
    
    # Step 3: Prepare data with subject-independent split
    print("\n[3/4] Preparing data (subject-independent split)...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_subject_independent_split(
        subjects_data, preprocessor, test_size=0.2, val_size=0.1, 
        max_segments_per_file=max_segments_per_file
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 4: Build and train model
    print("\n[4/4] Building and training CNN model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    print(f"Input shape: {input_shape}")
    
    model = StressDetectionCNN(input_shape=input_shape, num_classes=2)
    model.build_model()
    
    print("\nModel architecture:")
    model.get_model_summary()
    
    print(f"\nStarting training for {epochs} epochs...")

    if debug:
        # Lightweight debug training: perform a few train_on_batch steps on small batches
        print("DEBUG training: running quick train_on_batch steps instead of full .fit()")
        import numpy as _np
        _n_steps = min(5, max(1, epochs))
        _batch = max(1, min(batch_size, 8))
        # ensure small dataset
        X_train_small = X_train[: _batch * 2]
        y_train_small = y_train[: _batch * 2]

        for step in range(_n_steps):
            # sample a small batch
            idx = _np.random.randint(0, X_train_small.shape[0], size=_batch)
            xb = X_train_small[idx]
            yb = y_train_small[idx]
            loss = model.model.train_on_batch(xb, yb)
            print(f" debug step {step+1}/{_n_steps} — train_on_batch loss: {loss}")

        history = None
    else:
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size
        )
    
    # Step 5: Evaluate model
    print("\n[5/5] Evaluating model...")
    if debug:
        # evaluate on tiny test slice to keep debug fast
        metrics, y_pred, y_pred_proba = model.evaluate(X_test[:8], y_test[:8])
    else:
        metrics, y_pred, y_pred_proba = model.evaluate(X_test, y_test)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("=" * 60)
    
    # Plot results
    print("\nGenerating plots...")
    model.plot_training_history()
    model.plot_confusion_matrix(metrics['confusion_matrix'])
    model.plot_roc_curve(y_test, y_pred_proba)
    
    # Save model
    print("\nSaving model...")
    model.save_model('stress_detection_model.h5')
    
    # Save metrics to file
    metrics_df = pd.DataFrame([{
        'Test Loss': metrics['test_loss'],
        'Test Accuracy': metrics['test_accuracy'],
        'Sensitivity': metrics['sensitivity'],
        'Specificity': metrics['specificity'],
        'Precision': metrics['precision'],
        'F1 Score': metrics['f1_score']
    }])
    metrics_df.to_csv('model_metrics.csv', index=False)
    print("Metrics saved to 'model_metrics.csv'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog='train_model.py', description='Train the stress-detection CNN')
    parser.add_argument('data_directory', help='path to EDF directory')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=32)
    parser.add_argument('--debug', action='store_true', help='run lightweight debug training')
    parser.add_argument('--max-segs', type=int, dest='max_segs', default=None,
                        help='max segments per file (debug)')

    args = parser.parse_args()

    if not os.path.isdir(args.data_directory):
        print(f"Error: Directory '{args.data_directory}' does not exist!")
        sys.exit(1)

    train_model(args.data_directory, epochs=args.epochs, batch_size=args.batch_size,
                debug=args.debug, max_segments_per_file=args.max_segs)
