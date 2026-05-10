"""
Training Script for Hybrid Stress Detection Model (CNN + Spectral Features)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from preprocessor import EEGPreprocessor
from cnn_model import StressDetectionCNN

def load_files_by_subject(directory_path):
    subjects_data = {}
    for filename in os.listdir(directory_path):
        if not filename.endswith('.edf'): continue
        subject_id = filename.split('_')[0]
        if subject_id not in subjects_data:
            subjects_data[subject_id] = {'relaxed': None, 'stressed': None}
        if filename.endswith('_1.edf'):
            subjects_data[subject_id]['relaxed'] = os.path.join(directory_path, filename)
        elif filename.endswith('_2.edf'):
            subjects_data[subject_id]['stressed'] = os.path.join(directory_path, filename)
    return {k: v for k, v in subjects_data.items() if v['relaxed'] and v['stressed']}

def prepare_hybrid_data_split(subjects_dict, preprocessor, test_size=0.2, val_size=0.1):
    subject_ids = list(subjects_dict.keys())
    train_val_ids, test_ids = train_test_split(subject_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size/(1-test_size), random_state=42)
    
    def collect_data(ids):
        X_raw, X_feat, y = [], [], []
        for sid in ids:
            for condition, label in [('relaxed', 0), ('stressed', 1)]:
                segments, _ = preprocessor.preprocess_pipeline(subjects_dict[sid][condition], apply_ica=False)
                segments = preprocessor.extract_first_20_channels(segments)
                for seg in segments:
                    X_raw.append(seg)
                    X_feat.append(preprocessor.get_bandpower(seg))
                    y.append(label)
        return np.array(X_raw), np.array(X_feat), to_categorical(np.array(y), 2)

    print("Collecting and feature-extracting data...")
    X_train_raw, X_train_feat, y_train = collect_data(train_ids)
    X_val_raw, X_val_feat, y_val = collect_data(val_ids)
    X_test_raw, X_test_feat, y_test = collect_data(test_ids)
    
    return (X_train_raw, X_train_feat, y_train), (X_val_raw, X_val_feat, y_val), (X_test_raw, X_test_feat, y_test)

def train_hybrid():
    data_dir = 'eeg_data'
    preprocessor = EEGPreprocessor(500)
    subjects = load_files_by_subject(data_dir)
    
    (X_tr_r, X_tr_f, y_tr), (X_v_r, X_v_f, y_v), (X_te_r, X_te_f, y_te) = prepare_hybrid_data_split(subjects, preprocessor)
    
    print(f"\nTraining set: {X_tr_r.shape[0]} samples")
    print(f"Feature shape: {X_tr_f.shape[1]}")
    
    model_wrapper = StressDetectionCNN(input_shape=(X_tr_r.shape[1], X_tr_r.shape[2]))
    model_wrapper.build_hybrid_model(feature_shape=(X_tr_f.shape[1],))
    
    print("\nStarting Hybrid Training...")
    history = model_wrapper.model.fit(
        [X_tr_r, X_tr_f], y_tr,
        validation_data=([X_v_r, X_v_f], y_v),
        epochs=40,
        batch_size=32,
        verbose=1
    )
    
    print("\nEvaluating Hybrid Model...")
    loss, acc = model_wrapper.model.evaluate([X_te_r, X_te_f], y_te)
    print(f"\nFinal Hybrid Test Accuracy: {acc*100:.2f}%")
    
    model_wrapper.model.save('hybrid_stress_model.h5')
    print("Hybrid model saved to hybrid_stress_model.h5")

if __name__ == "__main__":
    train_hybrid()
