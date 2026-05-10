"""
Training Script for EEGNet Model
"""

import os
import sys
import numpy as np
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
        if filename.endswith('_1.edf'): subjects_data[subject_id]['relaxed'] = os.path.join(directory_path, filename)
        elif filename.endswith('_2.edf'): subjects_data[subject_id]['stressed'] = os.path.join(directory_path, filename)
    return {k: v for k, v in subjects_data.items() if v['relaxed'] and v['stressed']}

def prepare_eegnet_data(subjects_dict, preprocessor, test_size=0.2, val_size=0.1):
    subject_ids = list(subjects_dict.keys())
    train_val_ids, test_ids = train_test_split(subject_ids, test_size=test_size, random_state=42)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size/(1-test_size), random_state=42)
    
    def collect_data(ids):
        X, y = [], []
        for sid in ids:
            for condition, label in [('relaxed', 0), ('stressed', 1)]:
                # Use faster preprocessing for LOSO-style experiments
                segments, _ = preprocessor.preprocess_pipeline(subjects_dict[sid][condition], apply_ica=False)
                segments = preprocessor.extract_first_20_channels(segments)
                X.extend(segments)
                y.extend([label] * len(segments))
        X = np.array(X)
        # EEGNet expects (batch, channels, samples, 1)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        return X, to_categorical(np.array(y), 2)

    print("Preparing EEGNet data...")
    X_train, y_train = collect_data(train_ids)
    X_val, y_val = collect_data(val_ids)
    X_test, y_test = collect_data(test_ids)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def main():
    preprocessor = EEGPreprocessor(500)
    subjects = load_files_by_subject('eeg_data')
    (X_tr, y_tr), (X_v, y_v), (X_te, y_te) = prepare_eegnet_data(subjects, preprocessor)
    
    model_wrapper = StressDetectionCNN(input_shape=(X_tr.shape[1], X_tr.shape[2]))
    model_wrapper.build_eegnet()
    
    print(f"\nEEGNet Trainable Params: {model_wrapper.model.count_params()}")
    model_wrapper.model.summary()

    print("\nTraining EEGNet...")
    model_wrapper.model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=50,
        batch_size=16, # Small batches for EEGNet
        verbose=1
    )
    
    loss, acc = model_wrapper.model.evaluate(X_te, y_te)
    print(f"\nFinal EEGNet Test Accuracy: {acc*100:.2f}%")
    model_wrapper.model.save('eegnet_stress_model.h5')

if __name__ == "__main__":
    main()
