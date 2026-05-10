"""Quick end-to-end training check on real EEG segments (fast)

This script:
- Loads 1-2 EDF files from a directory
- Runs the normal preprocessing (with ICA off)
- Builds the CNN and runs a few train_on_batch steps (prints losses)
- Evaluates on a tiny test slice and saves a debug model + metrics

Use for quick verification that the pipeline + model can train on real data.
"""
import os
import numpy as np
from tensorflow import keras
from preprocessor import EEGPreprocessor
from cnn_model import StressDetectionCNN

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'subset_train')
OUT_MODEL = 'debug_real_model.h5'

def main():
    files = [
        os.path.join(DATA_DIR, 'Subject00_1.edf'),
        os.path.join(DATA_DIR, 'Subject00_2.edf')
    ]

    pre = EEGPreprocessor(500)

    X = []
    y = []
    # collect up to 6 segments per file
    max_per_file = 6
    for f in files:
        segs, ch = pre.preprocess_pipeline(f, apply_ica=False, segment_length=2.0, overlap=1.0)
        segs = pre.extract_first_20_channels(segs)
        segs = segs[:max_per_file]
        label = 0 if f.endswith('_1.edf') else 1
        for s in segs:
            X.append(s)
            y.append(label)

    X = np.array(X)
    y = keras.utils.to_categorical(np.array(y), num_classes=2)

    print(f"Collected X.shape={X.shape}, y.shape={y.shape}")

    # Shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    # Split tiny train/test
    split = max(2, int(0.8 * len(X)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model_wrapper = StressDetectionCNN(input_shape=(X.shape[1], X.shape[2]), num_classes=2)
    model = model_wrapper.build_model()
    print("Model built — running quick train_on_batch steps...")

    # Quick training loop
    batch_size = min(4, max(1, X_train.shape[0]))
    steps = 6
    for step in range(steps):
        ix = np.random.randint(0, X_train.shape[0], size=batch_size)
        xb, yb = X_train[ix], y_train[ix]
        loss = model.train_on_batch(xb, yb)
        print(f" step {step+1}/{steps} — loss: {loss}")

    # Quick evaluation
    if X_test.shape[0] > 0:
        ev = model.evaluate(X_test, y_test, verbose=0)
        print("Quick eval (loss, acc, precision, recall):", ev)

    # Save debug model and a tiny metrics file
    model.save(OUT_MODEL)
    print(f"Saved debug model to {OUT_MODEL}")

    with open('debug_metrics.txt', 'w') as fh:
        fh.write(str({'quick_eval': ev if X_test.shape[0] > 0 else None, 'train_steps': steps}))

    print('Quick training finished successfully.')

if __name__ == '__main__':
    main()
