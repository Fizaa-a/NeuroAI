"""Guaranteed quick smoke-train (explicit, heavily-instrumented)

- Runs preprocessing on two EDFs (no ICA)
- Builds the CNN, runs a single train_on_batch step on a tiny real-batch
- Saves a small model `debug_real_model.h5` and prints concise diagnostics

This script is designed to surface progress immediately via prints.
"""
import os
import sys
import traceback
print('START: guaranteed_smoke_train')
try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocessor import EEGPreprocessor
    from cnn_model import StressDetectionCNN
    import numpy as np
    print('Imported preprocessing + model modules')

    # Prepare files
    files = [
        os.path.join(os.path.dirname(__file__), '..', 'tiny_subset', 'Subject00_1.edf'),
        os.path.join(os.path.dirname(__file__), '..', 'tiny_subset', 'Subject00_2.edf')
    ]

    for f in files:
        print('checking file:', f, 'exists=', os.path.exists(f))

    pre = EEGPreprocessor(500)
    print('Initialized preprocessor')

    X = []
    y = []
    for f in files:
        segs, ch = pre.preprocess_pipeline(f, apply_ica=False, segment_length=2.0, overlap=1.0)
        segs = pre.extract_first_20_channels(segs)
        print(f'  got {len(segs)} segments from', os.path.basename(f))
        segs = segs[:4]
        label = 0 if f.endswith('_1.edf') else 1
        for s in segs:
            X.append(s)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print('Collected X.shape=', X.shape, 'y.shape=', y.shape)

    print('Now importing tensorflow and building model (this may take ~10s)')
    import tensorflow as tf
    print('TensorFlow version:', tf.__version__)

    model_wrap = StressDetectionCNN(input_shape=(X.shape[1], X.shape[2]), num_classes=2)
    model = model_wrap.build_model()
    print('Model built — running a single train_on_batch...')

    y_cat = tf.keras.utils.to_categorical(y, num_classes=2)
    # small shuffle
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    X = X[idx]
    y_cat = y_cat[idx]

    batch = max(1, min(4, X.shape[0]))
    xb = X[:batch]
    yb = y_cat[:batch]
    loss = model.train_on_batch(xb, yb)
    print('train_on_batch returned loss:', loss)

    out_model = 'debug_real_model.h5'
    model.save(out_model)
    print('Saved model to', out_model)

    # quick eval
    ev = model.evaluate(X[:batch], y_cat[:batch], verbose=0)
    print('quick eval (loss, acc, precision, recall):', ev)

    print('SUCCESS: guaranteed_smoke_train completed')
except Exception as e:
    print('ERROR during guaranteed_smoke_train:')
    traceback.print_exc()
    sys.exit(2)
