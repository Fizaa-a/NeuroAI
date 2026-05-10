"""
Generate a demo/temporary trained model for testing the GUI
This is just a placeholder that allows the GUI to load while the full model trains
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical

print("=" * 60)
print("GENERATING DEMO MODEL (For testing GUI)")
print("=" * 60)

# Build a simple CNN model with smaller kernels to avoid negative output size
input_shape = (20, 1000)
num_classes = 2

model = models.Sequential([
    # First Convolutional Block
    layers.Conv1D(64, kernel_size=5, activation='relu', padding='same',
                 input_shape=input_shape,
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Second Convolutional Block
    layers.Conv1D(128, kernel_size=5, activation='relu', padding='same',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(256, activation='relu', 
                kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    
    # Output Layer
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Model architecture created successfully!")
print(f"Input shape: {input_shape}")
print(f"Output classes: {num_classes}")

# Generate dummy training data
X_train = np.random.randn(30, 20, 1000).astype(np.float32)
y_train = np.random.randint(0, 2, 30)
y_train = to_categorical(y_train, num_classes=2)

X_val = np.random.randn(10, 20, 1000).astype(np.float32)
y_val = np.random.randint(0, 2, 10)
y_val = to_categorical(y_val, num_classes=2)

# Train for 1 epoch with dummy data
print("\nTraining demo model with dummy data (1 epoch)...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=16, verbose=1)

# Save the model
print("\nSaving demo model to: stress_detection_model.h5")
model.save('stress_detection_model.h5')

print("\n" + "=" * 60)
print("DEMO MODEL CREATED SUCCESSFULLY!")
print("=" * 60)
print("\nModel file: stress_detection_model.h5")
print("\nNOTE: This is a demo model trained on random data.")
print("For accurate predictions, train with real EEG data using:")
print("  python train_model.py eeg_data/")
print("=" * 60)
