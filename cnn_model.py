"""
CNN Model for EEG-based Stress Detection
Implements the architecture described in the project report
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Note: seaborn not installed. Using basic matplotlib for confusion matrix.")

class StressDetectionCNN:
    def __init__(self, input_shape=(20, 1000), num_classes=2):
        """
        Initialize the CNN model for stress detection
        
        Parameters:
        - input_shape: Shape of input EEG segments (channels, samples)
        - num_classes: Number of output classes (2 for stressed/relaxed)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Standard CNN architecture (Compatibility)"""
        return self.build_improved_model()

    def build_improved_model(self):
        """
        Build an improved CNN architecture for EEG data
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Permute((2, 1)),
            
            layers.Conv1D(64, kernel_size=15, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=4),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=9, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling1D(pool_size=4),
            layers.SpatialDropout1D(0.3),
            
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        return model

    def build_eegnet(self, dropout_rate=0.5):
        """
        Implementation of EEGNet (Lawhern et al., 2018).
        A compact convolutional neural network for EEG-based brain-computer interfaces.
        Extremely effective for small datasets because it has very few parameters.
        """
        # input_shape: (channels, samples)
        # We need (samples, channels, 1) for EEGNet style
        
        input_layer = layers.Input(shape=(self.input_shape[0], self.input_shape[1], 1))
        
        # Block 1: Temporal convolution
        # F1 filters of size (1, kernel_size)
        x = layers.Conv2D(8, (1, 64), padding='same', use_bias=False)(input_layer)
        x = layers.BatchNormalization()(x)
        
        # Block 2: Depthwise Convolution (Spatial patterns)
        # D filters per temporal filter. Output is F1 * D filters.
        x = layers.DepthwiseConv2D((self.input_shape[0], 1), use_bias=False, 
                                  depth_multiplier=2,
                                  depthwise_constraint=keras.constraints.MaxNorm(1.))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 4))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Block 3: Separable Convolution
        x = layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('elu')(x)
        x = layers.AveragePooling2D((1, 8))(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Output
        x = layers.Flatten()(x)
        output = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=input_layer, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """
        Train the CNN model
        
        Parameters:
        - X_train: Training data (n_samples, n_channels, n_timepoints)
        - y_train: Training labels (one-hot encoded)
        - X_val: Validation data
        - y_val: Validation labels
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        
        Returns:
        - history: Training history
        """
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        self.history = history
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        - X_test: Test data
        - y_test: Test labels (one-hot encoded)
        
        Returns:
        - metrics: Dictionary containing evaluation metrics
        """
        # Predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # F1 Score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)
        sensitivity = test_recall
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': test_precision,
            'f1_score': f1_score,
            'confusion_matrix': cm
        }
        
        return metrics, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss curves
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_confusion_matrix(self, cm, class_names=['Relaxed', 'Stressed']):
        """
        Plot confusion matrix
        
        Parameters:
        - cm: Confusion matrix
        - class_names: Names of classes
        """
        plt.figure(figsize=(8, 6))
        
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
        else:
            # Fallback without seaborn - manual heatmap
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar()
            
            # Add text annotations manually
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black",
                            fontsize=20)
            
            # Set ticks
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names)
            plt.yticks(tick_marks, class_names)
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """
        Plot ROC curve
        
        Parameters:
        - y_test: True labels (one-hot encoded)
        - y_pred_proba: Predicted probabilities
        """
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate ROC curve for stressed class (class 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def save_model(self, filepath='stress_detection_model.h5'):
        """
        Save the trained model
        
        Parameters:
        - filepath: Path to save the model
        """
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='stress_detection_model.h5'):
        """
        Load a pre-trained model
        
        Parameters:
        - filepath: Path to the model file
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        - X: Input data
        
        Returns:
        - predictions: Class predictions (0=Relaxed, 1=Stressed)
        - probabilities: Prediction probabilities
        """
        probabilities = self.model.predict(X)
        predictions = np.argmax(probabilities, axis=1)
        
        return predictions, probabilities
    
    def get_model_summary(self):
        """
        Print model summary
        """
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return
        
        return self.model.summary()


if __name__ == "__main__":
    # Test the model
    print("CNN Model module loaded successfully!")
    
    # Example usage
    model = StressDetectionCNN(input_shape=(20, 1000), num_classes=2)
    model.build_model()
    print("\nModel Architecture:")
    model.get_model_summary()