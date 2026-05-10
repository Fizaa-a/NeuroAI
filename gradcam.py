"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For visualizing which parts of EEG signals contribute to stress classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM
        
        Parameters:
        - model: Trained Keras model
        - layer_name: Name of the layer to visualize (default: last conv layer)
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            for layer in reversed(model.layers):
                # Check for Conv1D layers (either by class name or shape)
                if 'conv1d' in layer.__class__.__name__.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer: {layer_name} for Grad-CAM")
        
        # We build explicit Sub-Models to ensure TF2 Eager execution tracks gradients correctly
        conv_layer = model.get_layer(layer_name)
        
        # Model 1: Maps input to CNN layer output
        self.model_1 = keras.models.Model(inputs=model.inputs, outputs=conv_layer.output)
        
        # Model 2: Maps CNN layer output to final predictions
        layer_idx = model.layers.index(conv_layer)
        classifier_input = keras.Input(shape=conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[layer_idx+1:]:
            x = layer(x)
        self.model_2 = keras.models.Model(inputs=classifier_input, outputs=x)
    
    def compute_gradcam(self, input_data, class_idx=None):
        """
        Compute Grad-CAM heatmap
        
        Parameters:
        - input_data: Input EEG segment (shape: [1, channels, samples])
        - class_idx: Class index to compute Grad-CAM for (if None, uses predicted class)
        
        Returns:
        - heatmap: Grad-CAM heatmap
        - prediction: Model prediction
        """
        # Ensure input has batch dimension
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=0)
        
        # Use a tensor for gradient tracking
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)

        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Map input to conv activations
            conv_outputs = self.model_1(input_data)
            # Explicitly mark conv_outputs to be tracked for gradients
            tape.watch(conv_outputs)
            
            # Map conv activations to predictions
            predictions = self.model_2(conv_outputs)
            
            # If class_idx not specified, use the predicted class
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            
            # Get the score for the target class
            class_score = predictions[:, class_idx]
        
        # Compute gradients of class score with respect to conv layer output
        grads = tape.gradient(class_score, conv_outputs)
        
        # Global average pooling on gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Multiply each channel by its importance weight
        for i in range(len(pooled_grads)):
            conv_outputs[:, i] *= pooled_grads[i]
        
        # Create heatmap by averaging across channels
        heatmap = np.mean(conv_outputs, axis=-1)
        
        # Normalize heatmap between 0 and 1
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap, predictions.numpy()
    
    def visualize_gradcam(self, input_data, channel_idx=0, class_idx=None, 
                         channel_names=None, save_path=None):
        """
        Visualize Grad-CAM heatmap overlaid on EEG signal
        
        Parameters:
        - input_data: Input EEG segment
        - channel_idx: Index of channel to visualize
        - class_idx: Target class for Grad-CAM
        - channel_names: List of channel names
        - save_path: Path to save the visualization
        """
        # Compute Grad-CAM
        heatmap, predictions = self.compute_gradcam(input_data, class_idx)
        
        # Ensure input has correct shape
        if len(input_data.shape) == 3:
            input_data = input_data[0]
        
        # Get the signal for the specified channel
        signal_data = input_data[channel_idx, :]
        
        # Resize heatmap to match signal length
        heatmap_resized = self._resize_heatmap(heatmap, len(signal_data))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        
        # Plot 1: Original EEG signal with heatmap overlay
        time_points = np.arange(len(signal_data))
        
        # Create colored background based on heatmap
        for i in range(len(signal_data) - 1):
            alpha = heatmap_resized[i]
            color = cm.jet(heatmap_resized[i])
            ax1.axvspan(i, i+1, alpha=alpha*0.5, color=color, zorder=0)
        
        # Plot EEG signal
        ax1.plot(time_points, signal_data, 'k-', linewidth=1.5, zorder=1)
        
        # Labels and title
        channel_name = channel_names[channel_idx] if channel_names else f"Channel {channel_idx}"
        predicted_class = "Stressed" if np.argmax(predictions[0]) == 1 else "Relaxed"
        confidence = predictions[0][np.argmax(predictions[0])] * 100
        
        ax1.set_title(f'Grad-CAM Visualization - {channel_name}\n'
                     f'Prediction: {predicted_class} (Confidence: {confidence:.1f}%)',
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Points', fontsize=10)
        ax1.set_ylabel('Amplitude (standardized)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Heatmap only
        im = ax2.imshow(heatmap_resized.reshape(1, -1), 
                       cmap='jet', aspect='auto', 
                       extent=[0, len(signal_data), 0, 1])
        ax2.set_title('Grad-CAM Heatmap (Red = High Importance)', fontsize=11)
        ax2.set_xlabel('Time Points', fontsize=10)
        ax2.set_yticks([])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
        cbar.set_label('Activation Importance', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        
        return fig
    
    def visualize_multi_channel(self, input_data, channel_indices=[0, 1, 2], 
                               class_idx=None, channel_names=None, save_path=None):
        """
        Visualize Grad-CAM for multiple channels simultaneously
        
        Parameters:
        - input_data: Input EEG segment
        - channel_indices: List of channel indices to visualize
        - class_idx: Target class for Grad-CAM
        - channel_names: List of channel names
        - save_path: Path to save the visualization
        """
        # Compute Grad-CAM
        heatmap, predictions = self.compute_gradcam(input_data, class_idx)
        
        # Ensure input has correct shape
        if len(input_data.shape) == 3:
            input_data = input_data[0]
        
        # Create figure with subplots
        n_channels = len(channel_indices)
        fig, axes = plt.subplots(n_channels, 1, figsize=(14, 4*n_channels))
        
        if n_channels == 1:
            axes = [axes]
        
        predicted_class = "Stressed" if np.argmax(predictions[0]) == 1 else "Relaxed"
        confidence = predictions[0][np.argmax(predictions[0])] * 100
        
        fig.suptitle(f'Multi-Channel Grad-CAM Visualization\n'
                    f'Prediction: {predicted_class} (Confidence: {confidence:.1f}%)',
                    fontsize=14, fontweight='bold')
        
        for idx, channel_idx in enumerate(channel_indices):
            ax = axes[idx]
            
            # Get signal
            signal_data = input_data[channel_idx, :]
            time_points = np.arange(len(signal_data))
            
            # Resize heatmap
            heatmap_resized = self._resize_heatmap(heatmap, len(signal_data))
            
            # Create colored background
            for i in range(len(signal_data) - 1):
                alpha = heatmap_resized[i]
                color = cm.jet(heatmap_resized[i])
                ax.axvspan(i, i+1, alpha=alpha*0.5, color=color, zorder=0)
            
            # Plot signal
            ax.plot(time_points, signal_data, 'k-', linewidth=1.5, zorder=1)
            
            # Labels
            channel_name = channel_names[channel_idx] if channel_names else f"Channel {channel_idx}"
            ax.set_title(f'{channel_name}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time Points', fontsize=9)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-channel Grad-CAM visualization saved to {save_path}")
        
        return fig
    
    def _resize_heatmap(self, heatmap, target_length):
        """
        Resize heatmap to match signal length using interpolation
        
        Parameters:
        - heatmap: Original heatmap
        - target_length: Target length
        
        Returns:
        - resized_heatmap: Resized heatmap
        """
        from scipy.ndimage import zoom
        
        zoom_factor = target_length / len(heatmap)
        resized_heatmap = zoom(heatmap, zoom_factor, order=1)
        
        return resized_heatmap
    
    def generate_importance_map(self, input_data, class_idx=None):
        """
        Generate a channel-wise importance map
        
        Parameters:
        - input_data: Input EEG segment
        - class_idx: Target class
        
        Returns:
        - importance_scores: Array of importance scores per channel
        """
        heatmap, _ = self.compute_gradcam(input_data, class_idx)
        
        # Ensure input has correct shape
        if len(input_data.shape) == 3:
            input_data = input_data[0]
        
        n_channels = input_data.shape[0]
        importance_scores = np.zeros(n_channels)
        
        # Calculate importance for each channel
        for i in range(n_channels):
            signal = input_data[i, :]
            heatmap_resized = self._resize_heatmap(heatmap, len(signal))
            
            # Weighted importance: signal magnitude weighted by heatmap
            importance_scores[i] = np.mean(np.abs(signal) * heatmap_resized)
        
        # Normalize
        importance_scores = importance_scores / importance_scores.max()
        
        return importance_scores
    
    def plot_channel_importance(self, input_data, channel_names=None, save_path=None):
        """
        Plot bar chart of channel importance
        
        Parameters:
        - input_data: Input EEG segment
        - channel_names: List of channel names
        - save_path: Path to save the plot
        """
        importance_scores = self.generate_importance_map(input_data)
        
        if channel_names is None:
            channel_names = [f"Ch{i}" for i in range(len(importance_scores))]
        
        plt.figure(figsize=(12, 6))
        colors = cm.jet(importance_scores)
        plt.bar(range(len(importance_scores)), importance_scores, color=colors)
        plt.xlabel('EEG Channel', fontsize=11)
        plt.ylabel('Importance Score', fontsize=11)
        plt.title('Channel-wise Importance for Stress Classification', 
                 fontsize=12, fontweight='bold')
        plt.xticks(range(len(channel_names)), channel_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Channel importance plot saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    print("Grad-CAM module loaded successfully!")
    print("\nUsage example:")
    print("  gradcam = GradCAM(model)")
    print("  gradcam.visualize_gradcam(input_segment, channel_idx=0)")
