"""
Prediction Script for Stress Detection
Classifies a single EEG file as stressed or relaxed
"""

import sys
import os
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

from preprocessor import EEGPreprocessor
from gradcam import GradCAM

def predict_stress(edf_file_path, model_path='stress_detection_model.h5', 
                  visualize_gradcam=True):
    """
    Predict stress level from an EEG file
    
    Parameters:
    - edf_file_path: Path to the EDF file
    - model_path: Path to the trained model
    - visualize_gradcam: Whether to generate Grad-CAM visualizations
    
    Returns:
    - prediction: Overall prediction (0=Relaxed, 1=Stressed)
    - confidence: Confidence score
    - segment_predictions: Predictions for each segment
    """
    print("=" * 60)
    print("STRESS DETECTION - PREDICTION")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(edf_file_path):
        print(f"Error: EDF file '{edf_file_path}' not found!")
        return None
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return None
    
    # Step 1: Load the trained model
    print("\n[1/4] Loading trained model...")
    try:
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None
    
    # Step 2: Preprocess the EEG file
    print("\n[2/4] Preprocessing EEG data...")
    preprocessor = EEGPreprocessor(sampling_rate=500)
    
    try:
        segments, channel_names = preprocessor.preprocess_pipeline(
            edf_file_path,
            apply_ica=False,  # Set to True if you want ICA (slower but cleaner)
            segment_length=2.0,
            overlap=1.0
        )
        
        # Extract first 20 channels
        segments = preprocessor.extract_first_20_channels(segments)
        segments = np.array(segments)
        
        print(f"Extracted {len(segments)} segments")
        print(f"Segment shape: {segments[0].shape}")
        
    except Exception as e:
        print(f"Error preprocessing file: {str(e)}")
        return None
    
    # Step 3: Make predictions
    print("\n[3/4] Making predictions...")
    
    # Predict for all segments
    predictions = model.predict(segments, verbose=0)
    segment_predictions = np.argmax(predictions, axis=1)
    segment_confidences = np.max(predictions, axis=1)
    
    # Overall prediction (majority vote)
    stressed_count = np.sum(segment_predictions == 1)
    relaxed_count = np.sum(segment_predictions == 0)
    
    overall_prediction = 1 if stressed_count > relaxed_count else 0
    overall_confidence = np.mean(segment_confidences) * 100
    
    # Calculate percentage of stressed segments
    stress_percentage = (stressed_count / len(segment_predictions)) * 100
    
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"File: {os.path.basename(edf_file_path)}")
    print(f"\nOverall Prediction: {'STRESSED' if overall_prediction == 1 else 'RELAXED'}")
    print(f"Confidence: {overall_confidence:.1f}%")
    print(f"\nSegment Analysis:")
    print(f"  Total segments: {len(segment_predictions)}")
    print(f"  Stressed segments: {stressed_count} ({stress_percentage:.1f}%)")
    print(f"  Relaxed segments: {relaxed_count} ({100-stress_percentage:.1f}%)")
    print("=" * 60)
    
    # Step 4: Visualize with Grad-CAM (if requested)
    if visualize_gradcam:
        print("\n[4/4] Generating Grad-CAM visualizations...")
        
        # Initialize Grad-CAM
        gradcam = GradCAM(model)
        
        # Find a strongly predicted stressed segment
        stressed_segments_idx = np.where(segment_predictions == 1)[0]
        if len(stressed_segments_idx) > 0:
            # Get the segment with highest stress confidence
            best_stressed_idx = stressed_segments_idx[np.argmax(segment_confidences[stressed_segments_idx])]
            
            print(f"\nVisualizing strongly stressed segment (#{best_stressed_idx})...")
            
            # Visualize single channel
            gradcam.visualize_gradcam(
                segments[best_stressed_idx],
                channel_idx=0,  # Fp1
                channel_names=channel_names[:20],
                save_path='gradcam_single_channel.png'
            )
            
            # Visualize multiple important channels
            important_channels = [0, 1, 16]  # Fp1, Fp2, Fz (frontal regions)
            gradcam.visualize_multi_channel(
                segments[best_stressed_idx],
                channel_indices=important_channels,
                channel_names=channel_names[:20],
                save_path='gradcam_multi_channel.png'
            )
            
            # Channel importance
            gradcam.plot_channel_importance(
                segments[best_stressed_idx],
                channel_names=channel_names[:20],
                save_path='channel_importance.png'
            )
    
    # Create summary plot
    create_summary_plot(segment_predictions, segment_confidences, overall_prediction)
    
    print("\nPrediction completed successfully!")
    
    return overall_prediction, overall_confidence, segment_predictions

def create_summary_plot(segment_predictions, segment_confidences, overall_prediction):
    """
    Create a summary visualization of predictions
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Predictions over time
    colors = ['green' if pred == 0 else 'red' for pred in segment_predictions]
    axes[0].bar(range(len(segment_predictions)), segment_predictions, color=colors, alpha=0.6)
    axes[0].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Segment Number', fontsize=11)
    axes[0].set_ylabel('Prediction (0=Relaxed, 1=Stressed)', fontsize=11)
    axes[0].set_title('Stress Predictions Across EEG Segments', fontsize=12, fontweight='bold')
    axes[0].set_ylim([-0.1, 1.1])
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence over time
    axes[1].plot(range(len(segment_confidences)), segment_confidences * 100, 
                marker='o', linestyle='-', color='blue', markersize=4)
    axes[1].axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Segment Number', fontsize=11)
    axes[1].set_ylabel('Confidence (%)', fontsize=11)
    axes[1].set_title('Prediction Confidence Across Segments', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 100])
    axes[1].grid(True, alpha=0.3)
    
    # Add overall prediction text
    prediction_text = 'Overall: STRESSED' if overall_prediction == 1 else 'Overall: RELAXED'
    fig.text(0.5, 0.02, prediction_text, ha='center', fontsize=14, 
            fontweight='bold', color='red' if overall_prediction == 1 else 'green')
    
    plt.tight_layout()
    plt.savefig('prediction_summary.png', dpi=300, bbox_inches='tight')
    print("Summary plot saved to 'prediction_summary.png'")
    # Do not call plt.show() in batch/non-interactive runs (can block); close the figure instead
    try:
        plt.close(fig)
    except Exception:
        plt.close()

def batch_predict(directory_path, model_path='stress_detection_model.h5'):
    """
    Predict stress for all EDF files in a directory
    
    Parameters:
    - directory_path: Path to directory containing EDF files
    - model_path: Path to trained model
    """
    import pandas as pd
    
    results = []
    
    # Get all EDF files
    edf_files = [f for f in os.listdir(directory_path) if f.endswith('.edf')]
    
    print(f"Found {len(edf_files)} EDF files")
    print("\nProcessing files...\n")
    
    for idx, filename in enumerate(edf_files):
        file_path = os.path.join(directory_path, filename)
        print(f"[{idx+1}/{len(edf_files)}] Processing {filename}...")
        
        try:
            prediction, confidence, _ = predict_stress(
                file_path, 
                model_path, 
                visualize_gradcam=False
            )
            
            results.append({
                'Filename': filename,
                'Prediction': 'Stressed' if prediction == 1 else 'Relaxed',
                'Confidence': f"{confidence:.1f}%"
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            results.append({
                'Filename': filename,
                'Prediction': 'Error',
                'Confidence': 'N/A'
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('batch_predictions.csv', index=False)
    print("\nBatch predictions saved to 'batch_predictions.csv'")
    
    return df

if __name__ == "__main__":
    # CLI: use argparse so flags (like --batch) don't get interpreted as the model path
    import argparse

    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Predict stress from an EDF file or directory (use --batch)'
    )
    parser.add_argument('path', help='path to EDF file (or directory when --batch is used)')
    parser.add_argument('--model', '-m', default='stress_detection_model.h5',
                        help='path to trained model (default: stress_detection_model.h5)')
    parser.add_argument('--batch', action='store_true', help='process all EDF files in a directory')
    parser.add_argument('--no-gradcam', action='store_true', help="disable Grad-CAM visualization")

    args = parser.parse_args()

    if args.batch:
        batch_predict(args.path, args.model)
    else:
        visualize = not args.no_gradcam
        predict_stress(args.path, args.model, visualize_gradcam=visualize)
