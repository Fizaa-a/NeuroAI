import os
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessor import EEGPreprocessor
from gradcam import GradCAM
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class EEGEngine:
    def __init__(self, model_path='best_model.h5'):
        # Prefer best_model.h5, fallback to stress_detection_model.h5
        if not os.path.exists(model_path) and os.path.exists('stress_detection_model.h5'):
            self.model_path = 'stress_detection_model.h5'
        else:
            self.model_path = model_path
            
        self.model = None
        self.preprocessor = EEGPreprocessor(sampling_rate=500)
        self.grad_cam = None
        
        # Store latest session data
        self.segments = None
        self.channel_names = None
        self.predictions = None
        self.confidences = None
        self.status = "System Ready"

    def is_model_loaded(self):
        return self.model is not None

    def load_model(self, force=False):
        """Thread-safe model loading"""
        if self.model and not force:
            return True
        
        try:
            target = self.model_path
            if not os.path.exists(target):
                # Final fallback check
                for f in ['best_model.h5', 'stress_detection_model.h5']:
                    if os.path.exists(f): 
                        target = f
                        break
            
            if os.path.exists(target):
                print(f"DEBUG: Loading model from {target}")
                self.model = keras.models.load_model(target)
                self.grad_cam = None # Defer initialization until first prediction
                self.status = f"Model Loaded ({os.path.basename(target)})"
                return True
            else:
                self.status = "Error: No model files found (.h5)"
                print("DEBUG: No model file found!")
                return False
        except Exception as e:
            self.status = f"Load Error: {str(e)}"
            print(f"DEBUG: Model Load Crash: {e}")
            return False

    def process_and_analyze(self, file_path, on_progress_callback=None):
        """Processes the EDF file and runs prediction"""
        try:
            # Phase 1: Signal Cleaning
            if on_progress_callback: on_progress_callback(10, "Accessing EEG Recording...")
            
            # Ensure model is ready before we spend time processing
            if not self.model:
                if not self.load_model():
                    raise RuntimeError("Failed to load AI model. Please check if .h5 file exists.")

            if on_progress_callback: on_progress_callback(25, "Cleaning Signal Artifacts...")
            
            # Preprocess
            segments, channel_names = self.preprocessor.preprocess_pipeline(
                file_path, 
                apply_ica=False, 
                segment_length=2.0,
                overlap=1.0
            )
            
            if not segments:
                raise ValueError("No valid EEG segments could be extracted from the file.")

            if on_progress_callback: on_progress_callback(50, "Extracting Neural Features...")
            
            # Phase 2: Signal Preparation
            eeg_segments = self.preprocessor.extract_first_20_channels(segments)
            self.segments = np.array(eeg_segments)
            self.channel_names = channel_names[:20]
            
            if on_progress_callback: on_progress_callback(75, "Running Neural Inference...")
            
            # Step 3: Inference
            predictions_raw = self.model.predict(self.segments, verbose=0)
            self.predictions = np.argmax(predictions_raw, axis=1)
            self.confidences = np.max(predictions_raw, axis=1)
            
            # Summary stats
            total = len(self.predictions)
            stressed = int(np.sum(self.predictions == 1))
            relaxed = int(np.sum(self.predictions == 0))
            
            overall_res = 1 if stressed > relaxed else 0
            avg_conf = float(np.mean(self.confidences) * 100)
            stress_pct = float((stressed / total) * 100)
            
            results = {
                "overall": "STRESSED" if overall_res == 1 else "RELAXED",
                "prediction": overall_res,
                "confidence": avg_conf,
                "stress_percentage": stress_pct,
                "total_segments": total,
                "stressed_segments": stressed,
                "relaxed_segments": relaxed
            }
            
            print(f"DEBUG: Analysis Results: {results}")
            if on_progress_callback: on_progress_callback(100, "Analysis Complete.")
            return results

        except Exception as e:
            self.status = f"Analysis Error: {str(e)}"
            print(f"DEBUG: Analysis Crash: {e}")
            raise e

    def get_gradcam_figure(self, segment_idx=None):
        """Generates the Grad-CAM figure"""
        if self.segments is None:
            print("DEBUG: Segments missing")
            return None
        
        if self.grad_cam is None:
            print("DEBUG: Initializing Grad-CAM...")
            try:
                self.grad_cam = GradCAM(self.model)
            except Exception as e:
                print(f"DEBUG: Failed to initialize GradCAM: {e}")
                return None
                
        try:
            if segment_idx is None:
                # Target the most confident detection
                segment_idx = np.argmax(self.confidences)
            
            indices = [i for i in [0, 1, 16] if i < len(self.channel_names)]
            
            fig = self.grad_cam.visualize_multi_channel(
                self.segments[segment_idx],
                channel_indices=indices,
                channel_names=self.channel_names,
                save_path=None
            )
            return fig
        except Exception as e:
            print(f"DEBUG: Grad-CAM Fig Crash: {e}")
            return None
