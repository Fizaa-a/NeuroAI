"""
EEG Signal Preprocessing Module
Implements filtering, artifact removal, and segmentation
"""

import mne
import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.integrate import simps

class EEGPreprocessor:
    def __init__(self, sampling_rate=500):
        """
        Initialize the EEG preprocessor
        
        Parameters:
        - sampling_rate: Sampling frequency of EEG data (default: 500 Hz)
        """
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        
    def load_edf(self, file_path):
        """
        Load EEG data from EDF file
        
        Parameters:
        - file_path: Path to the EDF file
        
        Returns:
        - raw: MNE Raw object containing EEG data
        """
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        return raw
    
    def apply_filters(self, raw):
        """
        Apply high-pass filter and notch filter to remove noise
        
        Parameters:
        - raw: MNE Raw object
        
        Returns:
        - raw: Filtered MNE Raw object
        """
        # Apply high-pass filter at 0.5 Hz (removes slow drifts)
        raw.filter(l_freq=0.5, h_freq=None, verbose=False)
        
        # Apply low-pass filter at 60 Hz (removes high-frequency noise)
        raw.filter(l_freq=None, h_freq=60, verbose=False)
        
        # Apply notch filter at 50 Hz (removes power line interference)
        raw.notch_filter(freqs=50, verbose=False)
        
        return raw
    
    def remove_artifacts_ica(self, raw, n_components=15):
        """
        Remove artifacts using Independent Component Analysis (ICA)
        
        Parameters:
        - raw: MNE Raw object
        - n_components: Number of ICA components
        
        Returns:
        - raw: Cleaned MNE Raw object
        """
        # Create ICA object
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, verbose=False)
        
        # Fit ICA
        ica.fit(raw, verbose=False)
        
        # Automatically find and exclude eye blink and heart artifacts
        # This is a simplified version - in production, you'd manually inspect
        eog_indices, eog_scores = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices[:2] if len(eog_indices) > 0 else []
        
        # Apply ICA to remove artifacts
        raw = ica.apply(raw, verbose=False)
        
        return raw
    
    def segment_data(self, raw, segment_length=2.0, overlap=1.0):
        """
        Segment EEG data into overlapping windows
        
        Parameters:
        - raw: MNE Raw object
        - segment_length: Length of each segment in seconds (default: 2.0)
        - overlap: Overlap between segments in seconds (default: 1.0)
        
        Returns:
        - segments: List of numpy arrays, each containing one segment
        - channel_names: List of channel names
        """
        data = raw.get_data()
        n_channels, n_samples = data.shape
        
        segment_samples = int(segment_length * self.sampling_rate)
        overlap_samples = int(overlap * self.sampling_rate)
        step = segment_samples - overlap_samples
        
        segments = []
        start = 0
        
        while start + segment_samples <= n_samples:
            segment = data[:, start:start + segment_samples]
            segments.append(segment)
            start += step
        
        return segments, raw.ch_names
    
    def standardize_segments(self, segments):
        """
        Standardize (Z-score normalization) each segment
        
        Parameters:
        - segments: List of numpy arrays
        
        Returns:
        - standardized_segments: List of standardized numpy arrays
        """
        standardized_segments = []
        
        for segment in segments:
            # Standardize each channel independently
            segment_standardized = np.zeros_like(segment)
            for i in range(segment.shape[0]):
                channel_data = segment[i, :].reshape(-1, 1)
                standardized = self.scaler.fit_transform(channel_data).flatten()
                segment_standardized[i, :] = standardized
            
            standardized_segments.append(segment_standardized)
        
        return standardized_segments
    
    def preprocess_pipeline(self, file_path, apply_ica=True, segment_length=2.0, overlap=1.0):
        """
        Complete preprocessing pipeline
        
        Parameters:
        - file_path: Path to EDF file
        - apply_ica: Whether to apply ICA for artifact removal
        - segment_length: Length of segments in seconds
        - overlap: Overlap between segments in seconds
        
        Returns:
        - processed_segments: List of preprocessed segments
        - channel_names: List of channel names
        """
        # Load data
        raw = self.load_edf(file_path)
        
        # Apply filters
        raw = self.apply_filters(raw)
        
        # Remove artifacts (optional)
        if apply_ica:
            raw = self.remove_artifacts_ica(raw)
        
        # Segment data
        segments, channel_names = self.segment_data(raw, segment_length, overlap)
        
        # Standardize
        processed_segments = self.standardize_segments(segments)
        
        return processed_segments, channel_names
    
    def extract_first_20_channels(self, segments):
        """
        Extract only the first 20 EEG channels (remove ECG channel)
        """
        eeg_segments = [seg[:20, :] for seg in segments]
        return eeg_segments

    def get_bandpower(self, segment, bands=[(0.5, 4, 'delta'), (4, 8, 'theta'), (8, 12, 'alpha'), (12, 30, 'beta'), (30, 45, 'gamma')]):
        """
        Extract absolute band power for each channel in a segment
        Returns a feature vector of length (n_channels * n_bands)
        """
        n_channels, n_samples = segment.shape
        features = []
        
        for i in range(n_channels):
            # Compute PSD using Welch's method
            freqs, psd = signal.welch(segment[i], fs=self.sampling_rate, nperseg=min(n_samples, 256))
            
            for f_low, f_high, _ in bands:
                # Find frequency indices
                idx_band = np.logical_and(freqs >= f_low, freqs <= f_high)
                # Area under the curve (Power)
                band_power = simps(psd[idx_band], freqs[idx_band])
                features.append(band_power)
        
        return np.array(features)


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = EEGPreprocessor(sampling_rate=500)
    
    # Example usage
    # segments, channels = preprocessor.preprocess_pipeline("path/to/file.edf")
    # print(f"Number of segments: {len(segments)}")
    # print(f"Segment shape: {segments[0].shape}")
    print("EEG Preprocessor module loaded successfully!")
