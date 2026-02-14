"""
==========================================
MI-based Band Selection Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Mutual Information-based frequency band selection inspired by winner paper.
"""

import logging
import numpy as np
from scipy import signal
from scipy.integrate import simpson
from typing import List, Tuple
from sklearn.feature_selection import mutual_info_classif

from config import Config
from preprocess import SignalPreprocessor


logger = logging.getLogger(__name__)


class MIBandSelector:
    """Select optimal frequency band using Mutual Information."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = SignalPreprocessor(config)
        
    def compute_log_variance_features(self, epochs: np.ndarray) -> np.ndarray:
        """
        Compute log-variance features per channel.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples]
            
        Returns
        -------
        np.ndarray
            Features of shape [n_epochs, n_channels]
        """
        # Compute variance across time for each epoch and channel
        variances = np.var(epochs, axis=2)  # Shape: [n_epochs, n_channels]
        
        # Apply log with small epsilon for numerical stability
        log_var = np.log(variances + 1e-8)
        
        return log_var
    
    def compute_bandpower_features(
        self,
        epochs: np.ndarray,
        fs: float,
        band: Tuple[float, float]
    ) -> np.ndarray:
        """
        Compute bandpower features per channel using Welch's method.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples]
        fs : float
            Sampling rate
        band : tuple of (low, high)
            Frequency band
            
        Returns
        -------
        np.ndarray
            Bandpower features of shape [n_epochs, n_channels]
        """
        n_epochs, n_channels, n_samples = epochs.shape
        bandpower = np.zeros((n_epochs, n_channels))
        
        for ep in range(n_epochs):
            for ch in range(n_channels):
                # Compute PSD using Welch's method
                freqs, psd = signal.welch(
                    epochs[ep, ch],
                    fs=fs,
                    nperseg=min(n_samples, 256)
                )
                
                # Find indices within the band
                idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
                
                # Integrate PSD within the band
                if np.sum(idx_band) > 0:
                    bandpower[ep, ch] = simpson(psd[idx_band], freqs[idx_band])
                else:
                    bandpower[ep, ch] = 0.0
        
        # Log transform
        bandpower = np.log(bandpower + 1e-8)
        
        return bandpower
    
    def extract_band_features(
        self,
        epochs: np.ndarray,
        fs: float,
        band: Tuple[float, float]
    ) -> np.ndarray:
        """
        Extract combined features for a specific frequency band.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples]
        fs : float
            Sampling rate
        band : tuple of (low, high)
            Frequency band
            
        Returns
        -------
        np.ndarray
            Combined features of shape [n_epochs, n_channels * 2]
            (14 log-variance + 14 bandpower = 28 features)
        """
        # Compute log-variance
        log_var = self.compute_log_variance_features(epochs)  # [n_epochs, 14]
        
        # Compute bandpower
        bandpower = self.compute_bandpower_features(epochs, fs, band)  # [n_epochs, 14]
        
        # Concatenate features
        features = np.hstack([log_var, bandpower])  # [n_epochs, 28]
        
        return features
    
    def compute_mi_for_band(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        fs: float,
        band: Tuple[float, float]
    ) -> float:
        """
        Compute MI score for a frequency band.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples]
        labels : np.ndarray
            Labels of shape [n_epochs]
        fs : float
            Sampling rate
        band : tuple of (low, high)
            Frequency band
            
        Returns
        -------
        float
            Mean MI score for the band
        """
        # Filter epochs to the band
        filtered_epochs = self._filter_epochs_to_band(epochs, fs, band)
        
        # Extract features
        features = self.extract_band_features(filtered_epochs, fs, band)
        
        # Compute MI for each feature
        mi_scores = mutual_info_classif(
            features,
            labels,
            discrete_features=False,
            random_state=self.config.random_state
        )
        
        # Return mean MI across features
        return np.mean(mi_scores)
    
    def _filter_epochs_to_band(
        self,
        epochs: np.ndarray,
        fs: float,
        band: Tuple[float, float]
    ) -> np.ndarray:
        """
        Apply bandpass filter to epochs.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples]
        fs : float
            Sampling rate
        band : tuple of (low, high)
            Frequency band
            
        Returns
        -------
        np.ndarray
            Filtered epochs
        """
        n_epochs, n_channels, n_samples = epochs.shape
        filtered_epochs = np.zeros_like(epochs)
        
        # Design filter
        nyquist = fs / 2.0
        low = band[0] / nyquist
        high = band[1] / nyquist
        
        # Ensure valid band
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        sos = signal.butter(
            self.config.filter_order,
            [low, high],
            btype='band',
            output='sos'
        )
        
        # Apply to each epoch and channel
        for ep in range(n_epochs):
            for ch in range(n_channels):
                filtered_epochs[ep, ch] = signal.sosfiltfilt(sos, epochs[ep, ch])
        
        return filtered_epochs
    
    def select_optimal_band(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        fs: float
    ) -> Tuple[float, float]:
        """
        Select optimal frequency band using MI.
        
        Parameters
        ----------
        epochs : np.ndarray
            Training epochs of shape [n_epochs, n_channels, n_samples]
        labels : np.ndarray
            Training labels of shape [n_epochs]
        fs : float
            Sampling rate
            
        Returns
        -------
        tuple of (low, high)
            Optimal frequency band
        """
        logger.info("Selecting optimal frequency band using MI...")
        
        # Get candidate bands
        bands = self.config.get_mi_bands()
        logger.info(f"Testing {len(bands)} candidate bands")
        
        # Compute MI for each band
        mi_scores = []
        for band in bands:
            mi_score = self.compute_mi_for_band(epochs, labels, fs, band)
            mi_scores.append(mi_score)
            logger.debug(f"Band {band}: MI = {mi_score:.4f}")
        
        mi_scores = np.array(mi_scores)
        
        # Select bands based on strategy
        if self.config.mi_selection_method == "above_mean":
            mean_mi = np.mean(mi_scores)
            selected_idx = np.where(mi_scores > mean_mi)[0]
            logger.info(f"Mean MI: {mean_mi:.4f}, selected {len(selected_idx)} bands")
        else:  # top_k
            selected_idx = np.argsort(mi_scores)[-self.config.mi_top_k:]
            logger.info(f"Selected top {self.config.mi_top_k} bands")
        
        if len(selected_idx) == 0:
            logger.warning("No bands selected, using best single band")
            selected_idx = np.array([np.argmax(mi_scores)])
        
        # Get selected bands
        selected_bands = [bands[i] for i in selected_idx]
        
        # Merge continuous bands
        merged_bands = self._merge_continuous_bands(selected_bands)
        logger.info(f"Merged into {len(merged_bands)} continuous bands: {merged_bands}")
        
        # Choose widest band
        widths = [(high - low, (low, high)) for low, high in merged_bands]
        widths.sort(reverse=True)
        optimal_band = widths[0][1]
        
        logger.info(f"Optimal band selected: {optimal_band[0]:.1f}-{optimal_band[1]:.1f} Hz")
        
        return optimal_band
    
    def _merge_continuous_bands(self, bands: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Merge overlapping or continuous frequency bands.
        
        Parameters
        ----------
        bands : list of tuples
            List of (low, high) frequency bands
            
        Returns
        -------
        list of tuples
            Merged bands
        """
        if not bands:
            return []
        
        # Sort by lower frequency
        sorted_bands = sorted(bands, key=lambda x: x[0])
        
        merged = [sorted_bands[0]]
        
        for current_low, current_high in sorted_bands[1:]:
            last_low, last_high = merged[-1]
            
            # Check if current band overlaps or is continuous with last
            if current_low <= last_high + self.config.mi_band_step:
                # Merge
                merged[-1] = (last_low, max(last_high, current_high))
            else:
                # Add as new band
                merged.append((current_low, current_high))
        
        return merged
