"""
==========================================
Preprocessing Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Signal filtering, notch filter, bandpass filter, and epoching.
"""

import logging
import numpy as np
from scipy import signal
from typing import Tuple, Optional, Dict
import joblib
from pathlib import Path

from config import Config
from io_data import EEGFile, DatasetLoader
from utils import get_cache_path


logger = logging.getLogger(__name__)


class SignalPreprocessor:
    """Preprocess EEG signals: filtering and epoching."""
    
    def __init__(self, config: Config):
        self.config = config
        
    def apply_notch_filter(self, data: np.ndarray, fs: float) -> np.ndarray:
        """
        Apply notch filter to remove mains interference (50Hz).
        
        Parameters
        ----------
        data : np.ndarray
            EEG data of shape [n_samples, n_channels]
        fs : float
            Sampling rate in Hz
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        # Design notch filter
        b, a = signal.iirnotch(
            self.config.notch_freq,
            self.config.notch_quality,
            fs
        )
        
        # Apply to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.filtfilt(b, a, data[:, ch])
        
        return filtered
    
    def apply_bandpass_filter(
        self,
        data: np.ndarray,
        fs: float,
        lowcut: Optional[float] = None,
        highcut: Optional[float] = None,
        order: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Parameters
        ----------
        data : np.ndarray
            EEG data of shape [n_samples, n_channels]
        fs : float
            Sampling rate in Hz
        lowcut : float, optional
            Lower cutoff frequency. If None, uses config.global_lowcut
        highcut : float, optional
            Higher cutoff frequency. If None, uses config.global_highcut
        order : int, optional
            Filter order. If None, uses config.filter_order
            
        Returns
        -------
        np.ndarray
            Filtered data
        """
        if lowcut is None:
            lowcut = self.config.global_lowcut
        if highcut is None:
            highcut = self.config.global_highcut
        if order is None:
            order = self.config.filter_order
        
        # Validate frequencies
        nyquist = fs / 2.0
        if lowcut >= nyquist or highcut >= nyquist:
            logger.warning(
                f"Cutoff frequencies ({lowcut}, {highcut}) exceed Nyquist ({nyquist})"
            )
            highcut = min(highcut, nyquist * 0.95)
            lowcut = min(lowcut, highcut * 0.9)
        
        # Design Butterworth bandpass filter
        sos = signal.butter(
            order,
            [lowcut, highcut],
            btype='band',
            fs=fs,
            output='sos'
        )
        
        # Apply to each channel
        filtered = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered[:, ch] = signal.sosfiltfilt(sos, data[:, ch])
        
        return filtered
    
    def create_epochs(
        self,
        data: np.ndarray,
        fs: float,
        metadata: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment continuous data into overlapping epochs.
        
        Parameters
        ----------
        data : np.ndarray
            EEG data of shape [n_samples, n_channels]
        fs : float
            Sampling rate in Hz
        metadata : dict, optional
            Metadata containing quality and motion info
            
        Returns
        -------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples_per_epoch]
        quality_scores : np.ndarray or None
            Quality score per epoch if metadata available
        """
        window_samples = int(self.config.window_sec * fs)
        stride_samples = int(self.config.get_stride_sec() * fs)
        
        n_samples, n_channels = data.shape
        
        # Calculate number of epochs
        n_epochs = (n_samples - window_samples) // stride_samples + 1
        
        if n_epochs < 1:
            logger.warning("Signal too short for even one epoch")
            return np.array([]), None
        
        # Preallocate epochs array
        epochs = np.zeros((n_epochs, n_channels, window_samples))
        
        # Extract epochs
        for i in range(n_epochs):
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            epochs[i] = data[start_idx:end_idx].T  # Transpose to [n_channels, n_samples]
        
        # Compute quality scores if metadata available
        quality_scores = None
        if metadata is not None and self.config.use_quality_weighting:
            quality_scores = self._compute_quality_scores(
                metadata, n_epochs, stride_samples, window_samples
            )
        
        logger.debug(f"Created {n_epochs} epochs from {n_samples} samples")
        
        return epochs, quality_scores
    
    def _compute_quality_scores(
        self,
        metadata: Dict,
        n_epochs: int,
        stride_samples: int,
        window_samples: int
    ) -> np.ndarray:
        """
        Compute quality score for each epoch based on contact quality and motion.
        
        Parameters
        ----------
        metadata : dict
            Metadata with contact_quality and motion arrays
        n_epochs : int
            Number of epochs
        stride_samples : int
            Stride in samples
        window_samples : int
            Window size in samples
            
        Returns
        -------
        np.ndarray
            Quality scores of shape [n_epochs]
        """
        quality_scores = np.ones(n_epochs)
        
        # Contact quality component
        if metadata.get('contact_quality') is not None:
            cq = metadata['contact_quality']  # Shape: [n_samples, n_cq_channels]
            
            for i in range(n_epochs):
                start_idx = i * stride_samples
                end_idx = start_idx + window_samples
                
                # Mean contact quality across channels and time (normalized to [0, 1])
                # Assuming CQ scale is 0-4, where 4 is best
                epoch_cq = cq[start_idx:end_idx].mean() / 4.0
                quality_scores[i] *= np.clip(epoch_cq, 0, 1)
        
        # Motion component (high motion = lower quality)
        if metadata.get('motion') is not None:
            motion = metadata['motion']  # Shape: [n_samples, n_motion_channels]
            
            for i in range(n_epochs):
                start_idx = i * stride_samples
                end_idx = start_idx + window_samples
                
                epoch_motion = motion[start_idx:end_idx]
                
                # Compute motion energy
                # Accelerometer: compute norm
                if epoch_motion.shape[1] >= 7:  # Has AccX, AccY, AccZ
                    acc = epoch_motion[:, 4:7]  # Indices for AccX, AccY, AccZ
                    acc_norm = np.linalg.norm(acc, axis=1).mean()
                    # Normalize (rough estimate, adjust based on data)
                    motion_factor = 1.0 - np.clip(acc_norm / 10.0, 0, 0.5)
                    quality_scores[i] *= motion_factor
        
        # Clip final scores to [0, 1]
        quality_scores = np.clip(quality_scores, 0, 1)
        
        return quality_scores
    
    def preprocess_file(
        self,
        eeg_file: EEGFile,
        loader: DatasetLoader,
        apply_global_filter: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        """
        Preprocess a single file: load, filter, epoch.
        
        Parameters
        ----------
        eeg_file : EEGFile
            File metadata
        loader : DatasetLoader
            Dataset loader instance
        apply_global_filter : bool
            Whether to apply global bandpass filter
            
        Returns
        -------
        epochs : np.ndarray
            Epoched data of shape [n_epochs, n_channels, n_samples_per_epoch]
        quality_scores : np.ndarray or None
            Quality scores per epoch
        fs : float
            Sampling rate
        """
        # Check cache
        if self.config.use_cache:
            cache_path = get_cache_path(
                eeg_file.path,
                self.config.cache_dir,
                suffix="_preprocessed"
            )
            
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    epochs = cached['epochs']
                    quality_scores = cached.get('quality_scores', None)
                    fs = float(cached['fs'])
                    logger.debug(f"Loaded from cache: {eeg_file.path.name}")
                    return epochs, quality_scores, fs
                except Exception as e:
                    logger.warning(f"Cache load failed: {e}")
        
        # Load raw data
        data, fs, metadata = loader.load_raw_file(eeg_file)
        
        # Apply notch filter (50Hz)
        data = self.apply_notch_filter(data, fs)
        
        # Apply global bandpass filter if requested
        if apply_global_filter:
            data = self.apply_bandpass_filter(data, fs)
        
        # Create epochs
        epochs, quality_scores = self.create_epochs(data, fs, metadata)
        
        # Save to cache
        if self.config.use_cache and len(epochs) > 0:
            try:
                np.savez_compressed(
                    cache_path,
                    epochs=epochs,
                    quality_scores=quality_scores,
                    fs=fs
                )
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
        
        return epochs, quality_scores, fs
    
    def preprocess_all_files(
        self,
        files: list,
        loader: DatasetLoader
    ) -> Tuple[list, list, list, list]:
        """
        Preprocess all files in parallel.
        
        Parameters
        ----------
        files : list of EEGFile
            List of files to preprocess
        loader : DatasetLoader
            Dataset loader instance
            
        Returns
        -------
        all_epochs : list of np.ndarray
            List of epoch arrays
        all_quality : list of np.ndarray or None
            List of quality arrays
        all_labels : list of int
            List of labels
        all_fs : list of float
            List of sampling rates
        """
        logger.info(f"Preprocessing {len(files)} files...")
        
        all_epochs = []
        all_quality = []
        all_labels = []
        all_fs = []
        
        # Use joblib for parallel processing
        if self.config.n_jobs != 1:
            results = joblib.Parallel(n_jobs=self.config.n_jobs)(
                joblib.delayed(self.preprocess_file)(f, loader)
                for f in files
            )
            
            for i, (epochs, quality, fs) in enumerate(results):
                if len(epochs) > 0:
                    all_epochs.append(epochs)
                    all_quality.append(quality)
                    all_labels.append(files[i].label_idx)
                    all_fs.append(fs)
        else:
            # Sequential processing
            for file in files:
                epochs, quality, fs = self.preprocess_file(file, loader)
                if len(epochs) > 0:
                    all_epochs.append(epochs)
                    all_quality.append(quality)
                    all_labels.append(file.label_idx)
                    all_fs.append(fs)
        
        logger.info(f"Preprocessed {len(all_epochs)} files successfully")
        
        return all_epochs, all_quality, all_labels, all_fs
