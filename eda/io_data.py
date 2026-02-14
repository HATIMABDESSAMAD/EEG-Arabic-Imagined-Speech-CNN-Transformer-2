"""
==========================================
Data Loading Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Load EEG data files, extract labels, and organize dataset.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from config import Config


logger = logging.getLogger(__name__)


@dataclass
class EEGFile:
    """Container for a single EEG file with metadata."""
    path: Path
    label: str
    label_idx: int
    participant: Optional[str] = None
    session: Optional[str] = None


class DatasetLoader:
    """Load and organize ArEEG_Words dataset."""
    
    def __init__(self, config: Config):
        self.config = config
        self.files: List[EEGFile] = []
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        
    def discover_files(self) -> List[EEGFile]:
        """
        Discover all EEG files in the dataset directory.
        
        Returns
        -------
        List[EEGFile]
            List of discovered EEG files with metadata
        """
        logger.info(f"Discovering files in {self.config.data_dir}")
        
        if not self.config.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.config.data_dir}")
        
        # Get all subdirectories (word/class folders)
        class_folders = [f for f in self.config.data_dir.iterdir() if f.is_dir()]
        
        if not class_folders:
            raise ValueError(f"No class folders found in {self.config.data_dir}")
        
        logger.info(f"Found {len(class_folders)} class folders")
        
        # Build label mappings
        sorted_labels = sorted([f.name for f in class_folders])
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        logger.info(f"Classes: {list(self.label_to_idx.keys())}")
        
        # Collect all CSV files
        files = []
        for class_folder in class_folders:
            label = class_folder.name
            label_idx = self.label_to_idx[label]
            
            csv_files = list(class_folder.glob("*.csv"))
            
            for csv_file in csv_files:
                # Try to extract participant info from filename
                participant = self._extract_participant(csv_file.name)
                
                eeg_file = EEGFile(
                    path=csv_file,
                    label=label,
                    label_idx=label_idx,
                    participant=participant
                )
                files.append(eeg_file)
        
        logger.info(f"Total files discovered: {len(files)}")
        
        # Log class distribution
        label_counts = {}
        for file in files:
            label_counts[file.label] = label_counts.get(file.label, 0) + 1
        
        logger.info("Class distribution:")
        for label in sorted(label_counts.keys()):
            logger.info(f"  {label}: {label_counts[label]} files")
        
        self.files = files
        return files
    
    def _extract_participant(self, filename: str) -> Optional[str]:
        """
        Extract participant ID from filename.
        
        Typical format: "par.X word_EPOCX_..."
        
        Parameters
        ----------
        filename : str
            File name
            
        Returns
        -------
        str or None
            Participant ID if found
        """
        parts = filename.split()
        if parts and parts[0].startswith('par.'):
            return parts[0]  # e.g., "par.1", "par.10"
        return None
    
    def load_raw_file(self, eeg_file: EEGFile) -> Tuple[np.ndarray, float, Dict]:
        """
        Load raw EEG data from a single file.
        
        Parameters
        ----------
        eeg_file : EEGFile
            File metadata
            
        Returns
        -------
        eeg_data : np.ndarray
            EEG data of shape [n_samples, 14]
        sampling_rate : float
            Computed sampling rate in Hz
        metadata : dict
            Additional metadata (timestamps, quality, motion)
        """
        try:
            # Read CSV, skipping first row (often contains units or metadata)
            df = pd.read_csv(eeg_file.path, skiprows=1)
        except Exception as e:
            logger.error(f"Failed to load {eeg_file.path}: {e}")
            raise
        
        # Extract EEG channels
        available_eeg = [ch for ch in self.config.eeg_channels if ch in df.columns]
        
        if len(available_eeg) != 14:
            logger.warning(
                f"File {eeg_file.path.name} has {len(available_eeg)}/14 EEG channels"
            )
        
        eeg_data = df[available_eeg].values  # Shape: [n_samples, n_channels]
        
        # Ensure we have 14 channels (pad with zeros if missing)
        if eeg_data.shape[1] < 14:
            padded = np.zeros((eeg_data.shape[0], 14))
            padded[:, :eeg_data.shape[1]] = eeg_data
            eeg_data = padded
            logger.warning(f"Padded {eeg_file.path.name} to 14 channels")
        
        # Compute sampling rate from timestamps
        if 'Timestamp' in df.columns:
            timestamps = df['Timestamp'].values
            time_diffs = np.diff(timestamps)
            # Remove outliers for robust estimation
            time_diffs = time_diffs[time_diffs > 0]
            median_diff = np.median(time_diffs)
            sampling_rate = 1.0 / median_diff if median_diff > 0 else self.config.expected_fs
        else:
            logger.warning(f"No timestamp column in {eeg_file.path.name}, using expected fs")
            sampling_rate = self.config.expected_fs
        
        # Extract metadata
        metadata = {
            'timestamps': df['Timestamp'].values if 'Timestamp' in df.columns else None,
            'contact_quality': None,
            'motion': None
        }
        
        # Contact quality
        available_cq = [ch for ch in self.config.cq_channels if ch in df.columns]
        if available_cq:
            metadata['contact_quality'] = df[available_cq].values
        
        # Motion sensors
        available_motion = [ch for ch in self.config.motion_channels if ch in df.columns]
        if available_motion:
            metadata['motion'] = df[available_motion].values
        
        logger.debug(
            f"Loaded {eeg_file.path.name}: "
            f"shape={eeg_data.shape}, fs={sampling_rate:.2f}Hz"
        )
        
        return eeg_data, sampling_rate, metadata
    
    def get_file_groups(self) -> Dict[str, List[EEGFile]]:
        """
        Group files by participant for GroupKFold.
        
        Returns
        -------
        Dict[str, List[EEGFile]]
            Dictionary mapping participant ID to list of files
        """
        groups = {}
        for file in self.files:
            group_key = file.participant if file.participant else file.path.stem
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(file)
        
        logger.info(f"Files grouped into {len(groups)} groups")
        return groups
    
    def get_labels(self) -> np.ndarray:
        """Get array of labels for all files."""
        return np.array([f.label_idx for f in self.files])
    
    def get_file_paths(self) -> List[Path]:
        """Get list of all file paths."""
        return [f.path for f in self.files]
