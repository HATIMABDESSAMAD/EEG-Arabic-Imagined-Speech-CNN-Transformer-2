"""
==========================================
Configuration Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Dataclass-based configuration for the entire pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Main configuration for ArEEG_Words EEG classification pipeline."""
    
    # ==================== DATA PATHS ====================
    data_dir: Path = Path("../data")
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    
    # ==================== EEG CHANNELS ====================
    eeg_channels: List[str] = field(default_factory=lambda: [
        'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7',
        'EEG.P7', 'EEG.O1', 'EEG.O2', 'EEG.P8', 'EEG.T8',
        'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
    ])
    
    cq_channels: List[str] = field(default_factory=lambda: [
        'CQ.AF3', 'CQ.F7', 'CQ.F3', 'CQ.FC5', 'CQ.T7',
        'CQ.P7', 'CQ.O1', 'CQ.O2', 'CQ.P8', 'CQ.T8',
        'CQ.FC6', 'CQ.F4', 'CQ.F8', 'CQ.AF4'
    ])
    
    motion_channels: List[str] = field(default_factory=lambda: [
        'MOT.Q0', 'MOT.Q1', 'MOT.Q2', 'MOT.Q3',
        'MOT.AccX', 'MOT.AccY', 'MOT.AccZ',
        'MOT.MagX', 'MOT.MagY', 'MOT.MagZ'
    ])
    
    # ==================== PREPROCESSING ====================
    # Expected sampling rate (will be verified from timestamps)
    expected_fs: float = 128.0
    
    # Notch filter for mains interference (Morocco: 50Hz)
    notch_freq: float = 50.0
    notch_quality: float = 30.0
    
    # Global bandpass filter (stable range for EPOCX)
    global_lowcut: float = 1.0
    global_highcut: float = 40.0
    filter_order: int = 3
    
    # ==================== EPOCHING ====================
    window_sec: float = 1.0  # Window duration in seconds
    overlap: float = 0.5  # Overlap fraction (0-1)
    
    # ==================== MI BAND SELECTION ====================
    # Filter bank parameters
    mi_band_width: float = 3.0  # Hz
    mi_band_step: float = 2.0  # Hz
    mi_freq_range: tuple = (4.0, 40.0)  # Start and end frequencies
    
    # Band selection strategy
    mi_selection_method: str = "above_mean"  # "above_mean" or "top_k"
    mi_top_k: int = 5  # If using top_k method
    
    # ==================== CSP ====================
    csp_n_components_per_class: int = 4  # 2 pairs (largest + smallest eigenvalues)
    csp_reg: Optional[float] = None  # Regularization (None or float like 0.1)
    
    # ==================== NCA ====================
    nca_n_components: int = 25  # Number of features to keep
    nca_max_iter: int = 200
    nca_tol: float = 1e-5
    
    # ==================== CLASSIFICATION ====================
    # Cross-validation
    n_splits: int = 5  # Number of CV folds
    random_state: int = 42
    
    # Stacking ensemble configuration
    use_stacking: bool = True
    stacking_cv: int = 5  # Internal CV for stacking
    
    # Base classifiers to use
    base_classifiers: List[str] = field(default_factory=lambda: [
        'lda', 'knn', 'linear_svm', 'rbf_svm', 'naive_bayes', 'random_forest'
    ])
    
    # ==================== QUALITY WEIGHTING ====================
    use_quality_weighting: bool = True
    quality_threshold: float = 0.3  # Minimum quality score to keep epoch
    
    # ==================== PERFORMANCE ====================
    use_cache: bool = True  # Cache preprocessed files
    n_jobs: int = -1  # Number of parallel jobs (-1 = all cores)
    
    # ==================== LOGGING ====================
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    save_models: bool = True  # Save trained models per fold
    save_plots: bool = True  # Save confusion matrices and other plots
    
    def __post_init__(self):
        """Validate configuration and create directories."""
        # Convert string paths to Path objects
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate parameters
        assert 0 <= self.overlap < 1, "Overlap must be in [0, 1)"
        assert self.window_sec > 0, "Window size must be positive"
        assert self.n_splits >= 2, "Need at least 2 folds for CV"
        assert len(self.eeg_channels) == 14, "Must have exactly 14 EEG channels"
        
    def get_stride_sec(self) -> float:
        """Calculate stride in seconds based on window and overlap."""
        return self.window_sec * (1 - self.overlap)
    
    def get_mi_bands(self) -> List[tuple]:
        """Generate list of MI frequency bands."""
        bands = []
        f_low = self.mi_freq_range[0]
        f_high = self.mi_freq_range[1]
        
        current_low = f_low
        while current_low + self.mi_band_width <= f_high:
            current_high = current_low + self.mi_band_width
            bands.append((current_low, current_high))
            current_low += self.mi_band_step
        
        return bands
