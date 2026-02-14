"""
==========================================
Utility Functions
ArEEG_Words: EEG Classification Pipeline
==========================================
Seed setting, logging, and helper functions.
"""

import logging
import random
import numpy as np
from pathlib import Path
from typing import Optional
import hashlib


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : Path, optional
        Path to log file. If None, logs only to console.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler()]
    
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='w'))
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers,
        force=True
    )
    
    # Suppress noisy libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Parameters
    ----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Try to set sklearn seed
    try:
        import sklearn
        sklearn.utils.check_random_state(seed)
    except ImportError:
        pass
    
    logging.info(f"Random seed set to {seed}")


def get_file_hash(file_path: Path) -> str:
    """
    Compute hash of file for caching purposes.
    
    Parameters
    ----------
    file_path : Path
        Path to file
        
    Returns
    -------
    str
        MD5 hash of file
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cache_path(file_path: Path, cache_dir: Path, suffix: str = "") -> Path:
    """
    Generate cache file path for a given input file.
    
    Parameters
    ----------
    file_path : Path
        Original file path
    cache_dir : Path
        Cache directory
    suffix : str
        Additional suffix for cache filename
        
    Returns
    -------
    Path
        Cache file path
    """
    # Create a unique cache filename based on original file
    file_hash = get_file_hash(file_path)
    cache_name = f"{file_path.stem}_{file_hash[:8]}{suffix}.npz"
    return cache_dir / cache_name


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_section(title: str, char: str = "=", width: int = 80):
    """
    Print a formatted section header for logging.
    
    Parameters
    ----------
    title : str
        Section title
    char : str
        Character for border
    width : int
        Width of the section
    """
    logger = logging.getLogger(__name__)
    logger.info(char * width)
    logger.info(title.center(width))
    logger.info(char * width)


def print_subsection(title: str, char: str = "-", width: int = 80):
    """
    Print a formatted subsection header for logging.
    
    Parameters
    ----------
    title : str
        Subsection title
    char : str
        Character for border
    width : int
        Width of the subsection
    """
    logger = logging.getLogger(__name__)
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


class ProgressTracker:
    """Simple progress tracker for loops."""
    
    def __init__(self, total: int, desc: str = "Progress"):
        self.total = total
        self.desc = desc
        self.current = 0
        self.logger = logging.getLogger(__name__)
        
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        pct = (self.current / self.total) * 100
        self.logger.info(f"{self.desc}: {self.current}/{self.total} ({pct:.1f}%)")
        
    def finish(self):
        """Mark progress as complete."""
        self.logger.info(f"{self.desc}: Complete!")
