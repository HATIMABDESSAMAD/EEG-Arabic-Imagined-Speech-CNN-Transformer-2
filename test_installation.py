"""
Quick test script to verify installation and data loading.
Run this before training to ensure everything is set up correctly.
"""

import sys
import subprocess
from pathlib import Path

def check_imports():
    """Check if all required packages are installed."""
    print("Checking required packages...")
    
    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'tensorflow': 'tensorflow'
    }
    
    missing = []
    
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All packages installed")
        return True


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"  ✓ Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"    - {gpu.name}")
            return True
        else:
            print("  ⚠ No GPU found (will use CPU)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False


def check_data(data_root='./data'):
    """Check if data directory exists and has files."""
    print(f"\nChecking data directory: {data_root}")
    
    data_path = Path(data_root)
    
    if not data_path.exists():
        print(f"  ✗ Directory not found: {data_root}")
        return False
    
    # Count subdirectories (word classes)
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"  ✗ No subdirectories found in {data_root}")
        return False
    
    print(f"  ✓ Found {len(subdirs)} word classes")
    
    # Count CSV files
    total_files = 0
    for subdir in subdirs:
        csv_files = list(subdir.glob('*.csv'))
        total_files += len(csv_files)
        print(f"    - {subdir.name}: {len(csv_files)} files")
    
    if total_files == 0:
        print("  ✗ No CSV files found")
        return False
    
    print(f"  ✓ Total: {total_files} CSV files")
    return True


def test_data_loading(data_root='./data'):
    """Test loading a single CSV file."""
    print("\nTesting data loading...")
    
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        data_path = Path(data_root)
        
        # Find first CSV file
        csv_file = None
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                csv_files = list(subdir.glob('*.csv'))
                if csv_files:
                    csv_file = csv_files[0]
                    break
        
        if not csv_file:
            print("  ✗ No CSV file found to test")
            return False
        
        print(f"  Testing with: {csv_file.name}")
        
        # Load CSV (skip metadata row)
        df = pd.read_csv(csv_file, skiprows=1)
        
        # Check for EEG channels
        eeg_channels = [
            'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
            'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
        ]
        
        missing_channels = [ch for ch in eeg_channels if ch not in df.columns]
        
        if missing_channels:
            print(f"  ✗ Missing channels: {missing_channels}")
            return False
        
        eeg_data = df[eeg_channels].values
        
        print(f"  ✓ Loaded EEG data: shape {eeg_data.shape}")
        print(f"    - Channels: {eeg_data.shape[1]}")
        print(f"    - Samples: {eeg_data.shape[0]}")
        print(f"    - Duration: {eeg_data.shape[0] / 128:.2f} seconds (at 128 Hz)")
        print(f"    - Value range: [{np.min(eeg_data):.2f}, {np.max(eeg_data):.2f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("ArEEG_Words - Installation Test")
    print("="*70)
    
    # Run checks
    checks = [
        ("Package imports", lambda: check_imports()),
        ("GPU availability", lambda: check_gpu()),
        ("Data directory", lambda: check_data()),
        ("Data loading", lambda: test_data_loading())
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ All checks passed! Ready to train.")
        print("\nRun training with:")
        print("  python train_areeg_words_cnn_transformer.py --dry_run")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
