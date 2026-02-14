"""
Test script to verify pipeline installation and basic functionality.
Run this before executing the full pipeline.
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 60)
    print("CHECKING IMPORTS")
    print("=" * 60)
    
    required_modules = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('joblib', 'Joblib'),
    ]
    
    missing = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name:15s} - OK")
        except ImportError:
            print(f"✗ {display_name:15s} - MISSING")
            missing.append(display_name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!\n")
    return True


def check_pipeline_modules():
    """Check if all pipeline modules can be imported."""
    print("=" * 60)
    print("CHECKING PIPELINE MODULES")
    print("=" * 60)
    
    modules = [
        'config',
        'utils',
        'io_data',
        'preprocess',
        'mi_band_selection',
        'csp_ovr',
        'nca_selection',
        'stacking_model',
        'evaluation',
        'main',
    ]
    
    errors = []
    
    for module_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:20s} - OK")
        except Exception as e:
            print(f"✗ {module_name:20s} - ERROR")
            errors.append((module_name, str(e)))
    
    if errors:
        print("\n⚠️  Module import errors:")
        for module_name, error in errors:
            print(f"  {module_name}: {error}")
        return False
    
    print("\n✓ All pipeline modules OK!\n")
    return True


def check_data_directory():
    """Check if data directory exists."""
    print("=" * 60)
    print("CHECKING DATA DIRECTORY")
    print("=" * 60)
    
    data_paths = [
        Path("../data"),
        Path("data"),
    ]
    
    found = None
    for data_path in data_paths:
        if data_path.exists() and data_path.is_dir():
            found = data_path
            break
    
    if found:
        # Count class folders
        class_folders = [f for f in found.iterdir() if f.is_dir()]
        print(f"✓ Data directory found: {found}")
        print(f"  Classes detected: {len(class_folders)}")
        
        # Count total CSV files
        total_files = sum(len(list(folder.glob("*.csv"))) for folder in class_folders)
        print(f"  Total CSV files: {total_files}")
        
        if len(class_folders) < 2:
            print("⚠️  Warning: Less than 2 classes found. Check directory structure.")
            return False
        
        print("\n✓ Data directory OK!\n")
        return True
    else:
        print("✗ Data directory not found!")
        print("  Expected: '../data' or 'data'")
        print("  Provide correct path with --data_dir when running main.py")
        return False


def test_basic_pipeline():
    """Test basic pipeline instantiation."""
    print("=" * 60)
    print("TESTING BASIC PIPELINE")
    print("=" * 60)
    
    try:
        from config import Config
        config = Config()
        print(f"✓ Config created")
        print(f"  EEG channels: {len(config.eeg_channels)}")
        print(f"  MI bands: {len(config.get_mi_bands())}")
        print(f"  Output dir: {config.output_dir}")
        
        from utils import set_seed
        set_seed(42)
        print(f"✓ Random seed set")
        
        print("\n✓ Basic pipeline test passed!\n")
        return True
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print(" " * 15 + "PIPELINE VERIFICATION")
    print("=" * 60 + "\n")
    
    checks = [
        check_imports(),
        check_pipeline_modules(),
        check_data_directory(),
        test_basic_pipeline(),
    ]
    
    print("=" * 60)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nYou can now run the pipeline:")
        print('  python main.py --data_dir "../data"')
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before running the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
