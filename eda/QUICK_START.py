"""
==========================================
QUICK START GUIDE
ArEEG_Words: EEG Classification Pipeline
==========================================

STEP-BY-STEP EXECUTION GUIDE
"""

# ===========================
# STEP 1: VERIFY SETUP
# ===========================

"""
Run the test script to verify installation:

    python test_setup.py

This checks:
- All Python dependencies are installed
- All pipeline modules can be imported
- Data directory structure is correct
- Basic configuration works

If any checks fail, follow the instructions shown.
"""


# ===========================
# STEP 2: BASIC EXECUTION
# ===========================

"""
Run the pipeline with default settings:

    python main.py --data_dir "../data" --output_dir "output"

This will:
- Load all 359 EEG files from 16 classes
- Preprocess with notch (50Hz) + bandpass (1-40Hz)
- Create 1-second epochs with 50% overlap
- Run 5-fold cross-validation
- Train and evaluate models
- Save results to output/

Expected runtime: 15-30 minutes (depends on CPU)
"""


# ===========================
# STEP 3: CUSTOMIZE PARAMETERS
# ===========================

"""
Adjust parameters for your experiment:

    python main.py \
        --data_dir "../data" \
        --output_dir "results_exp1" \
        --n_splits 10 \
        --window_sec 1.5 \
        --overlap 0.6 \
        --seed 123

Key parameters:
- n_splits: Number of CV folds (5 or 10 recommended)
- window_sec: Epoch duration (0.5-2.0 seconds)
- overlap: Overlap fraction (0.0-0.8)
- seed: Random seed for reproducibility

Optional flags:
- --no_cache: Disable file caching
- --no_quality_weighting: Disable quality-based weighting
- --log_level DEBUG: Show detailed debug messages
"""


# ===========================
# STEP 4: ANALYZE RESULTS
# ===========================

"""
Check the output directory for results:

output/
├── pipeline.log                          # Detailed execution log
├── aggregated_results.json               # Final metrics (JSON)
├── fold_results.csv                      # Per-fold metrics (CSV)
├── per_class_metrics.csv                 # Per-class performance
├── confusion_matrix_aggregated.png       # Visual confusion matrix
├── confusion_matrices/                   # Per-fold matrices
│   ├── confusion_matrix_fold_0.png
│   └── ...
└── models/                               # Saved models per fold
    ├── fold_0/
    │   ├── csp.pkl                       # CSP model
    │   ├── nca.pkl                       # NCA selector
    │   ├── ensemble.pkl                  # Stacking ensemble
    │   └── optimal_band.txt              # Selected frequency band
    └── ...

Key metrics to check:
- Accuracy: Overall correctness
- Macro F1: Average F1 across classes (handles imbalance)
- Per-class F1: Identify strong/weak classes
- Confusion matrix: Error patterns
"""


# ===========================
# STEP 5: ADVANCED USAGE
# ===========================

"""
For advanced customization, edit config.py:

class Config:
    # Preprocessing
    notch_freq = 50.0           # Mains frequency
    global_lowcut = 1.0         # Bandpass low
    global_highcut = 40.0       # Bandpass high
    
    # MI band selection
    mi_band_width = 3.0         # Band width (Hz)
    mi_band_step = 2.0          # Band step (Hz)
    mi_freq_range = (4.0, 40.0) # Search range
    
    # CSP
    csp_n_components_per_class = 4  # Features per OVR model
    csp_reg = None              # Regularization (or 0.1)
    
    # NCA
    nca_n_components = 25       # Number of features
    
    # Stacking
    base_classifiers = [        # Base learners
        'lda', 'knn', 'linear_svm', 
        'rbf_svm', 'naive_bayes', 'random_forest'
    ]

Save changes and rerun main.py.
"""


# ===========================
# STEP 6: TROUBLESHOOTING
# ===========================

"""
Common issues and solutions:

1. ImportError: No module named '...'
   → Run: pip install -r requirements.txt

2. FileNotFoundError: Data directory not found
   → Check --data_dir path matches your directory structure
   → Default expects: كلمات/كلمات/

3. ValueError: All samples have the same class
   → Increase n_splits or check data distribution
   → Try --n_splits 5

4. LinAlgError: Eigenvalue decomposition failed
   → Enable CSP regularization in config.py:
     csp_reg = 0.1

5. Low accuracy (<30%)
   → Check signal quality with test_setup.py
   → Try different window_sec (0.5, 1.0, 1.5)
   → Disable quality filtering: --no_quality_weighting

6. Pipeline is too slow
   → Enable caching (default)
   → Reduce n_splits to 5
   → Check n_jobs=-1 in config.py

7. Cache files are huge
   → Clear cache folder: rm -rf cache/
   → Disable caching: --no_cache
"""


# ===========================
# STEP 7: INTERPRET RESULTS
# ===========================

"""
Understanding the output metrics:

ACCURACY (60-70% is good for 16 classes)
- Random baseline: 1/16 = 6.25%
- Simple baseline: ~20-30%
- Good performance: >60%

MACRO F1 (handles class imbalance)
- Average F1 across all classes
- More robust than accuracy when classes are imbalanced

PER-CLASS METRICS
- Look for patterns:
  • High F1: Clear, consistent patterns
  • Low F1: Noisy data or rare class
  • High precision, low recall: Model too conservative
  • Low precision, high recall: Model too aggressive

CONFUSION MATRIX
- Diagonal = correct predictions
- Off-diagonal = confusion patterns
- Look for systematic confusions between similar words

SELECTED FREQUENCY BAND (optimal_band.txt)
- Shows which frequencies are most informative
- Typical EEG bands:
  • Delta: 1-4 Hz
  • Theta: 4-8 Hz
  • Alpha: 8-13 Hz
  • Beta: 13-30 Hz
  • Gamma: >30 Hz
- Winner paper typically selects 8-30Hz for imagined speech
"""


# ===========================
# STEP 8: NEXT STEPS
# ===========================

"""
Improve performance:

1. Hyperparameter tuning
   - Grid search over window_sec, overlap
   - Tune NCA n_components
   - Adjust CSP components

2. Artifact removal
   - Implement ICA for eye blinks
   - Detect and remove bad epochs

3. Feature engineering
   - Add wavelet features
   - Time-frequency features (STFT, CWT)
   - Connectivity features (coherence)

4. Advanced models
   - Deep learning (CNN, LSTM, EEGNet)
   - Transfer learning from larger datasets
   - Ensemble with deep + classical

5. Data augmentation
   - Time jitter
   - Amplitude scaling
   - Synthetic minority oversampling (SMOTE)

6. Subject-specific models
   - Train/test split by participant
   - Fine-tune on target subject
"""


# ===========================
# EXAMPLE WORKFLOW
# ===========================

"""
Complete workflow example:

# 1. Verify setup
python test_setup.py

# 2. Run baseline experiment
python main.py --data_dir "كلمات/كلمات" --output_dir "exp_baseline"

# 3. Check results
cat exp_baseline/aggregated_results.json
open exp_baseline/confusion_matrix_aggregated.png

# 4. Run with different parameters
python main.py \
    --data_dir "كلمات/كلمات" \
    --output_dir "exp_longer_window" \
    --window_sec 1.5 \
    --n_splits 10

# 5. Compare results
python -c "
import json
with open('exp_baseline/aggregated_results.json') as f:
    baseline = json.load(f)
with open('exp_longer_window/aggregated_results.json') as f:
    exp = json.load(f)
print(f'Baseline:       {baseline[\"accuracy_mean\"]:.4f}')
print(f'Longer window:  {exp[\"accuracy_mean\"]:.4f}')
"

# 6. Use best model for prediction
# (Load models from models/fold_0/*.pkl and apply to new data)
"""

print(__doc__)
