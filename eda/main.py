"""
==========================================
Main Pipeline Entry Point
ArEEG_Words: EEG Classification Pipeline
==========================================

COMPLETE EEG CLASSIFICATION PIPELINE
Inspired by "Imagined Speech Decoding From EEG: The Winner of 3rd Iranian BCI Competition"
Adapted for ArEEG_Words dataset (16 Arabic words, 14 EEG channels, Emotiv EPOCX)

USAGE:
    python main.py --data_dir "كلمات/كلمات" --output_dir "output" --n_splits 5

PIPELINE STAGES:
    1. Data Loading & Preprocessing (notch 50Hz, bandpass 1-40Hz)
    2. Epoching (sliding windows with overlap, no leakage by file)
    3. MI-based Frequency Band Selection (train only, per fold)
    4. CSP One-vs-Rest Feature Extraction (16 OVR models)
    5. NCA Feature Selection (train only)
    6. Stacking Ensemble Classification (LDA, KNN, SVM, NB, RF + LogReg meta)
    7. Evaluation (accuracy, F1, confusion matrix)

FEATURES:
    - No data leakage (file-level CV split)
    - Quality weighting (contact quality + motion)
    - Caching for speed
    - Full reproducibility
    - Comprehensive metrics

REQUIREMENTS:
    numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, joblib
"""

import argparse
import logging
import time
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import StratifiedKFold, GroupKFold

from config import Config
from utils import setup_logging, set_seed, print_section, format_time
from io_data import DatasetLoader
from preprocess import SignalPreprocessor
from mi_band_selection import MIBandSelector
from csp_ovr import CSPOneVsRest
from nca_selection import NCAFeatureSelector
from stacking_model import StackingEnsemble
from evaluation import Evaluator


logger = logging.getLogger(__name__)


class EEGPipeline:
    """Complete EEG classification pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = DatasetLoader(config)
        self.preprocessor = SignalPreprocessor(config)
        self.mi_selector = MIBandSelector(config)
        
        # Discover files
        self.files = self.loader.discover_files()
        self.n_classes = len(self.loader.label_to_idx)
        
        # Create evaluator
        self.evaluator = Evaluator(
            config,
            label_names=list(self.loader.idx_to_label.values())
        )
        
    def run(self):
        """Execute the complete pipeline."""
        start_time = time.time()
        
        print_section("ArEEG_Words EEG CLASSIFICATION PIPELINE")
        logger.info(f"Configuration: {self.n_classes} classes, {len(self.files)} files")
        logger.info(f"CV: {self.config.n_splits} folds")
        logger.info(f"Output directory: {self.config.output_dir}")
        
        # Preprocess all files (global filtering + epoching)
        print_section("STAGE 1: PREPROCESSING")
        all_epochs, all_quality, all_labels, all_fs = self.preprocessor.preprocess_all_files(
            self.files, self.loader
        )
        
        # Verify sampling rates are consistent
        unique_fs = np.unique(all_fs)
        if len(unique_fs) > 1:
            logger.warning(f"Multiple sampling rates detected: {unique_fs}")
        fs = np.median(all_fs)
        logger.info(f"Using median sampling rate: {fs:.2f} Hz")
        
        # Setup cross-validation
        print_section("STAGE 2: CROSS-VALIDATION SETUP")
        file_indices = np.arange(len(all_epochs))
        file_labels = np.array(all_labels)
        
        # Try to use GroupKFold if participants are available
        groups = None
        if self.files[0].participant is not None:
            groups = np.array([f.participant for f in self.files])
            logger.info("Using GroupKFold (by participant)")
            cv = GroupKFold(n_splits=self.config.n_splits)
            splits = list(cv.split(file_indices, file_labels, groups))
        else:
            logger.info("Using StratifiedKFold (by label)")
            cv = StratifiedKFold(
                n_splits=self.config.n_splits,
                shuffle=True,
                random_state=self.config.random_state
            )
            splits = list(cv.split(file_indices, file_labels))
        
        # Run cross-validation
        fold_results = []
        
        for fold_idx, (train_file_idx, test_file_idx) in enumerate(splits):
            logger.info("")
            print_section(f"FOLD {fold_idx + 1}/{self.config.n_splits}", char="-")
            
            fold_result = self._run_fold(
                fold_idx,
                train_file_idx,
                test_file_idx,
                all_epochs,
                all_quality,
                all_labels,
                fs
            )
            
            fold_results.append(fold_result)
            
            # Log fold results
            logger.info(f"Fold {fold_idx + 1} Results:")
            logger.info(f"  Accuracy:  {fold_result['accuracy']:.4f}")
            logger.info(f"  Macro F1:  {fold_result['macro_f1']:.4f}")
        
        # Aggregate results
        print_section("FINAL RESULTS")
        aggregated = self.evaluator.aggregate_results(fold_results)
        self.evaluator.print_results(aggregated)
        
        # Save results
        self.evaluator.save_results(fold_results, aggregated, self.config.output_dir)
        
        if self.config.save_plots:
            self.evaluator.save_fold_confusion_matrices(fold_results, self.config.output_dir)
            self.evaluator.save_aggregated_confusion_matrix(aggregated, self.config.output_dir)
        
        # Total time
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"PIPELINE COMPLETED IN {format_time(total_time)}")
        logger.info(f"{'='*80}")
    
    def _run_fold(
        self,
        fold_idx: int,
        train_file_idx: np.ndarray,
        test_file_idx: np.ndarray,
        all_epochs: list,
        all_quality: list,
        all_labels: list,
        fs: float
    ) -> dict:
        """
        Run pipeline for a single fold.
        
        Returns
        -------
        dict
            Fold results including metrics
        """
        # Separate train/test files
        train_epochs_list = [all_epochs[i] for i in train_file_idx]
        train_quality_list = [all_quality[i] for i in train_file_idx]
        train_labels_file = [all_labels[i] for i in train_file_idx]
        
        test_epochs_list = [all_epochs[i] for i in test_file_idx]
        test_quality_list = [all_quality[i] for i in test_file_idx]
        test_labels_file = [all_labels[i] for i in test_file_idx]
        
        # Flatten epochs (files -> epochs)
        X_train_epochs, y_train, w_train = self._flatten_epochs(
            train_epochs_list, train_labels_file, train_quality_list
        )
        X_test_epochs, y_test, w_test = self._flatten_epochs(
            test_epochs_list, test_labels_file, test_quality_list
        )
        
        logger.info(f"Train: {len(X_train_epochs)} epochs from {len(train_file_idx)} files")
        logger.info(f"Test:  {len(X_test_epochs)} epochs from {len(test_file_idx)} files")
        
        # STAGE 3: MI-based Band Selection (TRAIN ONLY)
        logger.info("Stage 3: MI-based frequency band selection...")
        optimal_band = self.mi_selector.select_optimal_band(X_train_epochs, y_train, fs)
        
        # Apply optimal band filter to both train and test
        X_train_filtered = self._apply_band_filter(X_train_epochs, fs, optimal_band)
        X_test_filtered = self._apply_band_filter(X_test_epochs, fs, optimal_band)
        
        # STAGE 4: CSP One-vs-Rest Feature Extraction
        logger.info("Stage 4: CSP One-vs-Rest feature extraction...")
        csp = CSPOneVsRest(self.config, self.n_classes)
        X_train_csp = csp.fit_transform(X_train_filtered, y_train)
        X_test_csp = csp.transform(X_test_filtered)
        logger.info(f"CSP features: {X_train_csp.shape}")
        
        # STAGE 5: NCA Feature Selection (TRAIN ONLY)
        logger.info("Stage 5: NCA feature selection...")
        nca = NCAFeatureSelector(self.config, mode="selection")
        X_train_nca = nca.fit_transform(X_train_csp, y_train)
        X_test_nca = nca.transform(X_test_csp)
        logger.info(f"NCA features: {X_train_nca.shape}")
        
        # STAGE 6: Stacking Ensemble Classification
        logger.info("Stage 6: Training stacking ensemble...")
        ensemble = StackingEnsemble(self.config, self.n_classes)
        
        if self.config.use_quality_weighting and w_train is not None:
            ensemble.fit(X_train_nca, y_train, sample_weight=w_train)
        else:
            ensemble.fit(X_train_nca, y_train)
        
        # STAGE 7: Prediction & Evaluation
        logger.info("Stage 7: Prediction and evaluation...")
        y_pred = ensemble.predict(X_test_nca)
        
        # Evaluate
        fold_result = self.evaluator.evaluate_fold(y_test, y_pred, fold_idx)
        
        # Save models if requested
        if self.config.save_models:
            model_dir = self.config.output_dir / "models" / f"fold_{fold_idx}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(csp, model_dir / "csp.pkl")
            joblib.dump(nca, model_dir / "nca.pkl")
            joblib.dump(ensemble, model_dir / "ensemble.pkl")
            
            # Save optimal band
            with open(model_dir / "optimal_band.txt", 'w') as f:
                f.write(f"{optimal_band[0]:.2f},{optimal_band[1]:.2f}\n")
        
        return fold_result
    
    def _flatten_epochs(
        self,
        epochs_list: list,
        labels_file: list,
        quality_list: list
    ) -> tuple:
        """
        Flatten list of epoch arrays into single array with labels and weights.
        
        Parameters
        ----------
        epochs_list : list of np.ndarray
            List of epoch arrays, each shape [n_epochs_file, n_channels, n_samples]
        labels_file : list of int
            File-level labels
        quality_list : list of np.ndarray or None
            Quality scores per file
            
        Returns
        -------
        epochs : np.ndarray
            Flattened epochs [total_epochs, n_channels, n_samples]
        labels : np.ndarray
            Epoch-level labels
        weights : np.ndarray or None
            Epoch-level quality weights
        """
        all_epochs = []
        all_labels = []
        all_weights = []
        
        for file_epochs, file_label, file_quality in zip(epochs_list, labels_file, quality_list):
            n_epochs_file = file_epochs.shape[0]
            
            all_epochs.append(file_epochs)
            all_labels.extend([file_label] * n_epochs_file)
            
            if file_quality is not None:
                all_weights.extend(file_quality)
            else:
                all_weights.extend([1.0] * n_epochs_file)
        
        epochs = np.concatenate(all_epochs, axis=0)
        labels = np.array(all_labels)
        weights = np.array(all_weights) if any(q is not None for q in quality_list) else None
        
        # Filter low-quality epochs if threshold is set
        if weights is not None and self.config.use_quality_weighting:
            mask = weights >= self.config.quality_threshold
            logger.info(
                f"Quality filtering: keeping {np.sum(mask)}/{len(mask)} epochs "
                f"(threshold={self.config.quality_threshold})"
            )
            epochs = epochs[mask]
            labels = labels[mask]
            weights = weights[mask]
        
        return epochs, labels, weights
    
    def _apply_band_filter(
        self,
        epochs: np.ndarray,
        fs: float,
        band: tuple
    ) -> np.ndarray:
        """
        Apply bandpass filter to epochs.
        
        Parameters
        ----------
        epochs : np.ndarray
            Epochs [n_epochs, n_channels, n_samples]
        fs : float
            Sampling rate
        band : tuple
            (low, high) frequency band
            
        Returns
        -------
        np.ndarray
            Filtered epochs
        """
        n_epochs, n_channels, n_samples = epochs.shape
        filtered = np.zeros_like(epochs)
        
        for i in range(n_epochs):
            # Reshape to [n_samples, n_channels] for preprocessor
            epoch_reshaped = epochs[i].T
            # Filter
            epoch_filtered = self.preprocessor.apply_bandpass_filter(
                epoch_reshaped, fs, lowcut=band[0], highcut=band[1]
            )
            # Reshape back to [n_channels, n_samples]
            filtered[i] = epoch_filtered.T
        
        return filtered


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ArEEG_Words EEG Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default="كلمات/كلمات",
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default="output",
        help="Output directory for results"
    )
    
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        '--window_sec',
        type=float,
        default=1.0,
        help="Epoch window size in seconds"
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.5,
        help="Epoch overlap fraction (0-1)"
    )
    
    parser.add_argument(
        '--no_cache',
        action='store_true',
        help="Disable caching"
    )
    
    parser.add_argument(
        '--no_quality_weighting',
        action='store_true',
        help="Disable quality weighting"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        '--log_level',
        type=str,
        default="INFO",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Create configuration
    config = Config(
        data_dir=args.data_dir,
        output_dir=Path(args.output_dir),
        n_splits=args.n_splits,
        window_sec=args.window_sec,
        overlap=args.overlap,
        use_cache=not args.no_cache,
        use_quality_weighting=not args.no_quality_weighting,
        random_state=args.seed,
        log_level=args.log_level
    )
    
    # Setup logging
    log_file = config.output_dir / "pipeline.log"
    setup_logging(config.log_level, log_file)
    
    # Set random seed
    set_seed(config.random_state)
    
    # Run pipeline
    pipeline = EEGPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
