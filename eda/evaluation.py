"""
==========================================
Evaluation Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Evaluation metrics, confusion matrix, and result reporting.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import json

from config import Config


logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate classification results and generate reports."""
    
    def __init__(self, config: Config, label_names: list):
        self.config = config
        self.label_names = label_names
        self.n_classes = len(label_names)
        
    def evaluate_fold(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fold_idx: int
    ) -> dict:
        """
        Evaluate predictions for a single fold.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        fold_idx : int
            Fold index
            
        Returns
        -------
        dict
            Dictionary of metrics
        """
        metrics = {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro')
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, label in enumerate(self.label_names):
            metrics[f'precision_{label}'] = precision_per_class[i]
            metrics[f'recall_{label}'] = recall_per_class[i]
            metrics[f'f1_{label}'] = f1_per_class[i]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def aggregate_results(self, fold_results: list) -> dict:
        """
        Aggregate results across folds.
        
        Parameters
        ----------
        fold_results : list of dict
            Results from each fold
            
        Returns
        -------
        dict
            Aggregated metrics
        """
        # Extract numeric metrics
        metric_names = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']
        
        aggregated = {}
        
        for metric in metric_names:
            values = [fold[metric] for fold in fold_results]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
        
        # Per-class metrics
        for label in self.label_names:
            for metric in ['precision', 'recall', 'f1']:
                key = f'{metric}_{label}'
                values = [fold[key] for fold in fold_results if key in fold]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)
        
        # Aggregate confusion matrix
        cms = [fold['confusion_matrix'] for fold in fold_results]
        aggregated['confusion_matrix_mean'] = np.mean(cms, axis=0)
        aggregated['confusion_matrix_sum'] = np.sum(cms, axis=0)
        
        return aggregated
    
    def print_results(self, aggregated: dict):
        """
        Print aggregated results to log.
        
        Parameters
        ----------
        aggregated : dict
            Aggregated metrics
        """
        logger.info("=" * 80)
        logger.info("FINAL RESULTS (mean ± std across folds)")
        logger.info("=" * 80)
        
        logger.info(f"Accuracy:          {aggregated['accuracy_mean']:.4f} ± {aggregated['accuracy_std']:.4f}")
        logger.info(f"Macro F1:          {aggregated['macro_f1_mean']:.4f} ± {aggregated['macro_f1_std']:.4f}")
        logger.info(f"Weighted F1:       {aggregated['weighted_f1_mean']:.4f} ± {aggregated['weighted_f1_std']:.4f}")
        logger.info(f"Macro Precision:   {aggregated['macro_precision_mean']:.4f} ± {aggregated['macro_precision_std']:.4f}")
        logger.info(f"Macro Recall:      {aggregated['macro_recall_mean']:.4f} ± {aggregated['macro_recall_std']:.4f}")
        
        logger.info("\nPer-Class F1 Scores:")
        for label in self.label_names:
            key = f'f1_{label}_mean'
            if key in aggregated:
                mean_f1 = aggregated[key]
                std_f1 = aggregated[f'f1_{label}_std']
                logger.info(f"  {label:20s}: {mean_f1:.4f} ± {std_f1:.4f}")
    
    def save_results(self, fold_results: list, aggregated: dict, output_dir: Path):
        """
        Save results to files.
        
        Parameters
        ----------
        fold_results : list of dict
            Results from each fold
        aggregated : dict
            Aggregated metrics
        output_dir : Path
            Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save fold results as CSV
        fold_df = pd.DataFrame([
            {k: v for k, v in fold.items() if k != 'confusion_matrix'}
            for fold in fold_results
        ])
        fold_csv = output_dir / "fold_results.csv"
        fold_df.to_csv(fold_csv, index=False)
        logger.info(f"Saved fold results to {fold_csv}")
        
        # Save aggregated results as JSON
        aggregated_json = output_dir / "aggregated_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        aggregated_serializable = {}
        for key, value in aggregated.items():
            if isinstance(value, np.ndarray):
                aggregated_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                aggregated_serializable[key] = float(value)
            else:
                aggregated_serializable[key] = value
        
        with open(aggregated_json, 'w') as f:
            json.dump(aggregated_serializable, f, indent=2)
        logger.info(f"Saved aggregated results to {aggregated_json}")
        
        # Save per-class metrics as CSV
        per_class_metrics = []
        for label in self.label_names:
            row = {'class': label}
            for metric in ['precision', 'recall', 'f1']:
                key_mean = f'{metric}_{label}_mean'
                key_std = f'{metric}_{label}_std'
                if key_mean in aggregated:
                    row[f'{metric}_mean'] = aggregated[key_mean]
                    row[f'{metric}_std'] = aggregated[key_std]
            per_class_metrics.append(row)
        
        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_csv = output_dir / "per_class_metrics.csv"
        per_class_df.to_csv(per_class_csv, index=False)
        logger.info(f"Saved per-class metrics to {per_class_csv}")
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str,
        output_path: Path,
        normalize: bool = True
    ):
        """
        Plot confusion matrix.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix
        title : str
            Plot title
        output_path : Path
            Output file path
        normalize : bool
            Whether to normalize confusion matrix
        """
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_plot = cm_norm
            fmt = '.2f'
        else:
            cm_plot = cm
            fmt = 'd'
        
        plt.figure(figsize=(14, 12))
        
        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {output_path}")
    
    def save_fold_confusion_matrices(self, fold_results: list, output_dir: Path):
        """
        Save confusion matrices for all folds.
        
        Parameters
        ----------
        fold_results : list of dict
            Results from each fold
        output_dir : Path
            Output directory
        """
        cm_dir = output_dir / "confusion_matrices"
        cm_dir.mkdir(parents=True, exist_ok=True)
        
        for fold_result in fold_results:
            fold_idx = fold_result['fold']
            cm = fold_result['confusion_matrix']
            
            self.plot_confusion_matrix(
                cm,
                title=f"Confusion Matrix - Fold {fold_idx}",
                output_path=cm_dir / f"confusion_matrix_fold_{fold_idx}.png",
                normalize=True
            )
    
    def save_aggregated_confusion_matrix(self, aggregated: dict, output_dir: Path):
        """
        Save aggregated confusion matrix.
        
        Parameters
        ----------
        aggregated : dict
            Aggregated metrics
        output_dir : Path
            Output directory
        """
        cm_sum = aggregated['confusion_matrix_sum']
        
        self.plot_confusion_matrix(
            cm_sum,
            title="Aggregated Confusion Matrix (All Folds)",
            output_path=output_dir / "confusion_matrix_aggregated.png",
            normalize=True
        )
