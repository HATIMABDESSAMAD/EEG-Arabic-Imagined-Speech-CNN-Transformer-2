"""
==========================================
CSP One-vs-Rest Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Common Spatial Patterns (CSP) with One-vs-Rest for multi-class (16 classes).
"""

import logging
import numpy as np
from typing import Optional
from scipy import linalg

from config import Config


logger = logging.getLogger(__name__)


class CSPOneVsRest:
    """
    Common Spatial Patterns with One-vs-Rest strategy for multi-class.
    
    For each class k, trains a CSP(k vs all_others) and extracts features.
    Final features are concatenated across all OVR models.
    """
    
    def __init__(self, config: Config, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.csp_models = []  # One CSP per class
        self.n_components = config.csp_n_components_per_class
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP models using One-vs-Rest strategy.
        
        Parameters
        ----------
        X : np.ndarray
            Training epochs of shape [n_epochs, n_channels, n_samples]
        y : np.ndarray
            Training labels of shape [n_epochs]
        """
        logger.info(f"Fitting {self.n_classes} OVR CSP models...")
        
        self.csp_models = []
        
        for class_idx in range(self.n_classes):
            # Create binary labels: class_idx vs rest
            y_binary = (y == class_idx).astype(int)
            
            # Count samples
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == 0)
            
            logger.debug(
                f"Class {class_idx}: {n_pos} positive, {n_neg} negative samples"
            )
            
            # Train CSP for this class
            csp = self._fit_single_csp(X, y_binary)
            self.csp_models.append(csp)
        
        logger.info(f"CSP fitting complete. Total features: {self.get_n_features()}")
    
    def _fit_single_csp(self, X: np.ndarray, y_binary: np.ndarray) -> dict:
        """
        Fit a single binary CSP model.
        
        Parameters
        ----------
        X : np.ndarray
            Epochs of shape [n_epochs, n_channels, n_samples]
        y_binary : np.ndarray
            Binary labels (0 or 1)
            
        Returns
        -------
        dict
            CSP model containing filters and parameters
        """
        # Separate classes
        X_class0 = X[y_binary == 0]
        X_class1 = X[y_binary == 1]
        
        if len(X_class0) == 0 or len(X_class1) == 0:
            logger.warning("One class has no samples, returning identity filters")
            n_channels = X.shape[1]
            return {
                'filters': np.eye(n_channels),
                'patterns': np.eye(n_channels),
                'eigenvalues': np.ones(n_channels)
            }
        
        # Compute covariance matrices
        cov0 = self._compute_covariance(X_class0)
        cov1 = self._compute_covariance(X_class1)
        
        # Apply regularization if specified
        if self.config.csp_reg is not None:
            n_channels = cov0.shape[0]
            reg_matrix = self.config.csp_reg * np.eye(n_channels)
            cov0 += reg_matrix
            cov1 += reg_matrix
        
        # Solve generalized eigenvalue problem
        # cov0 * w = lambda * cov1 * w
        try:
            eigenvalues, eigenvectors = linalg.eigh(cov0, cov0 + cov1)
        except linalg.LinAlgError:
            logger.warning("Eigenvalue decomposition failed, using fallback")
            eigenvalues = np.ones(cov0.shape[0])
            eigenvectors = np.eye(cov0.shape[0])
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components: m largest and m smallest
        # m = n_components / 2 (default 2 pairs = 4 components)
        m = self.n_components // 2
        
        selected_idx = list(range(m)) + list(range(-m, 0))
        
        filters = eigenvectors[:, selected_idx].T  # Shape: [n_components, n_channels]
        
        # Compute patterns (inverse of filters)
        try:
            patterns = linalg.pinv(filters).T
        except linalg.LinAlgError:
            patterns = filters.T
        
        return {
            'filters': filters,
            'patterns': patterns,
            'eigenvalues': eigenvalues[selected_idx]
        }
    
    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute average covariance matrix for a set of epochs.
        
        Parameters
        ----------
        X : np.ndarray
            Epochs of shape [n_epochs, n_channels, n_samples]
            
        Returns
        -------
        np.ndarray
            Covariance matrix of shape [n_channels, n_channels]
        """
        n_epochs, n_channels, n_samples = X.shape
        
        # Compute covariance for each epoch and average
        cov_sum = np.zeros((n_channels, n_channels))
        
        for epoch in X:
            # Normalize by trace
            cov = np.cov(epoch)
            trace = np.trace(cov)
            if trace > 0:
                cov /= trace
            cov_sum += cov
        
        # Average over epochs
        cov_avg = cov_sum / n_epochs
        
        return cov_avg
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform epochs using fitted CSP models.
        
        Parameters
        ----------
        X : np.ndarray
            Epochs of shape [n_epochs, n_channels, n_samples]
            
        Returns
        -------
        np.ndarray
            CSP features of shape [n_epochs, n_classes * n_components]
        """
        n_epochs = X.shape[0]
        n_features = self.get_n_features()
        
        features = np.zeros((n_epochs, n_features))
        
        for class_idx, csp_model in enumerate(self.csp_models):
            # Extract features for this CSP model
            class_features = self._transform_single_csp(X, csp_model)
            
            # Place in appropriate columns
            start_col = class_idx * self.n_components
            end_col = start_col + self.n_components
            features[:, start_col:end_col] = class_features
        
        return features
    
    def _transform_single_csp(self, X: np.ndarray, csp_model: dict) -> np.ndarray:
        """
        Transform epochs using a single CSP model.
        
        Parameters
        ----------
        X : np.ndarray
            Epochs of shape [n_epochs, n_channels, n_samples]
        csp_model : dict
            Fitted CSP model
            
        Returns
        -------
        np.ndarray
            Features of shape [n_epochs, n_components]
        """
        n_epochs = X.shape[0]
        filters = csp_model['filters']
        n_components = filters.shape[0]
        
        features = np.zeros((n_epochs, n_components))
        
        for i, epoch in enumerate(X):
            # Project epoch onto CSP components
            Z = np.dot(filters, epoch)  # Shape: [n_components, n_samples]
            
            # Compute variance for each component
            var = np.var(Z, axis=1)  # Shape: [n_components]
            
            # Normalize by sum of variances (log-variance formula from literature)
            var_sum = np.sum(var)
            if var_sum > 0:
                # Normalized log-variance: log(var_i / sum(var_j))
                features[i] = np.log(var / var_sum + 1e-8)
            else:
                features[i] = np.log(var + 1e-8)
        
        return features
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit CSP models and transform data.
        
        Parameters
        ----------
        X : np.ndarray
            Training epochs of shape [n_epochs, n_channels, n_samples]
        y : np.ndarray
            Training labels of shape [n_epochs]
            
        Returns
        -------
        np.ndarray
            CSP features of shape [n_epochs, n_classes * n_components]
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_n_features(self) -> int:
        """Get total number of CSP features."""
        return self.n_classes * self.n_components
