"""
==========================================
NCA Feature Selection Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Neighborhood Component Analysis for feature selection.
"""

import logging
import numpy as np
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

from config import Config


logger = logging.getLogger(__name__)


class NCAFeatureSelector:
    """
    Feature selection using Neighborhood Component Analysis (NCA).
    
    Can operate in two modes:
    1. Projection: Project features to lower dimensional space
    2. Selection: Select top-k features based on NCA + L1 regularization
    """
    
    def __init__(self, config: Config, mode: str = "selection"):
        """
        Parameters
        ----------
        config : Config
            Configuration object
        mode : str
            "selection" or "projection"
        """
        self.config = config
        self.mode = mode
        self.scaler = StandardScaler()
        self.nca = None
        self.selector = None
        self.selected_features_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit NCA feature selector.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape [n_samples, n_features]
        y : np.ndarray
            Training labels of shape [n_samples]
        """
        logger.info(f"Fitting NCA feature selector (mode={self.mode})...")
        logger.info(f"Input shape: {X.shape}")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.mode == "projection":
            self._fit_projection(X_scaled, y)
        else:  # selection
            self._fit_selection(X_scaled, y)
        
        logger.info("NCA fitting complete")
    
    def _fit_projection(self, X: np.ndarray, y: np.ndarray):
        """
        Fit NCA in projection mode.
        
        Parameters
        ----------
        X : np.ndarray
            Scaled training features
        y : np.ndarray
            Training labels
        """
        n_components = min(self.config.nca_n_components, X.shape[1])
        
        logger.info(f"Projecting to {n_components} dimensions")
        
        self.nca = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            max_iter=self.config.nca_max_iter,
            tol=self.config.nca_tol,
            random_state=self.config.random_state,
            verbose=0
        )
        
        self.nca.fit(X, y)
        
        logger.info(f"NCA projection fitted: {X.shape[1]} -> {n_components} features")
    
    def _fit_selection(self, X: np.ndarray, y: np.ndarray):
        """
        Fit NCA in selection mode using L1-regularized logistic regression.
        
        Parameters
        ----------
        X : np.ndarray
            Scaled training features
        y : np.ndarray
            Training labels
        """
        # Step 1: Apply NCA transformation
        n_components = min(X.shape[1] - 1, X.shape[0] // 2)  # Prevent overfitting
        n_components = max(n_components, self.config.nca_n_components)
        
        logger.info(f"NCA intermediate projection to {n_components} dimensions")
        
        self.nca = NeighborhoodComponentsAnalysis(
            n_components=n_components,
            max_iter=self.config.nca_max_iter,
            tol=self.config.nca_tol,
            random_state=self.config.random_state,
            verbose=0
        )
        
        try:
            X_nca = self.nca.fit_transform(X, y)
        except Exception as e:
            logger.warning(f"NCA transformation failed: {e}, using original features")
            X_nca = X
        
        # Step 2: Apply L1-regularized Logistic Regression for feature selection
        logger.info("Selecting features using L1 regularization...")
        
        # Find optimal C using cross-validation
        C_candidates = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_C = 1.0
        best_score = -np.inf
        
        for C in C_candidates:
            lr = LogisticRegression(
                penalty='l1',
                C=C,
                solver='saga',
                max_iter=1000,
                random_state=self.config.random_state
            )
            
            try:
                scores = cross_val_score(lr, X_nca, y, cv=3, scoring='accuracy', n_jobs=1)
                mean_score = np.mean(scores)
                
                logger.debug(f"C={C}: accuracy={mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_C = C
            except Exception as e:
                logger.warning(f"CV failed for C={C}: {e}")
                continue
        
        logger.info(f"Best C: {best_C} (accuracy={best_score:.4f})")
        
        # Fit final model with best C
        lr_final = LogisticRegression(
            penalty='l1',
            C=best_C,
            solver='saga',
            max_iter=1000,
            random_state=self.config.random_state
        )
        
        lr_final.fit(X_nca, y)
        
        # Use SelectFromModel to select features
        self.selector = SelectFromModel(
            lr_final,
            prefit=True,
            max_features=self.config.nca_n_components
        )
        
        # Get selected feature indices
        support = self.selector.get_support(indices=True)
        self.selected_features_ = support
        
        logger.info(f"Selected {len(support)} features: indices {support[:10]}...")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted NCA.
        
        Parameters
        ----------
        X : np.ndarray
            Features of shape [n_samples, n_features]
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if self.mode == "projection":
            return self.nca.transform(X_scaled)
        else:  # selection
            # Apply NCA transformation
            try:
                X_nca = self.nca.transform(X_scaled)
            except Exception:
                X_nca = X_scaled
            
            # Select features
            X_selected = self.selector.transform(X_nca)
            
            return X_selected
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit NCA and transform features.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape [n_samples, n_features]
        y : np.ndarray
            Training labels of shape [n_samples]
            
        Returns
        -------
        np.ndarray
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_n_features(self) -> int:
        """Get number of output features."""
        if self.mode == "projection":
            return self.nca.n_components
        else:
            if self.selected_features_ is not None:
                return len(self.selected_features_)
            else:
                return self.config.nca_n_components
