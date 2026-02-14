"""
==========================================
Stacking Ensemble Module
ArEEG_Words: EEG Classification Pipeline
==========================================
Stacking classifier with multiple base learners and meta-learner.
"""

import logging
import numpy as np
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import Config


logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    Stacking ensemble classifier with multiple base learners.
    
    Base learners (inspired by winner paper):
    - LDA (Linear Discriminant Analysis)
    - KNN (K-Nearest Neighbors)
    - Linear SVM
    - RBF SVM
    - Naive Bayes
    - Random Forest or Decision Tree
    
    Meta-learner: Multinomial Logistic Regression
    """
    
    def __init__(self, config: Config, n_classes: int):
        self.config = config
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.stacking_clf = None
        
    def _create_base_classifiers(self):
        """
        Create base classifiers based on configuration.
        
        Returns
        -------
        list of tuples
            List of (name, classifier) tuples
        """
        base_classifiers = []
        
        for clf_name in self.config.base_classifiers:
            if clf_name == 'lda':
                clf = LinearDiscriminantAnalysis()
                base_classifiers.append(('lda', clf))
                
            elif clf_name == 'knn':
                clf = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
                base_classifiers.append(('knn', clf))
                
            elif clf_name == 'linear_svm':
                clf = SVC(
                    kernel='linear',
                    C=1.0,
                    probability=True,
                    class_weight='balanced',
                    random_state=self.config.random_state
                )
                base_classifiers.append(('linear_svm', clf))
                
            elif clf_name == 'rbf_svm':
                clf = SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    class_weight='balanced',
                    random_state=self.config.random_state
                )
                base_classifiers.append(('rbf_svm', clf))
                
            elif clf_name == 'naive_bayes':
                clf = GaussianNB()
                base_classifiers.append(('naive_bayes', clf))
                
            elif clf_name == 'decision_tree':
                clf = DecisionTreeClassifier(
                    max_depth=10,
                    class_weight='balanced',
                    random_state=self.config.random_state
                )
                base_classifiers.append(('decision_tree', clf))
                
            elif clf_name == 'random_forest':
                clf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    class_weight='balanced',
                    random_state=self.config.random_state,
                    n_jobs=1
                )
                base_classifiers.append(('random_forest', clf))
            
            else:
                logger.warning(f"Unknown classifier: {clf_name}")
        
        logger.info(f"Created {len(base_classifiers)} base classifiers: "
                   f"{[name for name, _ in base_classifiers]}")
        
        return base_classifiers
    
    def _create_meta_classifier(self):
        """
        Create meta-learner (multinomial logistic regression).
        
        Returns
        -------
        classifier
            Meta-learner instance
        """
        meta_clf = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=self.config.random_state,
            n_jobs=1
        )
        
        return meta_clf
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fit stacking ensemble.
        
        Parameters
        ----------
        X : np.ndarray
            Training features of shape [n_samples, n_features]
        y : np.ndarray
            Training labels of shape [n_samples]
        sample_weight : np.ndarray, optional
            Sample weights of shape [n_samples]
        """
        logger.info("Fitting stacking ensemble...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create base classifiers
        base_classifiers = self._create_base_classifiers()
        
        # Create meta-classifier
        meta_clf = self._create_meta_classifier()
        
        # Create stacking classifier
        self.stacking_clf = StackingClassifier(
            estimators=base_classifiers,
            final_estimator=meta_clf,
            cv=self.config.stacking_cv,
            stack_method='predict_proba',  # Use probabilities from base learners
            passthrough=False,  # Don't include original features
            n_jobs=1,
            verbose=0
        )
        
        # Fit stacking classifier
        # Note: StackingClassifier doesn't directly support sample_weight
        # We'll fit base learners directly if weights are provided
        if sample_weight is not None:
            logger.info("Using sample weights for base learners")
            self._fit_with_weights(X_scaled, y, sample_weight)
        else:
            self.stacking_clf.fit(X_scaled, y)
        
        logger.info("Stacking ensemble fitted successfully")
    
    def _fit_with_weights(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        """
        Fit stacking with sample weights.
        
        This is a workaround since StackingClassifier doesn't directly support sample_weight.
        We'll use weighted resampling instead.
        
        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training labels
        sample_weight : np.ndarray
            Sample weights
        """
        # Normalize weights
        weights = sample_weight / np.sum(sample_weight) * len(sample_weight)
        
        # Create weighted indices (bootstrap resampling based on weights)
        n_samples = len(X)
        rng = np.random.RandomState(self.config.random_state)
        
        # Sample with replacement according to weights
        weighted_indices = rng.choice(
            n_samples,
            size=n_samples,
            replace=True,
            p=weights / np.sum(weights)
        )
        
        X_weighted = X[weighted_indices]
        y_weighted = y[weighted_indices]
        
        # Fit stacking classifier on weighted samples
        self.stacking_clf.fit(X_weighted, y_weighted)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Features of shape [n_samples, n_features]
            
        Returns
        -------
        np.ndarray
            Predicted labels of shape [n_samples]
        """
        X_scaled = self.scaler.transform(X)
        return self.stacking_clf.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Features of shape [n_samples, n_features]
            
        Returns
        -------
        np.ndarray
            Predicted probabilities of shape [n_samples, n_classes]
        """
        X_scaled = self.scaler.transform(X)
        return self.stacking_clf.predict_proba(X_scaled)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            True labels
            
        Returns
        -------
        float
            Accuracy score
        """
        X_scaled = self.scaler.transform(X)
        return self.stacking_clf.score(X_scaled, y)
