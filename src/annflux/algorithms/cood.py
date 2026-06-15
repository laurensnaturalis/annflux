"""
COOD: Combined Out-of-Distribution Detection

Implementation of "Combined Out-of-distribution Detection Using Multiple Measures for Anomaly"
(Hogeweg et al., CVPRW 2024)

This module implements the COOD framework which combines multiple individual OOD measures
using a RandomForest classifier to produce a unified OOD score.
"""

import logging
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logger = logging.getLogger("annflux_training")

# Default RandomForest parameters for COOD
COOD_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


class IndividualOODMeasures:
    """
    Compute individual OOD measures as features for COOD.
    
    Based on the paper, these include:
    - Linear classifier based measures
    - kNN-based measures  
    - Entropy-based measures
    - Distance-based measures
    - Hierarchical measures (when taxonomy is available)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        features: Optional[np.ndarray] = None,
        nn_indices: Optional[np.ndarray] = None,
        nn_distances: Optional[np.ndarray] = None,
    ):
        """
        Initialize with annflux data.
        
        Args:
            data: DataFrame with annflux columns
            features: Optional feature embeddings (n_samples x n_features)
            nn_indices: Optional pre-computed kNN neighbor indices (n_samples x k)
            nn_distances: Optional pre-computed kNN distances (n_samples x k)
        """
        self.data = data
        self.features = features
        self.nn_indices = nn_indices
        self.nn_distances = nn_distances
        self.measures = {}
        
    def compute_all(self) -> pd.DataFrame:
        """Compute all individual OOD measures."""
        logger.info("[COOD] Computing individual OOD measures...")
        
        # Linear classifier based measures
        self._compute_linear_measures()
        
        # kNN-based measures
        self._compute_knn_measures()
        
        # Entropy-based measures
        self._compute_entropy_measures()
        
        # Distance-based measures
        self._compute_distance_measures()
        
        # Uncertainty-based measures
        self._compute_uncertainty_measures()
        
        # Feature-based measures (entropy, sum, magnitude)
        self._compute_feature_measures()
        
        # LDOF: Disabled for now (slow computation)
        # self._compute_ldof()
        
        # EnWeDi: Entropy Weighted Distance measures from paper
        self._compute_enwedi()
        
        # Feature-based Mahalanobis distance (if features available)
        self._compute_mahalanobis_distance()
        
        # Combine into feature dataframe
        measures_df = pd.DataFrame(self.measures)
        measures_df.index = self.data.index
        
        logger.info(f"[COOD] Computed {len(measures_df.columns)} individual measures")
        return measures_df
    
    def _compute_linear_measures(self):
        """Compute linear classifier-based OOD measures."""
        # Max(linear) - maximum softmax probability
        if "score_predicted" in self.data.columns:
            self.measures["max_linear"] = pd.to_numeric(
                self.data["score_predicted"], errors="coerce"
            ).fillna(0)
            
            # Max(linear-T-scaled): Temperature-scaled softmax with T=2.0 (from paper table)
            # Higher temperature (T>1) smooths distribution, reducing overconfidence
            # Proper formulation from ODIN: p_i^T = exp(logit_i/T) / sum_j exp(logit_j/T)
            # For max probability: p_max^T = exp(logit_max/T) / sum_j exp(logit_j/T)
            # With T=2.0: high-confidence predictions are "pushed down" toward uniform
            T_odin = 2.0  # T > 1 reduces overconfidence (as specified in paper)
            probs_clipped = np.clip(self.measures["max_linear"], 1e-10, 1.0)
            # Recover approximate logit: logit_i ≈ log(p_i) + log(Z) where Z is partition function
            # For temperature scaling: p_i^(1/T) = exp(logit_i/T) / Z^(1/T)
            # After scaling: p_i^T ∝ p_i^(1/T), so p_i^T = p_i^(1/T) / sum_j p_j^(1/T)
            # With T=2.0: p_i^2 = p_i^(0.5) / sum_j p_j^(0.5)
            # We apply power transform and renormalize
            probs_power = np.power(probs_clipped, 1.0 / T_odin)
            self.measures["max_linear_t_scaled"] = probs_power
            
            # Energy score (Liu et al., 2020) - exact formulation
            # Energy(x) = T * log(sum_i exp(logit_i / T))
            # Using all scores from scores_predicted if available, else approximate from p_max
            # With T=1: E(x) = log(sum_i exp(logit_i)) = log(sum_i exp(log(p_i)) * Z) = log(Z) + log(sum_i p_i)
            # Since sum_i p_i = 1, this gives E(x) = log(Z) = -log(p_max) + logit_max
            # Proper energy requires full logit vector, but we approximate using p_max
            T_energy = 1.0
            # From p_max, we can bound: log(sum exp(logits)) >= log(exp(logit_max)) = logit_max = -log(1/p_max - 1) for sigmoid
            # For softmax with many classes: E(x) ≈ -log(p_max) + log(K) where K is number of classes
            # We use the simpler form: E(x) = -log(p_max) which is the negative log-likelihood
            self.measures["energy"] = -T_energy * np.log(probs_clipped)
        
        # Linear probability margin
        if "score_possible" in self.data.columns:
            # Parse score_possible if it's a string representation of list
            try:
                scores = self.data["score_possible"].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [float(x)] if pd.notna(x) else [0]
                )
                if len(scores) > 0 and isinstance(scores.iloc[0], list):
                    max_scores = scores.apply(lambda s: max(s) if s else 0)
                    second_max = scores.apply(
                        lambda s: sorted(s, reverse=True)[1] if len(s) > 1 else 0
                    )
                    self.measures["linear_margin"] = max_scores - second_max
            except:
                pass
    
    def _compute_knn_measures(self):
        """Compute kNN-based OOD measures."""
        # kNN prediction confidence (if available)
        if "knn_score" in self.data.columns:
            self.measures["knn_max"] = pd.to_numeric(
                self.data["knn_score"], errors="coerce"
            ).fillna(0)
        
        # Distance to nearest labeled neighbor (related to FRE)
        if "fre" in self.data.columns:
            self.measures["fre"] = pd.to_numeric(
                self.data["fre"], errors="coerce"
            ).fillna(0)
        
        # Number of labeled nearest neighbors
        if "num_labeled_nn" in self.data.columns:
            self.measures["num_labeled_nn"] = pd.to_numeric(
                self.data["num_labeled_nn"], errors="coerce"
            ).fillna(0)
            
        # Minimum distance to neighbors
        if "min_distance" in self.data.columns:
            self.measures["min_distance"] = pd.to_numeric(
                self.data["min_distance"], errors="coerce"
            ).fillna(0)
    
    def _compute_entropy_measures(self):
        """Compute entropy-based OOD measures."""
        # Entropy of predicted class distribution
        if "entropy" in self.data.columns:
            self.measures["entropy"] = pd.to_numeric(
                self.data["entropy"], errors="coerce"
            ).fillna(0)
        
        # Parse scores_predicted for full distribution entropy
        if "scores_predicted" in self.data.columns:
            try:
                scores_list = self.data["scores_predicted"].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else []
                )
                
                def compute_entropy(scores):
                    if not scores or sum(scores) == 0:
                        return 0
                    probs = np.array(scores) / sum(scores)
                    probs = probs[probs > 0]  # Remove zeros for log
                    return -np.sum(probs * np.log(probs + 1e-10))
                
                self.measures["scores_entropy"] = scores_list.apply(compute_entropy)
            except:
                pass
    
    def _compute_distance_measures(self):
        """Compute distance-based OOD measures."""
        # NN underrepresented score
        if "nn_underrepresented" in self.data.columns:
            self.measures["nn_underrepresented"] = pd.to_numeric(
                self.data["nn_underrepresented"], errors="coerce"
            ).fillna(0)
        
        # Most needed score (active learning measure)
        if "most_needed" in self.data.columns:
            self.measures["most_needed"] = pd.to_numeric(
                self.data["most_needed"], errors="coerce"
            ).fillna(0)
    
    def _compute_uncertainty_measures(self):
        """Compute uncertainty-based OOD measures."""
        # Note: calibrated_uncertainty is computed separately and not included here
        # to avoid duplication and circular dependencies
        pass
    
    def _compute_feature_measures(self):
        """
        Compute feature-based OOD measures from paper table:
        - Feature entropy: entropy of normalized feature vector
        - Feature sum: sum of absolute feature values
        - Feature magnitude: L2 norm of feature vector
        """
        if self.features is None:
            return
        
        # Feature magnitude (L2 norm / length of feature vector)
        # OOD samples might have very low or very high feature values
        self.measures["feature_magnitude"] = np.linalg.norm(self.features, axis=1)
        
        # Feature sum: sum of absolute feature values
        # OOD features might have absence of feature responses
        self.measures["feature_sum"] = np.sum(np.abs(self.features), axis=1)
        
        # Feature entropy: entropy of normalized feature vector
        # For ID images, feature values could be more concentrated
        # Normalize features to [0, 1] range per sample, then compute entropy
        # Use softmax-like normalization: f_i / sum_j |f_j|
        features_abs = np.abs(self.features)
        features_sum = np.sum(features_abs, axis=1, keepdims=True) + 1e-10
        features_normalized = features_abs / features_sum
        # Compute entropy: -sum(p_i * log(p_i))
        # Avoid log(0) by masking
        log_features = np.log(features_normalized + 1e-10)
        entropy = -np.sum(features_normalized * log_features, axis=1)
        self.measures["feature_entropy"] = entropy
    
    def _compute_ldof(self):
        """
        Compute LDOF (Local Distance Outlier Factor) from paper table.
        
        LDOF = Avg. distance to NN / Avg. distance among NN
        
        High LDOF indicates the query point is far from its neighbors,
        while those neighbors are close to each other (clustered).
        This suggests the query point is an outlier.
        
        Uses pre-computed kNN indices/distances if available, otherwise computes from features.
        """
        # Use pre-computed kNN if available
        if self.nn_indices is not None and self.nn_distances is not None:
            nn_indices = self.nn_indices
            nn_distances = self.nn_distances
            k = nn_indices.shape[1]
        elif self.features is not None:
            # Compute kNN on the fly if features provided but no pre-computed kNN
            k = min(10, len(self.features) - 1)
            if k < 3:
                return
            
            try:
                from sklearn.neighbors import NearestNeighbors
                
                nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(self.features)
                distances, indices = nbrs.kneighbors(self.features)
                
                # Remove self (first neighbor is always self with distance 0)
                nn_distances = distances[:, 1:]  # (N, k)
                nn_indices = indices[:, 1:]  # (N, k)
            except Exception as e:
                logger.warning(f"[COOD] Could not compute kNN for LDOF: {e}")
                return
        else:
            return
        
        # Now compute LDOF from nn_indices and nn_distances
        try:
            # Avg. distance to NN for each sample
            avg_dist_to_nn = np.mean(nn_distances, axis=1)
            
            # Avg. distance among NN for each sample
            # For each sample i, compute pairwise distances among its k neighbors
            avg_dist_among_nn = np.zeros(len(self.data))
            
            if self.features is not None:
                for i in range(len(self.data)):
                    # Get the k nearest neighbors of sample i
                    neighbor_indices = nn_indices[i]
                    neighbor_features = self.features[neighbor_indices]
                    
                    # Compute pairwise distances among these neighbors
                    n_neighbors = len(neighbor_indices)
                    if n_neighbors >= 2:
                        dist_sum = 0
                        count = 0
                        for j in range(n_neighbors):
                            for l in range(j+1, n_neighbors):
                                dist = np.linalg.norm(neighbor_features[j] - neighbor_features[l])
                                dist_sum += dist
                                count += 1
                        avg_dist_among_nn[i] = dist_sum / count if count > 0 else 0
            else:
                # Without features, approximate using pre-computed distances
                # This is a fallback - ideally features should be available
                avg_dist_among_nn = np.mean(nn_distances, axis=1) * 0.5  # Rough approximation
            
            # LDOF = avg_dist_to_nn / avg_dist_among_nn
            # Handle division by zero
            ldof = np.zeros(len(self.data))
            nonzero_mask = avg_dist_among_nn > 1e-10
            ldof[nonzero_mask] = avg_dist_to_nn[nonzero_mask] / avg_dist_among_nn[nonzero_mask]
            ldof[~nonzero_mask] = avg_dist_to_nn[~nonzero_mask]  # If no internal distances, use avg_dist_to_nn
            
            self.measures["ldof"] = ldof
            self.measures["avg_dist_to_nn"] = avg_dist_to_nn
            self.measures["avg_dist_among_nn"] = avg_dist_among_nn
            
        except Exception as e:
            logger.warning(f"[COOD] Could not compute LDOF: {e}")
    
    def _compute_enwedi(self):
        """
        Compute EnWeDi (Entropy Weighted Distance) measures from paper table.
        
        EnWeDi(1st): Distance to 1st neighbor weighted by 1 + Entropy of NN's true class
        EnWeDi(average): Average distance to NN weighted by 1 + Entropy of NN's true class
        
        The entropy of NN's true class is calculated using the entropy of the
        class distribution among the k nearest neighbors. High entropy among
        neighbors indicates the query point is near a class boundary.
        """
        # Need pre-computed kNN
        if self.nn_indices is None or self.nn_distances is None:
            return
        
        # Need labels for entropy computation
        if "label_true" not in self.data.columns:
            return
        
        try:
            nn_indices = self.nn_indices
            nn_distances = self.nn_distances
            k = min(10, nn_indices.shape[1])  # Use up to 10 neighbors
            
            # Get true labels
            labels = self.data["label_true"].values
            
            enwedi_1st = np.zeros(len(self.data))
            enwedi_avg = np.zeros(len(self.data))
            nn_entropy = np.zeros(len(self.data))
            
            for i in range(len(self.data)):
                # Get neighbor indices and distances
                neighbor_idx = nn_indices[i, :k]
                neighbor_dist = nn_distances[i, :k]
                
                # Get labels of neighbors
                neighbor_labels = labels[neighbor_idx]
                
                # Compute entropy of neighbor class distribution
                # Filter out None/NaN labels
                valid_labels = [l for l in neighbor_labels if pd.notna(l) and l is not None]
                
                if len(valid_labels) > 0:
                    # Count occurrences of each label
                    unique, counts = np.unique(valid_labels, return_counts=True)
                    probs = counts / counts.sum()
                    # Compute entropy: -sum(p * log(p))
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    nn_entropy[i] = entropy
                    
                    # Weight factor: 1 + entropy (higher entropy = more uncertainty)
                    weight = 1.0 + entropy
                    
                    # EnWeDi(1st): Distance to 1st neighbor * weight
                    if len(neighbor_dist) > 0:
                        enwedi_1st[i] = neighbor_dist[0] * weight
                    
                    # EnWeDi(average): Average distance to NN * weight
                    enwedi_avg[i] = np.mean(neighbor_dist) * weight
            
            self.measures["enwedi_1st"] = enwedi_1st
            self.measures["enwedi_avg"] = enwedi_avg
            self.measures["nn_entropy"] = nn_entropy
            
        except Exception as e:
            logger.warning(f"[COOD] Could not compute EnWeDi: {e}")
    
    def _compute_mahalanobis_distance(self):
        """
        Compute Mahalanobis distance from feature embeddings.
        
        Mahalanobis distance measures how many standard deviations a sample
        is from the class mean. Higher distance = more OOD.
        """
        if self.features is None:
            return
        
        # Need labeled samples to compute class-conditional statistics
        if "labeled" not in self.data.columns:
            return
        
        labeled_mask = self.data["labeled"] == 1
        n_labeled = labeled_mask.sum()
        
        if n_labeled < 10 or self.features.shape[1] == 0:
            return
        
        try:
            from scipy.spatial.distance import mahalanobis
            from scipy.linalg import inv
            
            # Compute global mean and covariance on labeled samples
            labeled_features = self.features[labeled_mask]
            mean = np.mean(labeled_features, axis=0)
            cov = np.cov(labeled_features, rowvar=False)
            
            # Add small regularization for numerical stability
            cov += np.eye(cov.shape[0]) * 1e-6
            cov_inv = inv(cov)
            
            # Compute Mahalanobis distance for all samples
            distances = np.zeros(len(self.data))
            for i in range(len(self.data)):
                distances[i] = mahalanobis(self.features[i], mean, cov_inv)
            
            self.measures["mahalanobis_dist"] = distances
            
        except Exception as e:
            logger.warning(f"[COOD] Could not compute Mahalanobis distance: {e}")


class COODClassifier:
    """
    Combined OOD classifier using RandomForest.
    
    Trains on labeled data to distinguish:
    - ID-correct (in-distribution, correctly classified)
    - ID-incorrect (in-distribution, misclassified)
    - OOD (out-of-distribution)
    
    Produces a COOD score (0 = ID, 1 = OOD) and multi-class predictions.
    """
    
    def __init__(self, rf_params: Optional[Dict] = None):
        """
        Initialize COOD classifier.
        
        Args:
            rf_params: Optional custom RandomForest parameters
        """
        self.rf_params = rf_params or COOD_RF_PARAMS
        self.model: Optional[RandomForestClassifier] = None
        self.is_trained = False
        
    def train(
        self,
        measures_df: pd.DataFrame,
        labels: pd.Series,
        label_predictions: Optional[pd.Series] = None,
        ood_mask: Optional[pd.Series] = None,
        validation_split: float = 0.2,
    ) -> "COODClassifier":
        """
        Train the COOD classifier.
        
        Args:
            measures_df: DataFrame of individual OOD measures
            labels: Series of true labels (for determining ID-correct)
            label_predictions: Series of predicted labels (optional)
            ood_mask: Boolean series indicating OOD samples (if known)
            validation_split: Fraction for validation
            
        Returns:
            self for chaining
        """
        logger.info("[COOD] Training classifier...")
        
        try:
            # Prepare target: multi-class
            # 0 = ID-correct, 1 = ID-incorrect, 2 = OOD
            y = self._prepare_targets(measures_df, labels, label_predictions, ood_mask)
            
            # Extract features and clean data
            X = measures_df.fillna(0)
            
            # Replace infinity with large finite values and clip extreme values
            X = X.replace([np.inf, -np.inf], [1e10, -1e10])
            X = X.clip(lower=-1e10, upper=1e10)
            
            # Remove rows with invalid targets
            valid_mask = y >= 0
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 10:
                logger.warning(f"[COOD] Too few samples ({len(X)}) for training, skipping")
                return self
            
            logger.info(f"[COOD] Training with {len(X)} samples, {len(np.unique(y))} classes")
            
            # Split for validation
            if len(X) > 50 and validation_split > 0:
                # Only stratify if we have at least 2 classes
                unique_classes = len(np.unique(y))
                if unique_classes >= 2:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=validation_split, random_state=42, stratify=y
                    )
                else:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=validation_split, random_state=42
                    )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
            
            # Train RandomForest
            self.model = RandomForestClassifier(**self.rf_params)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Log training stats
            logger.info(f"[COOD] Training set: {len(X_train)} samples")
            logger.info(f"[COOD] Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            
            if X_val is not None:
                val_score = self.model.score(X_val, y_val)
                logger.info(f"[COOD] Validation accuracy: {val_score:.3f}")
            
            # Log feature importance
            self._log_feature_importance(X.columns)
            
        except Exception as e:
            logger.error(f"[COOD] Training failed: {e}", exc_info=True)
        
        return self
    
    def _prepare_targets(
        self,
        measures_df: pd.DataFrame,
        labels: pd.Series,
        label_predictions: Optional[pd.Series],
        ood_mask: Optional[pd.Series],
    ) -> pd.Series:
        """
        Prepare multi-class targets.
        
        Returns:
            Series with values: 0=ID-correct, 1=ID-incorrect, 2=OOD, -1=unknown
        """
        targets = pd.Series(-1, index=measures_df.index)
        
        # Check if predictions match labels
        if label_predictions is not None or "label_predicted" in measures_df.columns:
            if "label_predicted" in measures_df.columns:
                pred = measures_df["label_predicted"]
            else:
                pred = label_predictions
            
            # ID samples (have predictions)
            has_pred = pred.notna()
            
            # Correct predictions
            correct = has_pred & (pred == labels)
            targets[correct] = 0  # ID-correct
            
            # Incorrect predictions
            incorrect = has_pred & (pred != labels) & labels.notna()
            targets[incorrect] = 1  # ID-incorrect
        
        # OOD samples (explicitly marked or no prediction)
        if ood_mask is not None:
            targets[ood_mask] = 2  # OOD
        
        return targets
    
    def _log_feature_importance(self, feature_names: pd.Index):
        """Log top features by importance."""
        if self.model is None:
            return
            
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        logger.info("[COOD] Top 10 feature importances:")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            logger.info(f"[COOD]   {feature_names[idx]}: {importances[idx]:.4f}")
    
    def predict(self, measures_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict COOD scores and classes.
        
        Args:
            measures_df: DataFrame of individual OOD measures
            
        Returns:
            Tuple of (cood_scores, class_predictions)
            - cood_scores: 0 = ID, 1 = OOD (probability of OOD)
            - class_predictions: 0=ID-correct, 1=ID-incorrect, 2=OOD
        """
        if not self.is_trained or self.model is None:
            logger.warning("[COOD] Model not trained, returning zeros")
            return np.zeros(len(measures_df)), np.zeros(len(measures_df), dtype=int)
        
        X = measures_df.fillna(0)
        
        # Replace infinity with large finite values and clip extreme values
        X = X.replace([np.inf, -np.inf], [1e10, -1e10])
        X = X.clip(lower=-1e10, upper=1e10)
        
        # Get class probabilities
        proba = self.model.predict_proba(X)
        
        # COOD score = probability of OOD (class 2)
        # Handle case where model hasn't seen OOD samples
        if proba.shape[1] >= 3:
            cood_scores = proba[:, 2]  # OOD class probability
        elif proba.shape[1] == 2:
            # Binary case: ID vs ID-incorrect, no OOD training data
            # Use ID-incorrect probability as proxy for OOD
            cood_scores = proba[:, 1]
        else:
            cood_scores = np.zeros(len(X))
        
        # Class prediction
        class_pred = self.model.predict(X)
        
        return cood_scores, class_pred


def compute_cood(
    data: pd.DataFrame,
    features: Optional[np.ndarray] = None,
    nn_indices: Optional[np.ndarray] = None,
    nn_distances: Optional[np.ndarray] = None,
    train_on_labeled: bool = True,
    ood_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute COOD scores for all samples.
    
    This is the main entry point for COOD computation.
    
    Args:
        data: DataFrame with annflux data
        features: Optional feature embeddings
        nn_indices: Optional pre-computed kNN neighbor indices (n_samples x k)
        nn_distances: Optional pre-computed kNN distances (n_samples x k)
        train_on_labeled: Whether to train classifier on labeled samples
        ood_mask: Optional boolean series indicating known OOD samples
        
    Returns:
        DataFrame with COOD scores and class predictions
    """
    t_start = time.time()
    logger.info(f"[COOD] Starting computation for {len(data)} samples")
    if nn_indices is not None:
        logger.info(f"[COOD] Using pre-computed kNN: {nn_indices.shape}")
    
    # Compute individual measures
    logger.info("[COOD] Computing individual measures...")
    measures = IndividualOODMeasures(data, features, nn_indices, nn_distances)
    measures_df = measures.compute_all()
    logger.info(f"[COOD] Computed {len(measures_df.columns)} measures: {list(measures_df.columns)[:5]}...")
    
    # Initialize classifier
    logger.info("[COOD] Initializing classifier...")
    cood = COODClassifier()
    
    if train_on_labeled:
        # Get labeled samples
        labeled_mask = data.get("labeled", pd.Series(0, index=data.index)) == 1
        n_labeled = labeled_mask.sum()
        logger.info(f"[COOD] Found {n_labeled} labeled samples")
        if n_labeled > 10:
            labels = data.loc[labeled_mask, "label_true"] if "label_true" in data.columns else pd.Series(None, index=data.index[labeled_mask])
            label_predictions = data.loc[labeled_mask, "label_predicted"] if "label_predicted" in data.columns else None
            logger.info(f"[COOD] Training on {n_labeled} samples...")
            cood.train(
                measures_df[labeled_mask],
                labels,
                label_predictions,
                ood_mask[labeled_mask] if ood_mask is not None else None,
            )
            logger.info(f"[COOD] Training complete, model trained={cood.is_trained}")
    
    # Predict for all samples
    logger.info(f"[COOD] Predicting for {len(measures_df)} samples...")
    cood_scores, cood_classes = cood.predict(measures_df)
    logger.info(f"[COOD] Prediction complete, score range: [{cood_scores.min():.3f}, {cood_scores.max():.3f}]")
    
    # Create results dataframe
    results = pd.DataFrame({
        "cood_score": cood_scores,
        "cood_class": cood_classes,
    }, index=data.index)
    
    # Add individual measures for debugging
    for col in measures_df.columns:
        results[f"cood_{col}"] = measures_df[col]
    
    # Report COOD score calibration on labeled samples
    labeled_mask = data.get("labeled", pd.Series(0, index=data.index)) == 1
    n_labeled = labeled_mask.sum()
    if n_labeled > 0:
        labeled_scores = cood_scores[labeled_mask]
        labeled_classes = cood_classes[labeled_mask]
        
        # For calibration: focus on ID-correct (class 0) vs ID-incorrect (class 1)
        # COOD score should be LOW for ID-correct (confident, in-distribution correct)
        # and HIGHER for ID-incorrect (uncertain or misclassified)
        # Binary: 1 = ID-correct, 0 = ID-incorrect or OOD
        is_id_correct = (labeled_classes == 0).astype(int)
        
        # Binned calibration with ECE calculation
        bins = np.arange(0, 1.1, 0.1)
        ece_total = 0.0
        total_samples = n_labeled
        
        logger.info("[COOD] === Calibration Analysis (ID-Correct vs ID-Incorrect) ===")
        logger.info(f"[COOD] Labeled samples: {n_labeled}")
        logger.info(f"[COOD] ID-correct rate: {is_id_correct.mean():.3f} (class 0: {(labeled_classes == 0).sum()}, class 1: {(labeled_classes == 1).sum()}, class 2: {(labeled_classes == 2).sum()})")
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            if i == len(bins) - 2:  # Last bin [0.9, 1.0] inclusive
                mask = (labeled_scores >= lower) & (labeled_scores <= upper)
            else:
                mask = (labeled_scores >= lower) & (labeled_scores < upper)
            
            subset = labeled_scores[mask]
            n_subset = len(subset)
            if n_subset > 0:
                actual_id_correct_rate = is_id_correct[mask].mean()
                # For calibration: expected ID-correct rate should match COOD score
                # High COOD score = likely ID-incorrect/OOD, so expected ID-correct should be LOW
                # We invert: expected_id_correct = 1 - mean_score
                expected_id_correct = 1.0 - subset.mean()
                calibration_error = abs(expected_id_correct - actual_id_correct_rate)
                bin_weight = n_subset / total_samples
                ece_total += bin_weight * calibration_error
                
                logger.info(
                    f"[COOD] Bin [{lower:.1f}, {upper:.1f}]: "
                    f"n={n_subset:4d}, actual_id_correct={actual_id_correct_rate:.3f}, "
                    f"expected={expected_id_correct:.3f}, error={calibration_error:.3f}"
                )
        
        logger.info(f"[COOD] ECE (Expected Calibration Error): {ece_total:.4f}")
        
        # Class distribution among labeled samples
        class_counts = pd.Series(labeled_classes).value_counts().sort_index()
        logger.info(f"[COOD] Class distribution: {{0: ID-correct, 1: ID-incorrect, 2: OOD}} = {dict(class_counts)}")
    
    logger.info(f"[COOD] Computation complete in {time.time() - t_start:.2f}s")
    logger.info(f"[COOD] Mean COOD score: {cood_scores.mean():.3f}")
    logger.info(f"[COOD] High OOD samples (score > 0.5): {(cood_scores > 0.5).sum()}/{len(cood_scores)}")
    
    return results


def add_cood_to_data(
    data: pd.DataFrame,
    features: Optional[np.ndarray] = None,
    nn_indices: Optional[np.ndarray] = None,
    nn_distances: Optional[np.ndarray] = None,
    train_on_labeled: bool = True,
) -> pd.DataFrame:
    """
    Add COOD scores to annflux dataframe in-place.
    
    Args:
        data: DataFrame to modify
        features: Optional feature embeddings
        nn_indices: Optional pre-computed kNN neighbor indices
        nn_distances: Optional pre-computed kNN distances
        train_on_labeled: Whether to train on labeled samples
        
    Returns:
        Modified dataframe with COOD columns added
    """
    cood_results = compute_cood(data, features, nn_indices, nn_distances, train_on_labeled)
    
    # Add columns to data
    for col in cood_results.columns:
        data[col] = cood_results[col]
    
    return data
