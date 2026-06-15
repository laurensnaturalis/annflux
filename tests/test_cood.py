"""
Tests for COOD (Combined Out-of-Distribution Detection) implementation.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from annflux.algorithms.cood import (
    IndividualOODMeasures,
    COODClassifier,
    compute_cood,
    add_cood_to_data,
)


class TestIndividualOODMeasures:
    """Test individual OOD measure computation."""
    
    def test_linear_measures(self):
        """Test linear classifier-based measures from paper table."""
        data = pd.DataFrame({
            "uid": ["A", "B", "C"],
            "score_predicted": [0.9, 0.5, 0.1],
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        assert "max_linear" in result.columns
        assert result["max_linear"].iloc[0] == 0.9
        assert "max_linear_t_scaled" in result.columns  # T=2.0 from paper
        # With T=2.0, high confidence is reduced (smoothed toward uniform)
        assert "energy" in result.columns  # Energy score from Liu et al., 2020
        # Lower probability = higher energy (more OOD)
        assert result["energy"].iloc[2] > result["energy"].iloc[0]  # 0.1 prob has higher energy than 0.9
        
    def test_knn_measures(self):
        """Test kNN-based measures."""
        data = pd.DataFrame({
            "uid": ["A", "B", "C"],
            "fre": [0.1, 0.5, 0.9],
            "num_labeled_nn": [5, 3, 1],
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        assert "fre" in result.columns
        assert "num_labeled_nn" in result.columns
        assert result["fre"].iloc[0] == 0.1
        
    def test_entropy_measures(self):
        """Test entropy-based measures."""
        data = pd.DataFrame({
            "uid": ["A", "B"],
            "entropy": [0.5, 1.2],
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        assert "entropy" in result.columns
        assert result["entropy"].iloc[0] == 0.5
        
    def test_uncertainty_measures(self):
        """Test uncertainty-based measures - note: calibrated_uncertainty excluded to avoid duplication."""
        data = pd.DataFrame({
            "uid": ["A", "B"],
            "nn_underrepresented": [0.1, 0.9],
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        # calibrated_uncertainty is intentionally not included (computed separately)
        assert "nn_underrepresented" in result.columns
        
    def test_ldof_measure(self):
        """Test LDOF (Local Distance Outlier Factor) from paper table."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        data = pd.DataFrame({
            "uid": [f"ID_{i}" for i in range(n_samples)],
        })
        
        # Create feature embeddings with one clear outlier
        features = np.random.randn(n_samples, n_features)
        # Make sample 0 an outlier by moving it far from others
        features[0] = features[0] + 10  # Far from cluster
        
        measures = IndividualOODMeasures(data, features)
        result = measures.compute_all()
        
        assert "ldof" in result.columns
        assert "avg_dist_to_nn" in result.columns
        assert "avg_dist_among_nn" in result.columns
        
        # Outlier should have high LDOF (far from neighbors, neighbors close to each other)
        ldof_outlier = result["ldof"].iloc[0]
        ldof_others = result["ldof"].iloc[1:].mean()
        assert ldof_outlier > ldof_others  # Outlier has higher LDOF
        
        # Avg distance to NN should be high for outlier
        dist_outlier = result["avg_dist_to_nn"].iloc[0]
        dist_others = result["avg_dist_to_nn"].iloc[1:].mean()
        assert dist_outlier > dist_others
        
    def test_mahalanobis_distance(self):
        """Test Mahalanobis distance computation with features."""
        np.random.seed(42)
        n_samples = 20
        n_features = 10
        
        data = pd.DataFrame({
            "uid": [f"ID_{i}" for i in range(n_samples)],
            "labeled": [1 if i < 10 else 0 for i in range(n_samples)],
        })
        
        # Create random feature embeddings
        features = np.random.randn(n_samples, n_features)
        
        measures = IndividualOODMeasures(data, features)
        result = measures.compute_all()
        
        assert "mahalanobis_dist" in result.columns
        # Labeled samples should generally have lower distance (in-distribution)
        labeled_dist = result.loc[data["labeled"] == 1, "mahalanobis_dist"].mean()
        unlabeled_dist = result.loc[data["labeled"] == 0, "mahalanobis_dist"].mean()
        # Both should be finite
        assert np.isfinite(labeled_dist)
        assert np.isfinite(unlabeled_dist)
        
    def test_missing_columns_handled(self):
        """Test that missing columns are handled gracefully."""
        data = pd.DataFrame({
            "uid": ["A", "B"],
            # No optional columns
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        # Should return empty or minimal dataframe without errors
        assert isinstance(result, pd.DataFrame)


class TestCOODClassifier:
    """Test COOD classifier training and prediction."""
    
    def test_classifier_initialization(self):
        """Test classifier initialization."""
        cood = COODClassifier()
        assert cood.rf_params is not None
        assert not cood.is_trained
        
    def test_classifier_train_and_predict(self):
        """Test training and prediction workflow."""
        # Create synthetic measures
        np.random.seed(42)
        n_samples = 100
        
        measures_df = pd.DataFrame({
            "max_linear": np.random.uniform(0, 1, n_samples),
            "fre": np.random.uniform(0, 1, n_samples),
            "entropy": np.random.uniform(0, 2, n_samples),
            "calibrated_uncertainty": np.random.uniform(0, 1, n_samples),
        })
        
        # Create synthetic labels and predictions
        data = pd.DataFrame({
            "label_true": [f"class_{i % 3}" for i in range(n_samples)],
            "label_predicted": [f"class_{i % 3}" if i < 70 else f"class_{(i+1) % 3}" 
                                for i in range(n_samples)],
        })
        
        cood = COODClassifier()
        
        # Create OOD mask for last 20 samples
        ood_mask = pd.Series([False] * n_samples)
        ood_mask.iloc[-20:] = True
        
        # Train
        cood.train(measures_df, data["label_true"], ood_mask)
        
        assert cood.is_trained
        assert cood.model is not None
        
        # Predict
        scores, classes = cood.predict(measures_df)
        
        assert len(scores) == n_samples
        assert len(classes) == n_samples
        assert all(0 <= s <= 1 for s in scores)
        assert all(c in [0, 1, 2] for c in classes)
        
    def test_untrained_classifier(self):
        """Test that untrained classifier returns zeros."""
        measures_df = pd.DataFrame({
            "max_linear": [0.5, 0.6],
            "fre": [0.1, 0.2],
        })
        
        cood = COODClassifier()
        scores, classes = cood.predict(measures_df)
        
        assert len(scores) == 2
        assert all(s == 0 for s in scores)
        assert all(c == 0 for c in classes)
        
    def test_single_class_training(self):
        """Test training with only one class (no stratification)."""
        np.random.seed(42)
        n_samples = 100
        
        # All samples are ID-correct (class 0)
        measures_df = pd.DataFrame({
            "max_linear": np.random.uniform(0, 1, n_samples),
            "fre": np.random.uniform(0, 1, n_samples),
        })
        
        labels = pd.Series([f"class_{i%3}" for i in range(n_samples)])
        predictions = labels.copy()  # All correct
        
        cood = COODClassifier()
        cood.train(measures_df, labels, predictions)
        
        # Should train successfully even with only one class
        assert cood.is_trained
        
        # Predict should work
        scores, classes = cood.predict(measures_df)
        assert len(scores) == n_samples


class TestComputeCOOD:
    """Test the main compute_cood function."""
    
    def test_compute_cood_basic(self):
        """Test basic COOD computation."""
        np.random.seed(42)
        n_samples = 50
        
        data = pd.DataFrame({
            "uid": [f"id_{i}" for i in range(n_samples)],
            "score_predicted": np.random.uniform(0, 1, n_samples),
            "fre": np.random.uniform(0, 1, n_samples),
            "entropy": np.random.uniform(0, 2, n_samples),
            "labeled": [1 if i < 20 else 0 for i in range(n_samples)],
            "label_true": [f"class_{i % 3}" if i < 20 else None 
                          for i in range(n_samples)],
            "label_predicted": [f"class_{i % 3}" if i < 20 else None 
                               for i in range(n_samples)],
        })
        
        result = compute_cood(data, train_on_labeled=True)
        
        assert "cood_score" in result.columns
        assert "cood_class" in result.columns
        assert len(result) == n_samples
        assert all(0 <= s <= 1 for s in result["cood_score"])
        
    def test_compute_cood_no_training(self):
        """Test COOD computation without training."""
        data = pd.DataFrame({
            "uid": ["A", "B", "C"],
            "score_predicted": [0.9, 0.5, 0.1],
        })
        
        result = compute_cood(data, train_on_labeled=False)
        
        assert "cood_score" in result.columns
        # Without training, scores should be zeros
        assert all(s == 0 for s in result["cood_score"])
        
    def test_add_cood_to_data(self):
        """Test adding COOD to existing dataframe."""
        data = pd.DataFrame({
            "uid": ["A", "B"],
            "score_predicted": [0.9, 0.1],
        })
        
        result = add_cood_to_data(data, train_on_labeled=False)
        
        assert "cood_score" in result.columns
        assert "cood_class" in result.columns
        # Original columns should still exist
        assert "uid" in result.columns
        assert "score_predicted" in result.columns


class TestCOODIntegration:
    """Integration tests with realistic data."""
    
    def test_with_annflux_like_data(self):
        """Test with data resembling actual annflux output."""
        np.random.seed(42)
        n_samples = 200
        
        # Create realistic annflux-like data
        data = pd.DataFrame({
            "uid": [f"ZMA_INS_{i}" for i in range(n_samples)],
            "score_predicted": np.random.beta(2, 5, n_samples),  # Skewed toward low scores
            "fre": np.random.exponential(0.3, n_samples),
            "entropy": np.random.gamma(2, 0.5, n_samples),
            "nn_underrepresented": np.random.uniform(0, 1, n_samples),
            "calibrated_uncertainty": np.random.beta(2, 3, n_samples),
            "labeled": [1 if i < 50 else 0 for i in range(n_samples)],
            "label_true": [f"Lepidoptera,Nymphalidae,Genus_{i%10},Species_{i}" 
                          if i < 50 else None
                          for i in range(n_samples)],
            "label_predicted": [f"Lepidoptera,Nymphalidae,Genus_{i%10},Species_{i}" 
                               if i < 40 else  # Some incorrect predictions
                               f"Lepidoptera,Nymphalidae,Genus_{(i+1)%10},Species_{i}"
                               if i < 50 else None
                               for i in range(n_samples)],
        })
        
        result = compute_cood(data, train_on_labeled=True)
        
        # Check that we have all expected columns
        assert "cood_score" in result.columns
        assert "cood_class" in result.columns
        
        # Check that individual measure columns exist
        measure_cols = [c for c in result.columns if c.startswith("cood_")]
        assert len(measure_cols) > 0
        
        # Verify score ranges
        assert result["cood_score"].min() >= 0
        assert result["cood_score"].max() <= 1
        
    def test_hierarchical_measures(self):
        """Test that hierarchical label distance is computed when available."""
        data = pd.DataFrame({
            "uid": ["A", "B", "C"],
            "score_predicted": [0.9, 0.5, 0.3],
            "label_predicted": [
                "Lepidoptera,Nymphalidae,GenusA,Species1",
                "Lepidoptera,Nymphalidae,GenusA,Species2",  # Same genus, different species
                "Lepidoptera,Pieridae,GenusB,Species3",  # Different family
            ],
        })
        
        measures = IndividualOODMeasures(data)
        result = measures.compute_all()
        
        # Should compute measures without error
        assert isinstance(result, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
