"""
Calibration-based uncertainty score for Active Learning.

Trains a RandomForest calibration model on labeled data to predict correctness probability,
then applies it to all data to produce an uncertainty score for AL selection.
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

logger = logging.getLogger("annflux_training")


# Default RandomForest parameters (matching calibration_model.py)
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def _is_correct_prediction(row) -> bool:
    """Check if prediction is correct (exact label match)."""
    pred = row.get("label_predicted")
    true = row.get("label_true")
    
    if pd.isna(pred) or pd.isna(true):
        return False
    
    pred_set = set(str(pred).split(","))
    true_set = set(str(true).split(","))
    pred_set = {p.strip() for p in pred_set if p.strip()}
    true_set = {t.strip() for t in true_set if t.strip()}
    
    return pred_set == true_set


def _extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric features from annflux dataframe.
    
    Excludes string/id columns and target-related columns.
    """
    exclude_cols = {
        "uid",
        "path",
        "label_true",
        "label_predicted",
        "labels",
        "set",
        "labeled",
        "image_id",
        "url",
        "score_possible",
        "scores_predicted",
        "certainty",
        "correct",
        "calibrated_prob_correct",
        "calibrated_uncertainty",  # target column
        "uncalibrated_prob_correct",
        "incorrect_score"
    }
    
    # Convert score_predicted to numeric if present (it may be object dtype from CSV)
    if "score_predicted" in df.columns:
        original_dtype = df["score_predicted"].dtype
        df["score_predicted"] = pd.to_numeric(df["score_predicted"], errors="coerce")
        logger.info(f"[_extract_features] score_predicted converted: {original_dtype} -> {df['score_predicted'].dtype}, n_null={df['score_predicted'].isna().sum()}/{len(df)}")
    
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    X = df[feature_cols].copy()
    
    # Log all features being used
    logger.info(f"[_extract_features] Features used ({len(feature_cols)}): {feature_cols}")
    if "score_predicted" in feature_cols:
        # Use numpy for faster min/max
        score_vals = X['score_predicted'].values
        logger.info(f"[_extract_features] score_predicted range: [{np.nanmin(score_vals):.3f}, {np.nanmax(score_vals):.3f}]")
    else:
        logger.info("[_extract_features] score_predicted NOT in features - excluded or non-numeric")
    
    # Domain adaptation: normalize data-dependent features using labeled data statistics
    # Data-dependent features change based on which items are labeled (domain shift)
    # Intrinsic features (embeddings, score_predicted) are stable and should not be normalized
    if "labeled" in df.columns:
        # Ensure boolean mask aligned with X index
        labeled_mask = df["labeled"].astype(bool).values
        n_labeled = labeled_mask.sum()
        logger.info(f"[DOMAIN_ADAPT] Found {n_labeled} labeled samples for domain adaptation")
    else:
        labeled_mask = np.array([False] * len(df))
        n_labeled = 0
    
    if n_labeled > 1:
        # Data-dependent features: these depend on the labeled set and show distribution shift
        data_dependent_features = [
            "fre", "fre_strat", "nn_underrepresented",  # depend on labeled neighbors
            "num_labeled_nn", "min_distance", "entropy",  # depend on labeled neighbors
            "most_needed", "direct_most_needed", "dp_most_needed",  # ranking features
            "num_children", "num_children_alt",  # cluster features that depend on labeled
        ]
        # Intrinsic features to exclude: e_* (embeddings), score_predicted (model output),
        # dp_cluster, dp_depth, dp_parent, display_order (intrinsic to image)
        normalized_count = 0
        # Convert to numpy for faster operations
        X_values = X.values
        col_idx_map = {col: i for i, col in enumerate(X.columns)}
        
        for feat in data_dependent_features:
            if feat in col_idx_map:
                col_idx = col_idx_map[feat]
                feat_values = X_values[:, col_idx]
                labeled_vals = feat_values[labeled_mask]
                # Filter out both NaN and Inf values for stats computation
                labeled_vals = labeled_vals[~np.isnan(labeled_vals) & ~np.isinf(labeled_vals)]
                if len(labeled_vals) > 1:
                    labeled_mean = np.nanmean(labeled_vals)
                    labeled_std = np.nanstd(labeled_vals)
                    if labeled_std > 0 and not np.isnan(labeled_std):
                        # Normalize both labeled and unlabeled using labeled stats
                        X_values[:, col_idx] = (feat_values - labeled_mean) / labeled_std
                        normalized_count += 1
                        logger.info(f"[DOMAIN_ADAPT] {feat}: z-scored (labeled mean={labeled_mean:.3f}, std={labeled_std:.3f})")
        
        # Convert back to DataFrame
        X = pd.DataFrame(X_values, columns=X.columns, index=X.index)
        logger.info(f"[DOMAIN_ADAPT] Normalized {normalized_count} data-dependent features, left {len(X.columns) - normalized_count} intrinsic features unchanged")
    
    # Handle missing values and infinities using numpy for speed
    X = X.fillna(0)
    
    # Convert to numpy for vectorized operations
    X_values = X.values
    
    # Handle infinities: replace with nan, then fill with 0
    X_values = np.where(np.isinf(X_values), np.nan, X_values)
    X_values = np.nan_to_num(X_values, nan=0.0)
    
    # Clip extreme values
    X_values = np.clip(X_values, -1e6, 1e6)
    
    # Convert back to DataFrame
    return pd.DataFrame(X_values, columns=X.columns, index=X.index)


def _explain_features_with_shap(
    model: RandomForestClassifier,
    labeled_df: pd.DataFrame,
    n_samples: int = 100,
) -> None:
    """
    Compute and log SHAP feature importance for the calibration model.
    
    Shows which features the model uses to predict correctness.
    """
    try:
        import shap
    except ImportError:
        logger.info("[SHAP] shap not installed, skipping feature importance")
        return
    
    try:
        # Extract features
        X = _extract_features(labeled_df)
        
        # Sample for efficiency
        X_explain = X.head(n_samples) if len(X) > n_samples else X
        feature_names = X_explain.columns.tolist()
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = np.asarray(shap_values[1])  # Class 1 (correct)
        else:
            shap_values = np.asarray(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1]  # Class 1
        
        # Global importance: mean absolute SHAP value
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        shap_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        
        # Log top features
        logger.info("[SHAP] Top features by importance (mean |SHAP|):")
        for i, row in shap_importance.head(10).iterrows():
            logger.info(f"[SHAP]   {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        # Log example contributions for first prediction
        if len(X_explain) > 0:
            instance_shap = pd.DataFrame({
                "feature": feature_names,
                "shap_value": shap_values[0],
                "feature_value": X_explain.iloc[0].values,
            })
            instance_shap["abs_shap"] = instance_shap["shap_value"].abs()
            instance_shap = instance_shap.sort_values("abs_shap", ascending=False)
            
            logger.info("[SHAP] Top contributors for first prediction:")
            for _, row in instance_shap.head(5).iterrows():
                direction = "+" if row["shap_value"] > 0 else "-"
                logger.info(f"[SHAP]   {row['feature']}={row['feature_value']:.3f} ({direction}{abs(row['shap_value']):.3f})")
                
    except Exception as e:
        logger.warning(f"[SHAP] Error computing SHAP values: {e}")


def _train_calibration_model(
    labeled_df: pd.DataFrame,
    validation_split: float = 0.2,
) -> tuple[RandomForestClassifier, IsotonicRegression]:
    """
    Train calibration model on labeled data.
    
    Returns:
        Tuple of (RandomForest model, Isotonic calibrator)
    """
    # Target: correct prediction (binary)
    y = labeled_df.apply(_is_correct_prediction, axis=1).astype(int)
    
    # Extract features
    X = _extract_features(labeled_df)
    
    # Split for calibration
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=42, stratify=y
    )
    
    # Train RandomForest
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Handle edge case: if all labels are same class, predict_proba has only 1 column
    proba = rf.predict_proba(X_val)
    if proba.shape[1] == 1:
        # All predictions are the same class - return dummy calibrator that passes through
        logger.warning(f"[calibration_model] Only one class in predictions ({'correct' if y.iloc[0] == 1 else 'incorrect'}), skipping calibration")
        y_val_proba = np.full(len(X_val), float(y.iloc[0]))
    else:
        y_val_proba = proba[:, 1]
    
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_val_proba, y_val)
    
    # Log calibration model details
    logger.info(f"[calibration_model] Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
    logger.info(f"[calibration_model] Correct predictions: {y.sum()}/{len(y)} ({100*y.mean():.1f}%)")
    
    # Log isotonic regression calibration mapping (sample points)
    x_cal = calibrator.X_min_ + np.linspace(0, 1, 11) * (calibrator.X_max_ - calibrator.X_min_)
    y_cal = calibrator.predict(x_cal)
    cal_points = [f"({x:.2f}→{y:.2f})" for x, y in zip(x_cal, y_cal)]
    logger.info(f"[calibration_model] Isotonic calibration mapping: {' '.join(cal_points)}")
    
    # Log calibration effect on validation set
    y_val_cal = calibrator.predict(y_val_proba)
    logger.info(f"[calibration_model] Uncalibrated probs on val: min={y_val_proba.min():.3f}, max={y_val_proba.max():.3f}, mean={y_val_proba.mean():.3f}")
    logger.info(f"[calibration_model] Calibrated probs on val: min={y_val_cal.min():.3f}, max={y_val_cal.max():.3f}, mean={y_val_cal.mean():.3f}")
    
    return rf, calibrator


def _log_calibration_analysis(labeled_df: pd.DataFrame, rf: RandomForestClassifier, calibrator: IsotonicRegression) -> None:
    """
    Analyze and log calibration metrics on labeled data.
    Similar to analyze_calibration in calibration_model.py but logs to training logger.
    """
    try:
        # Get predictions on labeled data
        X_labeled = _extract_features(labeled_df)
        y_true = labeled_df.apply(_is_correct_prediction, axis=1).astype(int)
        proba_labeled = rf.predict_proba(X_labeled)
        if proba_labeled.shape[1] == 1:
            proba_uncalibrated = proba_labeled[:, 0] if len(proba_labeled) > 0 else np.array([])
        else:
            proba_uncalibrated = proba_labeled[:, 1]
        proba_calibrated = calibrator.predict(proba_uncalibrated)
        
        # Add to dataframe for analysis
        labeled_analysis = pd.DataFrame({
            "correct": y_true,
            "calibrated_prob_correct": proba_calibrated,
            "uncalibrated_prob_correct": proba_uncalibrated,
        })
        
        logger.info("[calibration_model] === Calibration Analysis ===")
        
        # Filter to only rows with non-null score_predicted for fair comparison
        if "score_predicted" in labeled_df.columns:
            score_not_null = labeled_df["score_predicted"].notna()
            labeled_analysis_filtered = labeled_analysis[score_not_null]
            filtered_count = len(labeled_analysis_filtered)
            logger.info(f"[calibration_model] Filtering to {filtered_count} rows with non-null score_predicted (from {len(labeled_analysis)} total)")
        else:
            labeled_analysis_filtered = labeled_analysis
        
        # Binned calibration with ECE calculation
        bins = np.arange(0, 1.1, 0.1)
        ece_total = 0.0
        total_samples = len(labeled_analysis_filtered)
        last_bin_error = None
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            # Last bin [0.9, 1.0] should be inclusive of 1.0
            if i == len(bins) - 2:
                mask = (labeled_analysis_filtered["calibrated_prob_correct"] >= lower) & (
                    labeled_analysis_filtered["calibrated_prob_correct"] <= upper
                )
            else:
                mask = (labeled_analysis_filtered["calibrated_prob_correct"] >= lower) & (
                    labeled_analysis_filtered["calibrated_prob_correct"] < upper
                )
            subset = labeled_analysis_filtered[mask]
            if len(subset) > 0:
                actual_acc = subset["correct"].mean()
                expected_prob = subset["calibrated_prob_correct"].mean()
                calibration_error = abs(expected_prob - actual_acc)
                bin_weight = len(subset) / total_samples
                ece_total += bin_weight * calibration_error
                
                # Store last bin (0.9-1.0) error
                if i == len(bins) - 2:  # Last bin is 0.9-1.0
                    last_bin_error = calibration_error
                
                logger.info(
                    f"[calibration_model] Prob [{lower:.1f}, {upper:.1f}]: "
                    f"n={len(subset)}, actual_acc={actual_acc:.3f}, "
                    f"expected={expected_prob:.3f}, cal_error={calibration_error:.3f}"
                )
        
        logger.info(f"[calibration_model] Expected Calibration Error (ECE): {ece_total:.4f}")
        if last_bin_error is not None:
            logger.info(f"[calibration_model] Calibration error for bin [0.9, 1.0]: {last_bin_error:.4f}")
        
        # ECE analysis for uncalibrated probabilities (without isotonic regression)
        logger.info("[calibration_model] === Uncalibrated Analysis ===")
        ece_uncal_total = 0.0
        last_bin_uncal_error = None
        
        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                mask = (labeled_analysis_filtered["uncalibrated_prob_correct"] >= lower) & (
                    labeled_analysis_filtered["uncalibrated_prob_correct"] <= upper
                )
            else:
                mask = (labeled_analysis_filtered["uncalibrated_prob_correct"] >= lower) & (
                    labeled_analysis_filtered["uncalibrated_prob_correct"] < upper
                )
            subset = labeled_analysis_filtered[mask]
            if len(subset) > 0:
                actual_acc = subset["correct"].mean()
                expected_prob = subset["uncalibrated_prob_correct"].mean()
                calibration_error = abs(expected_prob - actual_acc)
                bin_weight = len(subset) / total_samples
                ece_uncal_total += bin_weight * calibration_error
                
                if i == len(bins) - 2:
                    last_bin_uncal_error = calibration_error
                
                logger.info(
                    f"[calibration_model] Uncal [{lower:.1f}, {upper:.1f}]: "
                    f"n={len(subset)}, actual_acc={actual_acc:.3f}, "
                    f"expected={expected_prob:.3f}, cal_error={calibration_error:.3f}"
                )
        
        logger.info(f"[calibration_model] Uncalibrated ECE: {ece_uncal_total:.4f}")
        if last_bin_uncal_error is not None:
            logger.info(f"[calibration_model] Uncalibrated calibration error for bin [0.9, 1.0]: {last_bin_uncal_error:.4f}")
        logger.info(f"[calibration_model] Isotonic improvement: {ece_uncal_total - ece_total:+.4f}")
        
        # Correlation between predicted probability and actual correctness
        corr = labeled_analysis_filtered["calibrated_prob_correct"].corr(labeled_analysis_filtered["correct"])
        logger.info(f"[calibration_model] Correlation (prob vs actual): {corr:.4f}")
        
        # Compare with raw score_predicted calibration
        if "score_predicted" in labeled_df.columns:
            logger.info("[calibration_model] === Comparison with score_predicted ===")
            score_vals = labeled_df["score_predicted"]
            score_nan_count = score_vals.isna().sum()
            valid_count = len(score_vals) - score_nan_count
            logger.info(f"[calibration_model] score_predicted: {len(score_vals)} total, {score_nan_count} NaN, {valid_count} valid")
            labeled_analysis_filtered["score_prob"] = score_vals[score_not_null].clip(0, 1).values
            
            # Debug: show score distribution
            score_min = labeled_analysis_filtered["score_prob"].min()
            score_max = labeled_analysis_filtered["score_prob"].max()
            score_mean = labeled_analysis_filtered["score_prob"].mean()
            logger.info(f"[calibration_model] score_prob stats: min={score_min:.3f}, max={score_max:.3f}, mean={score_mean:.3f}")
            
            # ECE calculation for score_predicted
            ece_score_total = 0.0
            last_bin_score_error = None
            total_in_bins = 0
            
            for i in range(len(bins) - 1):
                lower, upper = bins[i], bins[i + 1]
                # Last bin [0.9, 1.0] should be inclusive of 1.0
                if i == len(bins) - 2:
                    mask = (labeled_analysis_filtered["score_prob"] >= lower) & (labeled_analysis_filtered["score_prob"] <= upper)
                else:
                    mask = (labeled_analysis_filtered["score_prob"] >= lower) & (labeled_analysis_filtered["score_prob"] < upper)
                subset = labeled_analysis_filtered[mask]
                bin_count = len(subset)
                total_in_bins += bin_count
                if bin_count > 0:
                    actual_acc = subset["correct"].mean()
                    expected_prob = subset["score_prob"].mean()
                    calibration_error = abs(expected_prob - actual_acc)
                    bin_weight = bin_count / valid_count
                    ece_score_total += bin_weight * calibration_error
                    
                    # Store last bin (0.9-1.0] error
                    if i == len(bins) - 2:
                        last_bin_score_error = calibration_error
                    
                    logger.info(
                        f"[calibration_model] Score [{lower:.1f}, {upper:.1f}]: "
                        f"n={bin_count}, actual_acc={actual_acc:.3f}, "
                        f"expected={expected_prob:.3f}, cal_error={calibration_error:.3f}"
                    )
            
            logger.info(f"[calibration_model] Total in all bins: {total_in_bins} (expected {valid_count})")
            logger.info(f"[calibration_model] Score ECE: {ece_score_total:.4f}")
            if last_bin_score_error is not None:
                logger.info(f"[calibration_model] Score calibration error for bin [0.9, 1.0]: {last_bin_score_error:.4f}")
            
            score_corr = labeled_analysis_filtered["score_prob"].corr(labeled_analysis_filtered["correct"])
            logger.info(f"[calibration_model] Correlation (score_predicted vs actual): {score_corr:.4f}")
            logger.info(f"[calibration_model] Model improvement: {corr - score_corr:+.4f}")
        
    except Exception as e:
        logger.warning(f"[calibration_model] Could not compute calibration analysis: {e}")


def _select_best_probability_score(
    labeled_df: pd.DataFrame,
    rf: RandomForestClassifier,
    calibrator: IsotonicRegression,
    proba_calibrated_all: np.ndarray,
    proba_uncalibrated_all: np.ndarray,
    score_predicted_all: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute last bin calibration errors on labeled data and select the best probability score.
    
    Returns the probability array (calibrated, uncalibrated, or score_predicted) 
    that has the lowest last bin [0.9, 1.0] calibration error.
    """
    try:
        # Get predictions on labeled data
        X_labeled = _extract_features(labeled_df)
        y_true = labeled_df.apply(_is_correct_prediction, axis=1).astype(int)
        
        proba_labeled = rf.predict_proba(X_labeled)
        if proba_labeled.shape[1] == 1:
            proba_uncal_labeled = proba_labeled[:, 0] if len(proba_labeled) > 0 else np.array([])
        else:
            proba_uncal_labeled = proba_labeled[:, 1]
        proba_cal_labeled = calibrator.predict(proba_uncal_labeled)
        
        # Build analysis dataframe
        labeled_analysis = pd.DataFrame({
            "correct": y_true,
            "calibrated": proba_cal_labeled,
            "uncalibrated": proba_uncal_labeled,
        })
        
        # Filter to valid score_predicted if available
        if "score_predicted" in labeled_df.columns:
            score_not_null = labeled_df["score_predicted"].notna()
            labeled_analysis_filtered = labeled_analysis[score_not_null]
            labeled_analysis_filtered["score"] = pd.to_numeric(
                labeled_df.loc[score_not_null, "score_predicted"], errors="coerce"
            ).clip(0, 1).values
        else:
            labeled_analysis_filtered = labeled_analysis
        
        # Compute last bin [0.9, 1.0] errors for all three scores
        bins = np.arange(0, 1.1, 0.1)
        last_bin_idx = len(bins) - 2  # Index of [0.9, 1.0] bin
        lower, upper = bins[last_bin_idx], bins[last_bin_idx + 1]
        
        errors = {}
        
        # Calibrated last bin error
        mask_cal = (labeled_analysis_filtered["calibrated"] >= lower) & (labeled_analysis_filtered["calibrated"] <= upper)
        subset_cal = labeled_analysis_filtered[mask_cal]
        if len(subset_cal) > 0:
            actual_acc = subset_cal["correct"].mean()
            expected_prob = subset_cal["calibrated"].mean()
            errors["calibrated"] = abs(expected_prob - actual_acc)
        
        # Uncalibrated last bin error
        mask_uncal = (labeled_analysis_filtered["uncalibrated"] >= lower) & (labeled_analysis_filtered["uncalibrated"] <= upper)
        subset_uncal = labeled_analysis_filtered[mask_uncal]
        if len(subset_uncal) > 0:
            actual_acc = subset_uncal["correct"].mean()
            expected_prob = subset_uncal["uncalibrated"].mean()
            errors["uncalibrated"] = abs(expected_prob - actual_acc)
        
        # Score_predicted last bin error
        if "score" in labeled_analysis_filtered.columns:
            mask_score = (labeled_analysis_filtered["score"] >= lower) & (labeled_analysis_filtered["score"] <= upper)
            subset_score = labeled_analysis_filtered[mask_score]
            if len(subset_score) > 0:
                actual_acc = subset_score["correct"].mean()
                expected_prob = subset_score["score"].mean()
                errors["score_predicted"] = abs(expected_prob - actual_acc)
        
        # Select best score (lowest last bin error)
        if not errors:
            logger.info("[calibration_selection] No valid last bin errors computed, using calibrated")
            return proba_calibrated_all
        
        best_score_name = min(errors, key=errors.get)
        best_error = errors[best_score_name]
        
        logger.info(f"[calibration_selection] Last bin [0.9,1.0] errors: {errors}")
        logger.info(f"[calibration_selection] Selected '{best_score_name}' with error={best_error:.4f}")
        
        if best_score_name == "calibrated":
            return proba_calibrated_all
        elif best_score_name == "uncalibrated":
            return proba_uncalibrated_all
        else:  # score_predicted
            if score_predicted_all is not None:
                logger.info("[calibration_selection] Using score_predicted (best last bin calibration)")
                return score_predicted_all
            logger.info("[calibration_selection] score_predicted has best calibration but no all-data scores available, using calibrated")
            return proba_calibrated_all
            
    except Exception as e:
        logger.warning(f"[calibration_selection] Error selecting best score: {e}, using calibrated")
        return proba_calibrated_all


def compute_calibrated_uncertainty(
    data: pd.DataFrame,
    labeled_indices: Optional[np.ndarray] = None,
) -> pd.Series:
    """
    Compute calibrated uncertainty score for all data.
    
    For labeled data: trains a calibration model and applies it to all rows.
    For unlabeled data: returns uncertainty based on the model's confidence.
    
    The uncertainty score is defined as: 1 - calibrated_prob_correct
    Higher values = more uncertain = better for active learning.
    
    Args:
        data: DataFrame with all annflux data
        labeled_indices: Array of indices for labeled rows (if None, uses data["labeled"] == 1)
    
    Returns:
        Series with calibrated_uncertainty score for each row
    """
    t_start = time.time()
    
    # Determine labeled rows
    if labeled_indices is None:
        if "labeled" not in data.columns:
            logger.warning("[calibration_uncertainty] No 'labeled' column found, cannot compute")
            return pd.Series(index=data.index, data=np.nan)
        labeled_mask = data["labeled"] == 1
        labeled_df = data[labeled_mask]
    else:
        labeled_df = data.iloc[labeled_indices]
    
    # Need minimum labeled samples to train
    min_labeled = 50
    if len(labeled_df) < min_labeled:
        logger.warning(
            f"[calibration_uncertainty] Insufficient labeled data: {len(labeled_df)} < {min_labeled}"
        )
        # Return fallback: use 1 - score_predicted if available
        if "score_predicted" in data.columns:
            logger.info("[calibration_uncertainty] Using 1 - score_predicted as fallback")
            scores = pd.to_numeric(data["score_predicted"], errors="coerce").fillna(0.5)
            return 1.0 - scores
        return pd.Series(index=data.index, data=0.5)  # neutral uncertainty
    
    # Also need enough labeled rows with label_true to assess correctness
    labeled_with_true = labeled_df[labeled_df["label_true"].notna()]
    if len(labeled_with_true) < min_labeled:
        logger.warning(
            f"[calibration_uncertainty] Insufficient labeled+verified data: {len(labeled_with_true)} < {min_labeled}"
        )
        # Return fallback
        if "score_predicted" in data.columns:
            scores = pd.to_numeric(data["score_predicted"], errors="coerce").fillna(0.5)
            return 1.0 - scores
        return pd.Series(index=data.index, data=0.5)
    
    logger.info(
        f"[calibration_uncertainty] Training on {len(labeled_with_true)} labeled+verified rows"
    )
    
    try:
        # Train calibration model
        t_train_start = time.time()
        rf, calibrator = _train_calibration_model(labeled_with_true)
        logger.info(f"[TIMING] calibration_uncertainty training={time.time() - t_train_start:.3f}s")
        
        # Log detailed calibration analysis
        _log_calibration_analysis(labeled_with_true, rf, calibrator)
        
        # SHAP feature importance explanation (disabled by default for speed, enable with ENABLE_SHAP=1)
        if os.environ.get("ENABLE_SHAP") == "1":
            _explain_features_with_shap(rf, labeled_with_true)
        
        # Apply to all data
        t_apply_start = time.time()
        X_all = _extract_features(data)
        proba_all = rf.predict_proba(X_all)
        if proba_all.shape[1] == 1:
            # Single class case - all predictions have same probability
            proba_uncalibrated = proba_all[:, 0] if len(proba_all) > 0 else np.array([])
        else:
            proba_uncalibrated = proba_all[:, 1]
        proba_calibrated = calibrator.predict(proba_uncalibrated)
        
        # Log distribution of probabilities
        logger.info(f"[calibration_model] All data - Uncalibrated: min={proba_uncalibrated.min():.3f}, max={proba_uncalibrated.max():.3f}, mean={proba_uncalibrated.mean():.3f}")
        logger.info(f"[calibration_model] All data - Calibrated: min={proba_calibrated.min():.3f}, max={proba_calibrated.max():.3f}, mean={proba_calibrated.mean():.3f}")
        
        # Prepare score_predicted_all if available
        score_predicted_all = None
        if "score_predicted" in data.columns:
            score_predicted_all = pd.to_numeric(data["score_predicted"], errors="coerce").fillna(0.5).values
        
        # Compute last bin errors on labeled data to select best score
        best_proba = _select_best_probability_score(labeled_with_true, rf, calibrator, proba_calibrated, proba_uncalibrated, score_predicted_all)
        
        # Uncertainty = 1 - probability of being correct
        # This means uncertain predictions (low prob_correct) get high uncertainty
        uncertainty = 1.0 - best_proba
        
        logger.info(f"[TIMING] calibration_uncertainty inference={time.time() - t_apply_start:.3f}s")
        logger.info(f"[TIMING] calibration_uncertainty total={time.time() - t_start:.3f}s")
        
        # Log statistics
        logger.info(
            f"[calibration_uncertainty] Score range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}], "
            f"mean={uncertainty.mean():.3f}"
        )
        
        return pd.Series(index=data.index, data=uncertainty)
        
    except Exception as e:
        logger.error(f"[calibration_uncertainty] Error during computation: {e}", exc_info=True)
        # Return fallback
        if "score_predicted" in data.columns:
            return 1.0 - data["score_predicted"].fillna(0.5)
        return pd.Series(index=data.index, data=0.5)


def compute_and_add_calibrated_uncertainty(
    data: pd.DataFrame,
    labeled_indices: Optional[np.ndarray] = None,
    column_name: str = "calibrated_uncertainty",
) -> None:
    """
    Compute calibrated uncertainty and add it to the dataframe in-place.
    
    Args:
        data: DataFrame to modify in-place
        labeled_indices: Optional array of labeled indices
        column_name: Name of the column to create/update (default: "calibrated_uncertainty")
    """
    t_start = time.time()
    
    uncertainty = compute_calibrated_uncertainty(data, labeled_indices)
    data[column_name] = uncertainty
    
    logger.info(f"[TIMING] {column_name} total (with assignment)={time.time() - t_start:.3f}s")
    
    # Log distribution histogram for all data (not just labeled)
    if column_name in data.columns:
        unc = data[column_name].dropna()
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, _ = np.histogram(unc, bins=bins)
        labeled_mask = data.index.isin(labeled_indices) if labeled_indices is not None else pd.Series(False, index=data.index)
        labeled_unc = data.loc[labeled_mask, column_name].dropna()
        unlabeled_unc = data.loc[~labeled_mask, column_name].dropna()
        
        logger.info(f"[{column_name}] Distribution (all rows): {hist}, total={len(unc)}")
        logger.info(f"[{column_name}] Labeled rows: {len(labeled_unc)}, mean={labeled_unc.mean():.3f}")
        logger.info(f"[{column_name}] Unlabeled rows: {len(unlabeled_unc)}, mean={unlabeled_unc.mean():.3f}")
        
        # Count likely certain (uncertainty < 0.1) and likely uncertain (uncertainty > 0.9)
        likely_certain_all = (unc < 0.1).sum()
        likely_uncertain_all = (unc > 0.9).sum()
        middle_range_all = ((unc >= 0.1) & (unc <= 0.9)).sum()
        logger.info(f"[{column_name}] likely certain (<0.1): {likely_certain_all}, middle [0.1-0.9]: {middle_range_all}, likely uncertain (>0.9): {likely_uncertain_all}")
        
        # Compare /simple vs /annflux counting logic
        unlabeled_mask = ~labeled_mask
        if "score_predicted" in data.columns:
            score_pred_numeric = pd.to_numeric(data["score_predicted"], errors="coerce")
            # /simple logic: calibrated_uncertainty < 0.1 and unlabeled
            simple_certain = ((unc < 0.1) & unlabeled_mask).sum()
            # /annflux logic: score_predicted > 0.90 and num_labeled_nn > 1 and unlabeled
            has_num_labeled_nn = "num_labeled_nn" in data.columns
            if has_num_labeled_nn:
                num_labeled_nn = pd.to_numeric(data["num_labeled_nn"], errors="coerce")
                annflux_certain = ((score_pred_numeric > 0.90) & (num_labeled_nn > 1) & unlabeled_mask).sum()
            else:
                annflux_certain = ((score_pred_numeric > 0.90) & unlabeled_mask).sum()
            if has_num_labeled_nn:
                overlap_count = ((unc < 0.1) & (score_pred_numeric > 0.90) & unlabeled_mask).sum()
            else:
                overlap_count = "N/A"
            logger.info(f"[UI_ALIGNMENT] /simple likely certain: {simple_certain}, /annflux likely certain: {annflux_certain}, overlap: {overlap_count}")
