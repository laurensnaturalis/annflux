#!/usr/bin/env python3
"""
Train a RandomForest model to estimate the probability that a prediction is correct.

Uses labeled rows from annflux.csv to train a binary classifier that predicts
P(correct | features), where correctness is determined by exact match between
label_predicted and label_true.

Feature Explanation Methods Available:
- SHAP (SHapley Additive exPlanations): Global and per-instance feature contributions
- Feature Importance: Built-in RandomForest Gini importance (mean impurity decrease)
- Permutation Importance: Can be computed by shuffling features and measuring performance drop
- Partial Dependence Plots: Show how changing a feature affects predictions

Output: Saved model and calibration analysis.
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    log_loss,
    classification_report,
    confusion_matrix,
)


# Configuration
PROJECT_PATH = "/mnt/big/indeed/lepisea"
ANNFLUX_CSV = os.path.join(PROJECT_PATH, "annflux", "annflux.csv")
OUTPUT_DIR = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(OUTPUT_DIR, "calibration_rf_model.pkl")

# Random Forest hyperparameters
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load annflux.csv with proper dtypes."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={"score_possible": str, "scores_predicted": str})
    print(f"Loaded {len(df)} rows")
    return df


def is_correct_prediction(row: pd.Series) -> bool:
    """
    Determine if a prediction is correct.
    
    For multi-label: returns True only if ALL predicted labels match ALL true labels.
    """
    pred = row.get("label_predicted")
    true = row.get("label_true")
    
    if pd.isna(pred) or pd.isna(true):
        return False
    
    # Normalize: split by comma, strip whitespace, sort for comparison
    pred_set = set(str(pred).split(","))
    true_set = set(str(true).split(","))
    
    # Remove empty strings
    pred_set = {p.strip() for p in pred_set if p.strip()}
    true_set = {t.strip() for t in true_set if t.strip()}
    
    return pred_set == true_set


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract numeric features from annflux dataframe.
    
    Excludes:
    - String/id columns (uid, path, etc.)
    - Target columns (label_true, label_predicted, labeled)
    - Columns with non-numeric data
    """
    # Columns to exclude
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
        "certainty",  # if exists
        "correct",  # target variable we create during training
        "calibrated_prob_correct",  # model output, not a feature
        "incorrect_score"  # only for labeled data
    }
    
    feature_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        # Only keep numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)
    
    print(f"Using {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Replace infinite values with large finite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Clip extreme values to avoid overflow
    for col in X.columns:
        col_max = X[col].abs().max()
        if col_max > 1e10:
            X[col] = X[col].clip(lower=-1e10, upper=1e10)
    
    return X


def train_calibration_model(df: pd.DataFrame) -> RandomForestClassifier:
    """
    Train RandomForest to predict correctness probability.
    
    Only uses labeled rows (where label_true is not null).
    """
    # Filter to labeled rows only
    labeled_df = df[df["labeled"] == 1].copy()
    labeled_df = labeled_df[~pd.isna(labeled_df["label_true"])]
    
    print(f"\nTraining on {len(labeled_df)} labeled rows")
    
    # Create target variable
    labeled_df["correct"] = labeled_df.apply(is_correct_prediction, axis=1)
    
    y = labeled_df["correct"].astype(int)
    print(f"Class distribution: {y.value_counts().to_dict()}")
    print(f"Accuracy on labeled data: {y.mean():.3f}")
    
    # Extract features
    X = extract_features(labeled_df)
    
    # Proper 3-way split: train (70%) / validation (15%) / test (15%)
    # First split: separate test set (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Second split: separate validation set from remaining (15% of total = ~17.6% of temp)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    
    print(f"\nData Split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train RandomForest on training set
    print("\nTraining RandomForest on training set...")
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    # Validate on validation set
    print("\n=== Validation Set Performance ===")
    y_val_proba = rf.predict_proba(X_val)[:, 1]
    y_val_pred = rf.predict(X_val)
    print(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    print(f"Brier Score: {brier_score_loss(y_val, y_val_proba):.4f}")
    
    # Final evaluation on held-out test set
    print("\n=== Held-Out Test Set Performance ===")
    y_test_proba = rf.predict_proba(X_test)[:, 1]
    y_test_pred = rf.predict(X_test)
    
    print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    print(f"Brier Score: {brier_score_loss(y_test, y_test_proba):.4f}")
    print(f"Log Loss: {log_loss(y_test, y_test_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Incorrect", "Correct"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Feature importance from model trained on train set
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    
    print("\nTop 10 Feature Importances (from training set model):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Calibrate using Isotonic Regression on validation set
    print("\n=== Calibration Step ===")
    print("Fitting Isotonic Regression on validation set...")
    
    # Get uncalibrated probabilities on validation set
    y_val_proba_uncal = rf.predict_proba(X_val)[:, 1]
    
    # Fit isotonic regression: maps uncalibrated -> calibrated
    isotonic_calibrator = IsotonicRegression(out_of_bounds="clip")
    isotonic_calibrator.fit(y_val_proba_uncal, y_val)
    
    # Test calibration on held-out test set
    y_test_proba_uncal = rf.predict_proba(X_test)[:, 1]
    y_test_proba_cal = isotonic_calibrator.predict(y_test_proba_uncal)
    
    print(f"Test Brier (uncalibrated): {brier_score_loss(y_test, y_test_proba_uncal):.4f}")
    print(f"Test Brier (calibrated):   {brier_score_loss(y_test, y_test_proba_cal):.4f}")
    
    # Retrain on combined train+validation for production (test set NEVER used)
    print("\nRetraining on train+validation sets for production...")
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    rf_final = RandomForestClassifier(**RF_PARAMS)
    rf_final.fit(X_train_val, y_train_val)
    
    # Re-fit calibrator on full train+val set for production
    y_train_val_proba = rf_final.predict_proba(X_train_val)[:, 1]
    calibrator_final = IsotonicRegression(out_of_bounds="clip")
    calibrator_final.fit(y_train_val_proba, y_train_val)
    
    return rf_final, calibrator_final, feature_importance


def predict_all(df: pd.DataFrame, model, calibrator=None) -> pd.DataFrame:
    """Apply model to all rows to get correctness probability estimates."""
    X = extract_features(df)
    
    # Get probability of being correct (class 1)
    proba_uncalibrated = model.predict_proba(X)[:, 1]
    
    # Apply calibration if calibrator provided
    if calibrator is not None:
        proba_correct = calibrator.predict(proba_uncalibrated)
    else:
        proba_correct = proba_uncalibrated
    
    df = df.copy()
    df["calibrated_prob_correct"] = proba_correct
    df["uncalibrated_prob_correct"] = proba_uncalibrated
    
    return df


def analyze_calibration(df_with_proba: pd.DataFrame):
    """Analyze calibration of the probability estimates on labeled data."""
    labeled = df_with_proba[df_with_proba["labeled"] == 1].copy()
    labeled["correct"] = labeled.apply(is_correct_prediction, axis=1)
    
    print("\n=== Calibration Analysis ===")
    
    # Binned calibration
    bins = np.arange(0, 1.1, 0.1)
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        mask = (labeled["calibrated_prob_correct"] >= lower) & (
            labeled["calibrated_prob_correct"] < upper
        )
        subset = labeled[mask]
        if len(subset) > 0:
            actual_acc = subset["correct"].mean()
            print(
                f"Prob [{lower:.1f}, {upper:.1f}): "
                f"n={len(subset)}, actual_acc={actual_acc:.3f}"
            )
    
    # Correlation between predicted probability and actual correctness
    corr = labeled["calibrated_prob_correct"].corr(labeled["correct"])
    print(f"\nCorrelation (prob vs actual): {corr:.4f}")
    
    # Compare with raw score_predicted
    print("\n=== Comparison with score_predicted ===")
    labeled["score_prob"] = labeled["score_predicted"].clip(0, 1)
    
    print("\nRaw score_predicted calibration:")
    for i in range(len(bins) - 1):
        lower, upper = bins[i], bins[i + 1]
        mask = (labeled["score_prob"] >= lower) & (labeled["score_prob"] < upper)
        subset = labeled[mask]
        if len(subset) > 0:
            actual_acc = subset["correct"].mean()
            print(
                f"Score [{lower:.1f}, {upper:.1f}): "
                f"n={len(subset)}, actual_acc={actual_acc:.3f}"
            )
    
    score_corr = labeled["score_prob"].corr(labeled["correct"])
    print(f"\nCorrelation (score_predicted vs actual): {score_corr:.4f}")
    print(f"Model improvement: {corr - score_corr:+.4f}")


def explain_features_with_shap(model, X_sample, n_samples=100):
    """
    Compute SHAP values for feature explanations.
    
    Shows:
    - Global feature importance (mean |SHAP|)
    - Example contributions for a few individual predictions
    """
    try:
        import shap
    except ImportError:
        print("\n[SHAP not installed, skipping feature explanations]")
        print("Install with: pip install shap")
        return
    
    print(f"\n=== Feature Explanations (SHAP) ===")
    print(f"Computing SHAP values for {n_samples} samples...")
    
    # Sample for efficiency
    X_explain = X_sample.head(n_samples) if len(X_sample) > n_samples else X_sample
    
    # Use columns from X_explain directly
    feature_names = X_explain.columns.tolist()
    
    # TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    
    # Handle different SHAP output formats
    # Format 1: List of arrays [class_0_values, class_1_values]
    # Format 2: 3D array (samples, features, classes)
    if isinstance(shap_values, list):
        # List format - extract class 1
        shap_values = np.asarray(shap_values[1])  # (samples, features)
    else:
        # 3D array format - extract class 1
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]  # (samples, features)
    
    # Global importance: mean absolute SHAP value
    mean_shap = np.abs(shap_values).mean(axis=0)  # (features,)
    
    print(f"DEBUG: shap_values shape={shap_values.shape}, mean_shap shape={mean_shap.shape}, len(features)={len(feature_names)}")
    
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    
    print("\nTop 10 Features by Mean |SHAP| (global importance):")
    print(shap_importance.head(10).to_string(index=False))
    
    # Example contributions for 3 individual predictions
    print("\n--- Example Contributions (first 3 predictions) ---")
    for i in range(min(3, len(X_explain))):
        print(f"\nPrediction {i+1}:")
        # Top contributing features for this instance
        instance_shap = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_values[i],
            "feature_value": X_explain.iloc[i].values,
        })
        instance_shap["abs_shap"] = instance_shap["shap_value"].abs()
        instance_shap = instance_shap.sort_values("abs_shap", ascending=False)
        
        print("  Top contributors:")
        for _, row in instance_shap.head(5).iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            print(f"    {row['feature']}={row['feature_value']:.3f} "
                  f"({direction} P(correct) by {abs(row['shap_value']):.3f})")


def save_model(model, calibrator, output_path: str):
    """Save trained model and calibrator to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_bundle = {
        "rf_model": model,
        "calibrator": calibrator,
    }
    with open(output_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\nModel and calibrator saved to {output_path}")


def main():
    # Load data
    df = load_data(ANNFLUX_CSV)
    
    # Check if we have labeled data
    n_labeled = (df["labeled"] == 1).sum()
    n_with_true = (~pd.isna(df["label_true"])).sum()
    
    print(f"\nDataset stats:")
    print(f"  Total rows: {len(df)}")
    print(f"  Labeled (labeled=1): {n_labeled}")
    print(f"  With label_true: {n_with_true}")
    
    if n_labeled < 100:
        print("\nERROR: Need at least 100 labeled rows to train. Exiting.")
        return
    
    # Train model (returns RF + Isotonic calibrator)
    model, calibrator, feature_importance = train_calibration_model(df)
    
    # Apply to all data with calibration
    df_with_proba = predict_all(df, model, calibrator)
    
    # Analyze calibration (now using calibrated probabilities)
    analyze_calibration(df_with_proba)
    
    # Feature explanations (SHAP) - optional, requires shap package
    X_full = extract_features(df_with_proba)
    explain_features_with_shap(model, X_full)
    
    # Save model with calibrator
    save_model(model, calibrator, MODEL_PATH)
    
    # Save predictions with probabilities
    output_csv = os.path.join(OUTPUT_DIR, "annflux_with_calibration.csv")
    df_with_proba.to_csv(output_csv, index=False)
    print(f"\nData with calibration probabilities saved to {output_csv}")


if __name__ == "__main__":
    main()
