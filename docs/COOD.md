# COOD: Combined Out-of-Distribution Detection

Implementation of the paper "Combined Out-of-distribution Detection Using Multiple Measures for Anomaly" (Hogeweg et al., CVPRW 2024) for the AnnFlux annotation tool.

## Overview

COOD combines multiple individual OOD (Out-of-Distribution) detection measures using a RandomForest classifier to produce a unified OOD score. This approach significantly outperforms individual OOD detection methods by leveraging the complementary strengths of different measures.

## Key Features

- **19 Individual OOD Measures**: Combines multiple detection strategies including:
  - Linear classifier-based measures (Max linear, temperature-scaled)
  - kNN-based measures (FRE, neighbor distances)
  - Entropy-based measures (prediction entropy, score distribution entropy)
  - Distance-based measures (underrepresented detection, most-needed scores)
  - Uncertainty calibration measures

- **Multi-class Classification**: Distinguishes between:
  - ID-correct: In-distribution, correctly classified
  - ID-incorrect: In-distribution, misclassified
  - OOD: Out-of-distribution samples

- **RandomForest Combination**: Uses supervised learning to optimally combine individual measures

## Installation

COOD is integrated into AnnFlux and requires no additional installation beyond the standard dependencies:

```bash
pip install scikit-learn pandas numpy
```

## Usage

### Automatic Integration

COOD is automatically computed during the quick reclassification pipeline when enabled:

```bash
ENABLE_COOD=1 python -m annflux.ui.basic.run_server /path/to/project
```

Set `ENABLE_COOD=0` to disable COOD computation (saves computation time).

### Manual Usage

```python
from annflux.algorithms.cood import compute_cood, add_cood_to_data

# Compute COOD scores for a dataframe
cood_results = compute_cood(
    data=df,
    features=feature_embeddings,  # Optional: feature embeddings
    train_on_labeled=True,  # Train classifier on labeled samples
)

# Or add directly to existing dataframe
df_with_cood = add_cood_to_data(df, features=feature_embeddings)
```

### Output Columns

The COOD computation adds the following columns to your dataframe:

- `cood_score`: Combined OOD score (0 = ID, 1 = OOD)
- `cood_class`: Predicted class (0=ID-correct, 1=ID-incorrect, 2=OOD)
- `cood_*`: Individual OOD measure values (for debugging/analysis)

## Configuration

### Environment Variables

- `ENABLE_COOD`: Enable/disable COOD computation (default: "1")
  - "1": Enabled
  - "0": Disabled

### RandomForest Parameters

Default parameters (can be customized in `cood.py`):

```python
COOD_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
```

## Implementation Details

### Individual Measures

The implementation includes these individual OOD measures:

1. **Linear Measures**:
   - `max_linear`: Maximum softmax probability
   - `max_linear_t_scaled`: Temperature-scaled softmax (T=0.5)
   - `linear_margin`: Difference between top two class probabilities

2. **kNN Measures**:
   - `fre`: Feature Reconstruction Error
   - `num_labeled_nn`: Number of labeled nearest neighbors
   - `min_distance`: Minimum distance to nearest neighbor

3. **Entropy Measures**:
   - `entropy`: Entropy of prediction distribution
   - `scores_entropy`: Entropy of full score distribution

4. **Distance/AL Measures**:
   - `nn_underrepresented`: Underrepresented detection score
   - `most_needed`: Active learning "most needed" score
   - `calibrated_uncertainty`: Calibration-based uncertainty

### Training

The classifier is trained on labeled samples using:
- Features: Individual OOD measures
- Target: Correctness (ID-correct vs ID-incorrect vs OOD)

Training automatically handles:
- Class imbalance (using `class_weight="balanced"`)
- Missing features (filled with 0)
- Small sample sizes (minimum 10 samples required)

## Performance

Typical computation time:
- Individual measures: ~0.5s for 10k samples
- RandomForest training: ~1-2s for 100+ labeled samples
- Total COOD computation: ~2-3s for 10k samples

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{hogeweg2024cood,
  title={Combined Out-of-distribution Detection Using Multiple Measures for Anomaly},
  author={Hogeweg, Laurens and others},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024}
}
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_cood.py -v
```

## Troubleshooting

### COOD scores are all zeros
- Check that individual OOD measures are available in your data
- Ensure you have labeled samples for training (minimum 10)
- Verify `ENABLE_COOD=1` environment variable

### Slow computation
- COOD is only computed during reclassification, not per-request
- Disable with `ENABLE_COOD=0` if not needed

### Feature importance
- Enable debug logging to see top contributing features
- Check that your data has the expected columns

## Related Papers

- Deep k-Nearest Neighbors (Papernot & McDaniel, 2018)
- Out-of-Distribution Detection with Deep Nearest Neighbors (Sun et al., 2022)
- A Unifying Review of Deep and Shallow Anomaly Detection (Ruff et al., 2021)
