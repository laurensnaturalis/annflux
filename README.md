![AnnFlux Logo](src/annflux/ui/basic/static/annflux.png)

# [AnnFlux] Annotation flux
## A research tool for exploring and annotating large datasets with Active Learning 

This standalone tool provides a basic interface for interacting with large datasets so that they can be explored and annotated efficiently. 

The extensible design of the tool allows researchers from in- and outside Intel to contribute to the development of the functionality

# License 

Apache 2.0, see [LICENSE.md](LICENSE)

# Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md)

# Install

1. Create and activate a new Conda environment:

   ```bash
   conda create -n AnnFlux_dev python=3.11
   conda activate AnnFlux_dev
   ```

2. Install the requirements:

   ```bash
   pip install -e .
   ```
# Examples

## StreetSurfaceVis

The public (Creative Commons Attribution Share Alike 4.0 International) road type dataset 'StreetSurfaceVis' (https://zenodo.org/records/11449977 / https://www.nature.com/articles/s41597-024-04295-9).

See [StreetSurfaceVis](src/annflux/examples/streetsurfacevis.md)

## From images folder    
## Init project, compute features and embed

`annflux go {PROJECT_FOLDER}`

`PROJECT_FOLDER` should have at least a `images` folder

## Example

Use an image dataset with a folder of images with a .jpg extension. A good size is 5,000 to 10,000 images.

Extract to `~/annflux/data/envdataset/images`

On commandline

```bash
annflux go ~/annflux/data/envdataset --start_labels Your_label_A Your_label_B Your_label_C`
```

Label some images, then

```bash
annflux train_then_features ~/annflux/data/envdataset
```

to perform parameter efficient fine-tuning of the (default) CLIP model, followed by computation of the adapted features.

# Tests and code coverage

```bash
export HUGGINGFACE_CLIP_NAME={a hugging face CLIP model that supports the peft package}
```

```bash
export USER_DATASET_PATH={your project folder with an 'images' folder inside}
```

```bash
python run_coverage.py
```


# Known issues

HIGH 
- (None)

# Changelog

## 1.3.1.1

- Fix NN example images fullscreen overlay (handle both uid string and data object in `openOverlay`)
- Fix test output parser to handle errors and seconds-only time format
- Lazy-import umap to avoid TensorFlow/ParametricUMAP import error

## 1.3.1.0

- BioClip2 PEFT (LoRA) training support via `open_clip`
- Full-screen image overlay uses `image_url` from dataset when available
- Dynamic column ordering in data head (fallback from `fre_strat` to `score_predicted` to `uid`)
- Reduced parquet size for `/simple` by sending only needed columns
- Annotation logging per project
- File utility: `append_to_filename`
- `image_url` support in map view (`annflux.js`)

## 1.3.0.0

- Calibration model with SHAP explanations and train/val/test split
- Calibrated uncertainty selection for active learning
- Removed TensorFlow/Keras dependency
- Atomic `labels.json` writes to prevent corruption
- Stacked progress bar with labeled/certain/uncertain segments
- Certainty dropdown, multilabel examples, and prediction probabilities in `/simple`
- Debounce and image loading prevention while typing in filter
- Selenium test for `/simple` endpoint
- Fix label picker to show descendants per row, prioritizing most specific label first
- Fix calibration uncertainty NaN/Inf handling
- Fix save button logic and partial label sort option
- Clean up old parquet files before creating new cache

## 1.2.0.0

- Simple annotator UI at `/simple` endpoint
- Hierarchy-aware label assignment with label search and parent sorting
- Inline child label creation from chip + button
- Async `/label` endpoint with file locking
- Visible metadata and annotator UX improvements
- Class examples column
- Performance: vectorise `make_predictions`, `compute_fre`, `color_and_label`, `compute_near_labeled`, PCA
- Fix multi-label normalisation in `make_predictions`
- Fix FRE coloring NaN crash
- KNN prediction tests
- Separate log files per component

## 1.1.0.0

- Added density peak based display order and most needed computation
- Support for tiling
- Preliminary support for video
- Support for instance & group embeddings
- Full image display
- Preliminary streaming support
- Data is now transferred using Parquet in the browser
- Many small improvements

## 1.0.3.0

- Fix for distance_weight==0 in quick.py
- Reset local activeTime when starting new browser session

## 1.0.2.0

- Improved reporting on when label save fails

## 1.0.1.0

- Basic UI contribution from Naturalis

## 1.0.0.0

- Final first release

## 0.9.3.0

- Improved test reporting

## 0.9.3.0

- Improved test reporting

## 0.9.2.0

- Preparing for open source

# UI documentation

# Filter
Server side filtering of dataset

blurry-or-low-res NOT IN row.label_possible AND blurry-or-low-res NOT IN row.label_predicted