# AnnFlux Architecture Documentation

## Overview

AnnFlux is a research tool for exploring and annotating large datasets with Active Learning. It provides a standalone tool with a basic interface for efficient dataset exploration and annotation, designed with an extensible architecture to allow researchers to contribute functionality.

## Core Philosophy

The architecture follows these design principles:

- **Extensibility**: Components are designed to be easily modified or replaced
- **Common Data Formats**: Communication uses standard formats (NumPy arrays, Pandas DataFrames, CSV, Parquet)
- **Simplicity**: Complex internal communication or slow execution signals need for redesign
- **Research-Oriented**: Built for experimental machine learning workflows

## High-Level Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Layer     │    │   Web UI Layer  │    │  Training Layer │
│                 │    │                 │    │                 │
│ annflux_cli.py  │    │ run_server.py   │    │ clip.py         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                    Core Layer                                  │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │ Repository  │  │ Algorithms  │  │    Tools/Utils      │    │
│  │             │  │             │  │                     │    │
│  │ dataset.py  │  │ embeddings  │  │ core.py (state)     │    │
│  │ model.py    │  │ most_needed │  │ data.py             │    │
│  │ resultset.py│  │ fastdpeak   │  │ io.py               │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
└─────────────────────────────────┼─────────────────────────────────┘
                                 │
┌─────────────────────────────────┼─────────────────────────────────┐
│                   Data Layer                                   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │   Images    │  │  Features   │  │   Annotations       │    │
│  │             │  │             │  │                     │    │
│  │ .jpg/.png   │  │ .npz files  │  │ CSV/Parquet         │    │
│  │ Video frames│  │ Embeddings  │  │ Labels              │    │
│  └─────────────┘  └─────────────┘  └─────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. CLI Layer (`src/annflux/scripts/`)

**Primary Entry Point**: `annflux_cli.py`

- **Commands**: `go`, `train_then_features`, `export_model`
- **Workflow orchestration**: Initializes projects, computes features, manages training
- **Data preprocessing**: Video frame extraction, image tiling, dataset preparation
- **Model management**: Export and deployment utilities

**Key Scripts**:
- `annflux_cli.py`: Main CLI interface with argparse-based commands
- `extract_video_frames.py`: Video processing for frame extraction
- `tile_images.py`: Image tiling for large dataset processing
- `from_class_folder.py`: Dataset initialization from class-organized folders

### 2. Web UI Layer (`src/annflux/ui/basic/`)

**Primary Server**: `run_server.py`

- **Flask-based web interface**: RESTful API for dataset interaction
- **Real-time annotation**: Interactive labeling interface
- **Visualization**: 2D embeddings, clustering, active learning suggestions
- **Data streaming**: Parquet-based efficient data transfer

**Architecture**:
- Flask web server with CORS support
- Arrow/Parquet for efficient data streaming
- Real-time state management through DuckDB
- Thumbnail generation and caching

### 3. Training Layer (`src/annflux/training/`)

#### AnnFlux Training (`src/annflux/training/annflux/`)

**Core Components**:
- `clip.py`: CLIP-based feature extraction with PEFT fine-tuning
- `feature_extractor.py`: Abstract base class for feature extractors
- `quick.py`: Fast retraining and kNN-based predictions
- `clip_server.py`: Model deployment as web service

**Model Support**:
- OpenAI CLIP variants
- BioCLIP and BioCLIP2 for biological data
- Parameter Efficient Fine-Tuning (PEFT) support

#### TensorFlow Backend (`src/annflux/training/tensorflow/`)

- `tf_backend.py`: Linear models and TensorFlow utilities

### 4. Repository Layer (`src/annflux/repository/`)

**Core Classes**:
- `Repository`: Version control for ML objects
- `Dataset`: Immutable data selections with metadata
- `Model`: Trained model representations
- `Resultset`: Model predictions on datasets

**Design Pattern**: Git-like versioning for ML artifacts
- Ancestry tracking between datasets, models, and results
- Provenance and reproducibility
- JSON-based metadata storage

### 5. Algorithm Layer (`src/annflux/algorithms/`)

**Active Learning Algorithms**:
- `most_needed.py`: Sample selection for efficient annotation
- `feature_reconstruction_error.py`: Uncertainty-based sampling
- `fastdpeak.py`: Density-peak clustering
- `fastdpeak_merge.py`: Peak merging for active learning

**Core Algorithms**:
- `embeddings.py`: Dimensionality reduction (UMAP, t-SNE)
- `basic_ml.py`: Basic ML utilities (softmax, etc.)

### 6. Tools Layer (`src/annflux/tools/`)

**Core Utilities**:
- `core.py`: Annotation state management
- `data.py`: AnnFlux-specific data operations
- `io.py`: File I/O and format conversion
- `mixed.py`: General utilities including logging

## Data Flow Architecture

### Primary Workflow

```
1. Dataset Initialization
   ├── Images/Video Files
   ├── Feature Extraction (CLIP)
   └── Embedding Computation

2. Annotation Phase
   ├── Web UI Interaction
   ├── Active Learning Suggestions
   └── Label Assignment

3. Training Phase
   ├── Model Fine-tuning (PEFT)
   ├── Feature Recomputation
   └── Performance Evaluation

4. Deployment
   ├── Model Export
   └── Service Deployment
```

### Data Formats

| Component | Input Format | Output Format | Storage |
|-----------|--------------|---------------|---------|
| Feature Extraction | Images (.jpg, .png) | Features (.npz) | File system |
| Embeddings | Features (.npz) | 2D Coordinates | DuckDB |
| Annotations | Web UI | Labels (CSV/Parquet) | DuckDB |
| Models | Training Data | Model Files | Repository |

## State Management

### Annotation State (`src/annflux/tools/core.py`)

**State Table Schema**:
- `dp_most_needed`: Density-peak based active learning scores
- `dp_is_ldp`: Density peak indicators
- `dp_depth`: Clustering depth information
- `label_predicted`: Model predictions
- `label_possible`: Human annotations

### Persistence Strategy

- **DuckDB**: For structured queryable data (annotations, embeddings)
- **File System**: For large binary data (features, models)
- **Repository**: For ML artifact versioning and provenance

## Performance Architecture

### Scalability Features

- **Parquet Streaming**: Efficient browser data transfer
- **Thumbnail Caching**: Fast image preview generation
- **Batch Processing**: Vectorized feature computation
- **Memory Management**: Streaming for large datasets

### Optimization Strategies

- **FAISS**: Fast similarity search for embeddings
- **Numba**: JIT compilation for numerical algorithms
- **Async Processing**: Non-blocking UI operations
- **Lazy Loading**: On-demand data loading

## Extension Points

### Custom Feature Extractors

```python
class CustomFeatureExtractor(FeatureExtractor):
    def extract_features(self, dataset: Dataset) -> NDArray:
        # Custom implementation
        pass
```

### Custom Active Learning

```python
def custom_active_learning_score(embeddings: NDArray) -> NDArray:
    # Custom scoring algorithm
    pass
```

### Custom UI Components

- Flask route extensions
- JavaScript frontend components
- Custom visualization methods

## Security Architecture

- **Authentication**: Flask-HTTPAuth for access control
- **CORS**: Cross-origin resource sharing configuration
- **Rate Limiting**: Request throttling with Flask-Limiter
- **Input Validation**: File type and size restrictions

## Deployment Architecture

### Development Mode
```bash
annflux go {PROJECT_FOLDER}
basic_ui  # Starts development server
```

### Production Deployment
```bash
# Model export
annflux export_model {PROJECT_FOLDER}

# Service deployment
python clip_server.py {MODEL_PACKAGE}
```

## Configuration Management

### Environment Variables
- `HUGGINGFACE_CLIP_NAME`: CLIP model selection
- `USER_DATASET_PATH`: Dataset location
- `TF_CPP_MIN_LOG_LEVEL`: TensorFlow logging

### Project Structure
```
{PROJECT_FOLDER}/
├── images/           # Source images
├── annflux/          # AnnFlux data
│   ├── repository/   # ML artifacts
│   ├── state.db      # DuckDB database
│   └── cache/        # Temporary files
└── config.json       # Project configuration
```

## Technology Stack

### Core Dependencies
- **PyTorch**: Deep learning framework
- **TensorFlow**: Alternative backend
- **Flask**: Web framework
- **DuckDB**: Analytical database
- **FAISS**: Similarity search
- **UMAP**: Dimensionality reduction

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **OpenCV**: Computer vision
- **Pillow**: Image processing
- **PyArrow**: Columnar data format

### ML/AI
- **Transformers**: Hugging Face models
- **PEFT**: Parameter efficient fine-tuning
- **scikit-learn**: Traditional ML algorithms

## Integration Patterns

### Plugin Architecture
- Algorithm registration through decorators
- Dynamic feature extractor loading
- Configurable model architectures

### API Design
- RESTful endpoints for UI interaction
- Streaming responses for large datasets
- JSON-based configuration

### Error Handling
- Graceful degradation for missing components
- Comprehensive logging throughout system
- User-friendly error messages in UI

This architecture enables researchers to efficiently explore and annotate large datasets while maintaining extensibility for custom algorithms and workflows.
