# Subsystem 5: Deep Learning Model Training and Comparison

## 📋 Overview

This subsystem implements a complete deep learning pipeline for training and comparing multiple neural network models. It includes training CNNs from scratch, fine-tuning pre-trained models, comprehensive performance evaluation, and interactive visualization dashboards.

---

## 🎯 Objectives

1. **Train CNN from Scratch**: Build and train a custom Convolutional Neural Network with cross-validation
2. **Fine-Tuning**: Implement transfer learning using pre-trained models (ResNet50, MobileNetV2, VGG16, InceptionV3)
3. **Model Comparison**: Compare performance metrics across all models
4. **Visualization**: Generate comprehensive plots and interactive dashboards
5. **Metrics Analysis**: Detailed analysis of accuracy, precision, recall, AUC, and loss

---

## 🏗️ Architecture

### System Components

```
subsystem_5/
├── python/
│   └── training/
│       ├── cnn_from_scratch.py       # Custom CNN training
│       ├── fine_tuning.py            # Transfer learning
│       ├── compare_models.py         # Model comparison
│       └── dashboard.py              # Interactive Streamlit dashboard
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Preprocessed data
├── results/
│   ├── models/                       # Saved models (.h5 files)
│   ├── plots/                        # Generated visualizations
│   └── metrics/                      # Performance metrics (JSON, CSV)
└── docs/
    ├── README.md                     # This file
    ├── ARCHITECTURE.md               # Detailed architecture
    └── METRICAS.md                   # Metrics documentation
```

### Data Flow

```
Raw Data → Preprocessing → Training → Evaluation → Visualization
    ↓           ↓              ↓           ↓            ↓
  CIFAR-10   Resizing     CNN Models   Metrics    Dashboards
             Normalize    Fine-tuned   JSON/CSV    Plots/GIFs
             Augment      Models       Reports     Interactive
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn streamlit plotly
```

### 1. Train CNN from Scratch

```bash
cd python/training
python cnn_from_scratch.py
```

**Features:**
- Custom CNN architecture with 4 convolutional blocks
- Batch normalization and dropout layers
- K-Fold cross-validation (optional)
- Early stopping and learning rate reduction
- Comprehensive metrics tracking

**Output:**
- Trained model: `results/models/cnn_scratch_*.h5`
- Training plots: `results/plots/training_history.png`
- Metrics: `results/metrics/cnn_scratch_*_metrics.json`

### 2. Fine-Tune Pre-trained Models

```bash
python fine_tuning.py
```

**Available Models:**
1. ResNet50
2. MobileNetV2
3. VGG16
4. InceptionV3

**Training Strategy:**
- **Phase 1**: Feature extraction (freeze base model, train top layers)
- **Phase 2**: Fine-tuning (unfreeze last N layers, train entire model)

**Output:**
- Feature extraction model: `results/models/{model}_feature_extraction_best.h5`
- Fine-tuned model: `results/models/{model}_fine_tuned_best.h5`
- Final model: `results/models/{model}_final.h5`
- Training plots: `results/plots/{model}_training_history.png`
- Metrics: `results/metrics/{model}_metrics.json`

### 3. Compare Models

```bash
python compare_models.py
```

**Generated Visualizations:**
- Metrics comparison bar charts
- Radar chart (multi-metric view)
- ROC curves
- Confusion matrices
- Precision vs Recall scatter plot
- Comprehensive summary dashboard

**Output:**
- All comparison plots in `results/plots/`
- Comparison table: `results/metrics/models_comparison.csv`

### 4. Launch Interactive Dashboard

```bash
streamlit run dashboard.py
```

**Dashboard Features:**
- **Overview Tab**: Key metrics, radar charts, heatmaps
- **Detailed Metrics Tab**: Individual metric comparisons
- **Comparisons Tab**: Side-by-side model comparison
- **Raw Data Tab**: Tables, CSV download, JSON details

**Access:** http://localhost:8501

---

## 📊 Models

### 1. CNN from Scratch

**Architecture:**
```python
Input (128x128x3)
    ↓
Conv Block 1 (32 filters) → BN → Conv (32) → BN → MaxPool → Dropout(0.25)
    ↓
Conv Block 2 (64 filters) → BN → Conv (64) → BN → MaxPool → Dropout(0.25)
    ↓
Conv Block 3 (128 filters) → BN → Conv (128) → BN → MaxPool → Dropout(0.25)
    ↓
Conv Block 4 (256 filters) → BN → Conv (256) → BN → MaxPool → Dropout(0.25)
    ↓
Flatten → Dense(512) → BN → Dropout(0.5) → Dense(256) → BN → Dropout(0.5)
    ↓
Output (10 classes, softmax)
```

**Parameters:**
- Total: ~2.5M parameters
- Trainable: ~2.5M
- Optimizer: Adam (lr=0.001)
- Loss: Categorical crossentropy

### 2. ResNet50 (Fine-tuned)

**Architecture:**
```python
Input (224x224x3)
    ↓
ResNet50 Base (frozen initially)
    ↓
GlobalAveragePooling2D
    ↓
Dense(512, relu) → Dropout(0.5) → Dense(256, relu) → Dropout(0.3)
    ↓
Output (10 classes, softmax)
```

**Parameters:**
- Total: ~25M parameters
- Trainable (Phase 1): ~2M
- Trainable (Phase 2): ~23M
- Fine-tuning: Last 50 layers

### 3. MobileNetV2 (Fine-tuned)

**Architecture:**
- Lightweight model optimized for mobile devices
- Depthwise separable convolutions
- Inverted residuals with linear bottlenecks

**Parameters:**
- Total: ~3.5M parameters
- Trainable (Phase 1): ~1M
- Trainable (Phase 2): ~3M

### 4. VGG16 (Fine-tuned)

**Architecture:**
- Classic deep CNN with 16 layers
- Small 3x3 convolution filters
- Multiple stacked conv layers

**Parameters:**
- Total: ~15M parameters
- Trainable (Phase 1): ~2M
- Trainable (Phase 2): ~14M

### 5. InceptionV3 (Fine-tuned)

**Architecture:**
- Multi-scale processing with inception modules
- Factorized convolutions
- Auxiliary classifiers

**Parameters:**
- Total: ~22M parameters
- Trainable (Phase 1): ~2M
- Trainable (Phase 2): ~20M

---

## 📈 Metrics

### Tracked Metrics

1. **Accuracy**: Overall classification accuracy
2. **Precision**: True positives / (True positives + False positives)
3. **Recall**: True positives / (True positives + False negatives)
4. **AUC**: Area Under the ROC Curve
5. **Loss**: Categorical cross-entropy loss
6. **F1-Score**: Harmonic mean of precision and recall (in reports)

### Cross-Validation

CNN from scratch supports K-Fold cross-validation:
- Default: 5 folds
- Reports mean ± std for all metrics
- Stores fold-wise results

### Evaluation Pipeline

```python
1. Train model
2. Predict on test set
3. Calculate metrics:
   - Accuracy, Precision, Recall, AUC
   - Classification report (per-class metrics)
   - Confusion matrix
   - ROC curves (per class)
4. Save metrics to JSON
5. Generate visualizations
```

---

## 🎨 Visualizations

### Generated Plots

1. **Training History**
   - Accuracy curves (train/val)
   - Loss curves (train/val)
   - Precision curves
   - Recall curves

2. **Confusion Matrix**
   - Heatmap visualization
   - Per-class performance

3. **ROC Curves**
   - Per-class ROC curves
   - AUC scores

4. **Model Comparison**
   - Bar charts (all metrics)
   - Radar chart
   - Precision vs Recall scatter
   - Loss comparison
   - Comprehensive summary

### Interactive Dashboard

- Real-time metric updates
- Filterable model selection
- Interactive Plotly charts
- Downloadable reports
- JSON metric inspector

---

## 🔧 Configuration

### Training Parameters

```python
# cnn_from_scratch.py
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
K_FOLDS = 5

# fine_tuning.py
IMAGE_SIZE = (224, 224)  # Pre-trained model standard
EPOCHS_FEATURE_EXTRACTION = 10
EPOCHS_FINE_TUNING = 30
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001
UNFREEZE_LAYERS = 50
```

### Callbacks

```python
- EarlyStopping (patience=10, monitor='val_loss')
- ReduceLROnPlateau (factor=0.5, patience=5)
- ModelCheckpoint (save_best_only=True)
- TensorBoard (for visualization)
```

---

## 📁 Outputs

### Models

All trained models are saved in `results/models/`:
- `cnn_scratch_*.h5` - CNN from scratch
- `{model}_feature_extraction_best.h5` - Feature extraction phase
- `{model}_fine_tuned_best.h5` - Fine-tuning phase
- `{model}_final.h5` - Final model

### Metrics

JSON files in `results/metrics/`:
```json
{
  "cross_validation": {
    "folds": [...],
    "mean_accuracy": 0.XX,
    "std_accuracy": 0.XX
  },
  "test": {
    "accuracy": 0.XX,
    "precision": 0.XX,
    "recall": 0.XX,
    "auc": 0.XX,
    "loss": 0.XX
  }
}
```

### Plots

All visualizations in `results/plots/`:
- `training_history.png`
- `confusion_matrix_*.png`
- `roc_curves_*.png`
- `metrics_comparison.png`
- `radar_chart_comparison.png`
- `comprehensive_summary.png`

---

## 🧪 Testing

### Verify Installation

```python
import tensorflow as tf
import streamlit as st
import plotly

print(f"TensorFlow: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

### Run Quick Test

```bash
# Test CNN training (5 epochs)
python -c "from cnn_from_scratch import *; Config.EPOCHS=5; main()"

# Test dashboard
streamlit run dashboard.py
```

---

## 🎯 Best Practices

1. **Data Augmentation**: Applied during training to prevent overfitting
2. **Regularization**: Dropout and batch normalization
3. **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning
4. **Early Stopping**: Prevent overfitting
5. **Cross-Validation**: Ensure robust performance estimates
6. **Transfer Learning**: Leverage pre-trained weights

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Keras Applications](https://keras.io/api/applications/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Model Performance Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```python
# Reduce batch size
BATCH_SIZE = 16  # or 8

# Use mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

**2. Dashboard Not Loading**
```bash
# Check if metrics exist
ls results/metrics/*.json

# Verify Streamlit installation
streamlit --version
```

**3. Slow Training**
```python
# Use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

---

## 📞 Support

For issues or questions:
- Check documentation in `docs/`
- Review code comments
- Examine generated metrics and plots

---

**Last Updated:** December 2025  
**Version:** 1.0  
**Subsystem:** 5 - Deep Learning Training and Comparison
