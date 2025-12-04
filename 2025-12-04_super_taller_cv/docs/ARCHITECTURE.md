# Architecture Documentation - Subsystem 5

## 🏛️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Subsystem 5 Architecture                  │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐│
│  │  Data Layer │───▶│ Training     │───▶│  Evaluation    ││
│  │             │    │ Layer        │    │  Layer         ││
│  │ - CIFAR-10  │    │              │    │                ││
│  │ - Custom DS │    │ - CNN        │    │ - Metrics      ││
│  │ - Augment   │    │ - ResNet     │    │ - Confusion    ││
│  │             │    │ - MobileNet  │    │ - ROC Curves   ││
│  └─────────────┘    │ - VGG16      │    └────────────────┘│
│                     │ - Inception  │           │           │
│                     └──────────────┘           │           │
│                            │                   │           │
│                            ▼                   ▼           │
│                     ┌──────────────┐    ┌────────────────┐│
│                     │  Model       │    │  Visualization ││
│                     │  Storage     │    │  Layer         ││
│                     │              │    │                ││
│                     │ - .h5 files  │    │ - Plots        ││
│                     │ - Checkpts   │    │ - Dashboard    ││
│                     └──────────────┘    │ - Comparison   ││
│                                         └────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Architecture

### Training Pipeline

```
┌────────────┐
│  Raw Data  │
│  (CIFAR10) │
└─────┬──────┘
      │
      ▼
┌────────────────────┐
│  Preprocessing     │
│  - Resize          │
│  - Normalize       │
│  - Augmentation    │
└─────┬──────────────┘
      │
      ├─────────────────┐
      │                 │
      ▼                 ▼
┌───────────┐    ┌─────────────────┐
│ CNN from  │    │  Fine-Tuning    │
│  Scratch  │    │  - ResNet50     │
│           │    │  - MobileNetV2  │
│  - Build  │    │  - VGG16        │
│  - Train  │    │  - InceptionV3  │
│  - K-Fold │    │                 │
└─────┬─────┘    │  Phase 1: FE    │
      │          │  Phase 2: FT    │
      │          └────────┬────────┘
      │                   │
      ▼                   ▼
┌────────────────────────────┐
│  Model Evaluation          │
│  - Accuracy, Precision     │
│  - Recall, AUC, Loss       │
│  - Confusion Matrix        │
│  - Classification Report   │
└─────┬──────────────────────┘
      │
      ▼
┌────────────────────────────┐
│  Results Storage           │
│  - Models (*.h5)           │
│  - Metrics (*.json)        │
│  - Plots (*.png)           │
└─────┬──────────────────────┘
      │
      ▼
┌────────────────────────────┐
│  Visualization             │
│  - Comparison Plots        │
│  - Interactive Dashboard   │
└────────────────────────────┘
```

---

## 🧩 Component Architecture

### 1. CNN from Scratch Module

```python
CNNTrainer
    │
    ├── CNNArchitecture
    │   └── build_model()
    │       ├── Conv Blocks (4x)
    │       ├── BatchNorm Layers
    │       ├── Dropout Layers
    │       └── Dense Layers
    │
    ├── DataLoader
    │   ├── load_cifar10_data()
    │   └── create_data_generators()
    │
    ├── MetricsVisualizer
    │   ├── plot_training_history()
    │   ├── plot_confusion_matrix()
    │   └── plot_roc_curves()
    │
    └── Training Methods
        ├── train_with_cross_validation()
        ├── train_final_model()
        └── evaluate_model()
```

### 2. Fine-Tuning Module

```python
TransferLearningModel
    │
    ├── Model Selection
    │   ├── ResNet50
    │   ├── MobileNetV2
    │   ├── VGG16
    │   └── InceptionV3
    │
    ├── build_model()
    │   ├── Load Pre-trained Base
    │   ├── Add Custom Top Layers
    │   └── Compile with Optimizer
    │
    ├── Two-Phase Training
    │   ├── Phase 1: feature_extraction_training()
    │   │   └── Freeze base, train top
    │   │
    │   └── Phase 2: fine_tuning_training()
    │       └── Unfreeze layers, train all
    │
    └── Evaluation
        ├── evaluate_model()
        └── plot_training_progress()
```

### 3. Model Comparison Module

```python
ModelComparison
    │
    ├── Data Management
    │   ├── load_all_metrics()
    │   └── create_comparison_table()
    │
    └── Visualizations
        ├── plot_metrics_comparison()
        ├── plot_radar_chart()
        ├── plot_accuracy_vs_parameters()
        ├── plot_loss_comparison()
        ├── plot_precision_recall_comparison()
        └── create_comprehensive_summary()
```

### 4. Interactive Dashboard Module

```python
Dashboard (Streamlit)
    │
    ├── DashboardData
    │   ├── load_metrics()
    │   └── create_comparison_dataframe()
    │
    ├── Tabs
    │   ├── Overview Tab
    │   │   ├── Key Metrics
    │   │   ├── Radar Chart
    │   │   └── Heatmap
    │   │
    │   ├── Detailed Metrics Tab
    │   │   ├── Metrics Comparison
    │   │   └── Loss & PR Plots
    │   │
    │   ├── Comparisons Tab
    │   │   └── Side-by-side Analysis
    │   │
    │   └── Raw Data Tab
    │       ├── DataFrames
    │       └── JSON Details
    │
    └── Interactive Plots (Plotly)
        ├── plot_metrics_comparison()
        ├── plot_radar_chart()
        ├── plot_precision_recall_scatter()
        └── plot_heatmap()
```

---

## 🗄️ Data Architecture

### Input Data Structure

```
data/
├── raw/
│   └── cifar-10-batches-py/
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       └── test_batch
│
└── processed/
    ├── train/
    │   ├── class_0/
    │   ├── class_1/
    │   └── ...
    └── test/
        ├── class_0/
        ├── class_1/
        └── ...
```

### Output Data Structure

```
results/
├── models/
│   ├── cnn_scratch_20251204_120000_best.h5
│   ├── cnn_scratch_20251204_120000_final.h5
│   ├── resnet50_feature_extraction_best.h5
│   ├── resnet50_fine_tuned_best.h5
│   ├── resnet50_final.h5
│   └── ...
│
├── plots/
│   ├── training_history.png
│   ├── training_history_fold1.png
│   ├── confusion_matrix_cnn.png
│   ├── roc_curves_cnn.png
│   ├── metrics_comparison.png
│   ├── radar_chart_comparison.png
│   └── comprehensive_summary.png
│
└── metrics/
    ├── cnn_scratch_20251204_120000_metrics.json
    ├── resnet50_metrics.json
    ├── mobilenetv2_metrics.json
    ├── models_comparison.csv
    └── all_models_comparison.json
```

---

## 🔌 Integration Points

### 1. Data Input Integration

```python
# Custom dataset integration
def load_custom_dataset(data_dir):
    """
    Replace CIFAR-10 with custom dataset
    
    Args:
        data_dir: Path to dataset directory
        
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    # Load and preprocess custom data
    ...
```

### 2. Model Export Integration

```python
# Export for deployment
def export_model_for_serving(model_path, export_dir):
    """
    Export model for TensorFlow Serving
    
    Args:
        model_path: Path to .h5 model
        export_dir: Directory for SavedModel format
    """
    model = keras.models.load_model(model_path)
    model.save(export_dir, save_format='tf')
```

### 3. API Integration

```python
# REST API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request:
        image: Base64 encoded image
        
    Response:
        predictions: Class probabilities
    """
    ...
```

---

## 🔒 Design Patterns

### 1. Singleton Pattern - Configuration

```python
class Config:
    """Single configuration instance"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 2. Factory Pattern - Model Creation

```python
class ModelFactory:
    """Create different model types"""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        if model_type == 'cnn':
            return CNNArchitecture.build_model(**kwargs)
        elif model_type == 'resnet50':
            return TransferLearningModel('resnet50')
        ...
```

### 3. Strategy Pattern - Training Strategies

```python
class TrainingStrategy:
    """Define training strategy interface"""
    
    def train(self, model, data):
        raise NotImplementedError

class CrossValidationStrategy(TrainingStrategy):
    def train(self, model, data):
        # K-Fold cross-validation
        ...

class StandardTrainingStrategy(TrainingStrategy):
    def train(self, model, data):
        # Standard train/val split
        ...
```

### 4. Observer Pattern - Callbacks

```python
class MetricsObserver:
    """Observe training metrics"""
    
    def on_epoch_end(self, epoch, logs):
        # Log metrics
        self.metrics.append(logs)
```

---

## 🎯 Scalability Considerations

### Horizontal Scaling

```python
# Multi-GPU training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = build_model()
    model.compile(...)
    
model.fit(..., batch_size=BATCH_SIZE * strategy.num_replicas_in_sync)
```

### Vertical Scaling

```python
# Memory optimization
tf.config.experimental.set_memory_growth(gpu, True)

# Mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Distributed Training

```python
# Multi-worker training
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Training on multiple machines
...
```

---

## 🧪 Testing Architecture

```
tests/
├── unit/
│   ├── test_cnn_architecture.py
│   ├── test_data_loader.py
│   └── test_metrics.py
│
├── integration/
│   ├── test_training_pipeline.py
│   └── test_model_evaluation.py
│
└── e2e/
    └── test_full_workflow.py
```

---

## 📊 Performance Optimization

### 1. Data Pipeline Optimization

```python
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.cache()
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

### 2. Model Optimization

```python
# Quantization for deployment
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

### 3. Inference Optimization

```python
# Model pruning
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model)

# Knowledge distillation
teacher_model = large_model
student_model = small_model
```

---

## 🔐 Security Considerations

1. **Model Protection**: Encrypt saved models
2. **Input Validation**: Sanitize input data
3. **Access Control**: Restrict dashboard access
4. **Data Privacy**: Anonymize training data

---

## 📝 Logging Architecture

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Training started...")
```

---

**Version:** 1.0  
**Last Updated:** December 2025  
**Maintainer:** Subsystem 5 Team
