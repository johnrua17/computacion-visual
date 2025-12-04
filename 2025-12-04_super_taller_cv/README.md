# 🤖 Subsistema 5: Entrenamiento y Comparación de Modelos CNN

## Taller Integral de Computación Visual Avanzada - Subsistema 5

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Descripción

Subsistema completo de **entrenamiento, evaluación y comparación de modelos de Deep Learning**. Implementa:

✅ **CNN desde cero** con validación cruzada  
✅ **Fine-tuning** de modelos preentrenados (ResNet50, MobileNetV2, VGG16, InceptionV3)  
✅ **Análisis comparativo** con métricas exhaustivas  
✅ **Dashboard interactivo** para visualización de resultados  
✅ **Generación automática de visualizaciones** (gráficas, matrices, curvas ROC)

---

## 🎯 Características Principales

### 1. 🔬 Entrenamiento de CNN Personalizada
- Arquitectura profunda con 4 bloques convolucionales
- Batch Normalization y Dropout
- Validación cruzada K-Fold
- Early Stopping y Learning Rate Scheduling
- Métricas completas (Accuracy, Precision, Recall, AUC)

### 2. 🚀 Transfer Learning
- Modelos preentrenados de ImageNet
- Estrategia de dos fases:
  - **Fase 1**: Feature Extraction (top layers)
  - **Fase 2**: Fine-Tuning (todo el modelo)
- Soporte para múltiples arquitecturas

### 3. 📊 Análisis Comparativo
- Comparación automática entre modelos
- Visualizaciones:
  - Gráficas de barras
  - Radar charts
  - Matrices de confusión
  - Curvas ROC
  - Scatter plots Precision vs Recall

### 4. 🎨 Dashboard Interactivo
- Streamlit UI moderna y responsiva
- Filtros dinámicos
- Comparación lado a lado
- Exportación de datos (CSV, JSON)
- Gráficas interactivas con Plotly

---

## 🏗️ Estructura del Proyecto

```
2025-12-04_super_taller_cv/
├── python/
│   └── training/
│       ├── cnn_from_scratch.py      # Entrenamiento CNN
│       ├── fine_tuning.py           # Transfer Learning
│       ├── compare_models.py        # Comparación
│       ├── dashboard.py             # Dashboard Streamlit
│       ├── run_all.py               # Script todo-en-uno
│       └── requirements.txt         # Dependencias
├── data/
│   ├── raw/                         # Datos originales
│   └── processed/                   # Datos preprocesados
├── results/
│   ├── models/                      # Modelos guardados (.h5)
│   ├── plots/                       # Visualizaciones (.png)
│   └── metrics/                     # Métricas (.json, .csv)
└── docs/
    ├── README.md                    # Documentación principal
    ├── ARCHITECTURE.md              # Arquitectura detallada
    └── METRICAS.md                  # Explicación de métricas
```

---

## 🚀 Inicio Rápido

### Requisitos Previos

```bash
# Python 3.10+
python --version

# Instalar dependencias
cd 2025-12-04_super_taller_cv/python/training
pip install -r requirements.txt
```

### Opción 1: Pipeline Completo (Automático)

```bash
# Ejecutar todo el pipeline
python run_all.py --all
```

Esto ejecutará:
1. ✅ Entrenamiento de CNN desde cero
2. ✅ Fine-tuning de modelos preentrenados
3. ✅ Generación de comparaciones
4. ✅ Lanzamiento del dashboard

### Opción 2: Ejecución Manual

#### Paso 1: Entrenar CNN desde cero

```bash
python cnn_from_scratch.py
```

**Salida:**
- Modelo entrenado: `results/models/cnn_scratch_*.h5`
- Gráficas: `results/plots/training_history.png`
- Métricas: `results/metrics/cnn_scratch_*_metrics.json`

#### Paso 2: Fine-Tuning de Modelos Preentrenados

```bash
python fine_tuning.py
```

Selecciona los modelos a entrenar:
- 1. ResNet50
- 2. MobileNetV2
- 3. VGG16
- 4. InceptionV3

**Salida:**
- Modelos: `results/models/{model}_final.h5`
- Gráficas: `results/plots/{model}_training_history.png`
- Métricas: `results/metrics/{model}_metrics.json`

#### Paso 3: Generar Comparaciones

```bash
python compare_models.py
```

**Salida:**
- `results/plots/metrics_comparison.png`
- `results/plots/radar_chart_comparison.png`
- `results/plots/comprehensive_summary.png`
- `results/metrics/models_comparison.csv`

#### Paso 4: Lanzar Dashboard

```bash
streamlit run dashboard.py
```

Abre tu navegador en: **http://localhost:8501**

---

## 📊 Modelos Disponibles

### 1. CNN from Scratch

```
Input (128×128×3)
    ↓
[Conv32 → BN → Conv32 → BN → Pool → Dropout] × 1
[Conv64 → BN → Conv64 → BN → Pool → Dropout] × 1
[Conv128 → BN → Conv128 → BN → Pool → Dropout] × 1
[Conv256 → BN → Conv256 → BN → Pool → Dropout] × 1
    ↓
Flatten → Dense512 → Dense256 → Output(10)
```

**Parámetros:** ~2.5M

### 2. ResNet50 (Fine-tuned)

```
Input (224×224×3) → ResNet50 Base → GAP → Dense512 → Dense256 → Output(10)
```

**Parámetros:** ~25M (23M trainable en fine-tuning)

### 3. MobileNetV2 (Fine-tuned)

```
Input (224×224×3) → MobileNetV2 Base → GAP → Dense512 → Dense256 → Output(10)
```

**Parámetros:** ~3.5M (ligero, optimizado para dispositivos móviles)

### 4. VGG16 (Fine-tuned)

```
Input (224×224×3) → VGG16 Base → GAP → Dense512 → Dense256 → Output(10)
```

**Parámetros:** ~15M

### 5. InceptionV3 (Fine-tuned)

```
Input (224×224×3) → InceptionV3 Base → GAP → Dense512 → Dense256 → Output(10)
```

**Parámetros:** ~22M

---

## 📈 Métricas Evaluadas

| Métrica | Descripción | Rango | Interpretación |
|---------|-------------|-------|----------------|
| **Accuracy** | Proporción de predicciones correctas | [0, 1] | 1 = Perfecto |
| **Precision** | TP / (TP + FP) | [0, 1] | Pocos falsos positivos |
| **Recall** | TP / (TP + FN) | [0, 1] | Pocos falsos negativos |
| **F1-Score** | Media armónica Precision/Recall | [0, 1] | Balance |
| **AUC** | Área bajo curva ROC | [0, 1] | 1 = Perfecto |
| **Loss** | Cross-entropy loss | [0, ∞) | 0 = Perfecto |

---

## 🎨 Visualizaciones Generadas

### 1. Training History
![Training History](results/plots/training_history.png)

### 2. Confusion Matrix
![Confusion Matrix](results/plots/confusion_matrix_cnn.png)

### 3. ROC Curves
![ROC Curves](results/plots/roc_curves_cnn.png)

### 4. Model Comparison
![Comparison](results/plots/comprehensive_summary.png)

### 5. Radar Chart
![Radar](results/plots/radar_chart_comparison.png)

---

## 🎮 Uso del Dashboard

### Tabs Disponibles

#### 📊 Overview
- Métricas clave de todos los modelos
- Radar chart interactivo
- Heatmap de métricas

#### 📈 Detailed Metrics
- Comparación detallada de métricas
- Gráficas de loss
- Precision vs Recall

#### 🎯 Comparisons
- Comparación lado a lado de 2 modelos
- Análisis de diferencias

#### 📄 Raw Data
- Tablas de datos
- Exportación a CSV
- Visualización de JSON

---

## ⚙️ Configuración

### Parámetros de Entrenamiento

```python
# CNN from Scratch
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
K_FOLDS = 5

# Fine-Tuning
IMAGE_SIZE = (224, 224)
EPOCHS_FEATURE_EXTRACTION = 10
EPOCHS_FINE_TUNING = 30
LEARNING_RATE_INITIAL = 0.001
LEARNING_RATE_FINE_TUNE = 0.0001
UNFREEZE_LAYERS = 50
```

### Callbacks

- **EarlyStopping**: patience=10
- **ReduceLROnPlateau**: factor=0.5, patience=5
- **ModelCheckpoint**: save_best_only=True
- **TensorBoard**: histograms

---

## 📚 Documentación Completa

- **[README.md](docs/README.md)** - Documentación principal
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Arquitectura detallada del sistema
- **[METRICAS.md](docs/METRICAS.md)** - Explicación completa de métricas

---

## 🧪 Ejemplo de Uso

```python
# 1. Entrenar CNN
from cnn_from_scratch import CNNTrainer, DataLoader

# Cargar datos
(x_train, y_train), (x_test, y_test) = DataLoader.load_cifar10_data()

# Entrenar
trainer = CNNTrainer()
trainer.train_final_model(x_train, y_train, x_val, y_val)
trainer.evaluate_model(x_test, y_test)

# 2. Fine-tuning
from fine_tuning import TransferLearningModel

model = TransferLearningModel('resnet50')
model.feature_extraction_training(x_train, y_train, x_val, y_val)
model.fine_tuning_training(x_train, y_train, x_val, y_val, base_model)
metrics = model.evaluate_model(x_test, y_test)

# 3. Comparar
from compare_models import ModelComparison

comparator = ModelComparison(metrics_dir, plots_dir)
comparator.generate_all_comparisons()
```

---

## 🛠️ Solución de Problemas

### Error: Out of Memory (OOM)

```python
# Reducir batch size
BATCH_SIZE = 16  # o 8

# Usar mixed precision
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Dashboard no carga

```bash
# Verificar métricas
ls results/metrics/*.json

# Reinstalar Streamlit
pip install --upgrade streamlit
```

### Entrenamiento lento

```bash
# Verificar GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Reducir epochs para pruebas
Config.EPOCHS = 10
```

---

## 📦 Entregables

✅ **Código fuente completo**  
✅ **Modelos entrenados** (.h5 files)  
✅ **Métricas** (JSON, CSV)  
✅ **Visualizaciones** (PNG, high-res)  
✅ **Dashboard interactivo** (Streamlit)  
✅ **Documentación detallada** (Markdown)  
✅ **Scripts de automatización**

---

## 🎯 Resultados Esperados

### CIFAR-10 Dataset

| Modelo | Accuracy | Precision | Recall | AUC | Training Time |
|--------|----------|-----------|--------|-----|---------------|
| CNN Scratch | 70-75% | 0.70-0.75 | 0.70-0.75 | 0.85-0.90 | ~30 min |
| ResNet50 | 85-90% | 0.85-0.90 | 0.85-0.90 | 0.92-0.95 | ~60 min |
| MobileNetV2 | 80-85% | 0.80-0.85 | 0.80-0.85 | 0.90-0.93 | ~45 min |
| VGG16 | 85-88% | 0.85-0.88 | 0.85-0.88 | 0.91-0.94 | ~75 min |
| InceptionV3 | 87-92% | 0.87-0.92 | 0.87-0.92 | 0.93-0.96 | ~90 min |

*Tiempos en GPU NVIDIA RTX 3060*

---

## 🤝 Contribuciones

Este subsistema forma parte del **Taller Integral de Computación Visual Avanzada** y cumple con todos los requisitos especificados:

✅ Entrenamiento de CNN desde cero  
✅ Validación cruzada  
✅ Fine-tuning con modelos preentrenados  
✅ Comparación de modelos  
✅ Métricas comprehensivas  
✅ Visualizaciones profesionales  
✅ Dashboard interactivo  
✅ Documentación completa  
✅ Commits en inglés  

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para detalles.

---

## 👥 Autores

- **Equipo Subsistema 5**
- Taller Integral de Computación Visual Avanzada
- Diciembre 2025

---

## 📞 Soporte

Para problemas o preguntas:
1. Revisar documentación en `docs/`
2. Verificar logs de entrenamiento
3. Examinar métricas generadas
4. Consultar código comentado

---

## 🌟 Características Destacadas

- ✨ **Arquitectura modular** y extensible
- ✨ **Código limpio** y bien documentado
- ✨ **Pipeline automatizado** completo
- ✨ **Visualizaciones profesionales**
- ✨ **Dashboard moderno** e interactivo
- ✨ **Métricas exhaustivas** y precisas
- ✨ **Soporte GPU** para entrenamiento rápido
- ✨ **Compatible** con datasets personalizados

---

**¡Disfruta entrenando y comparando modelos de Deep Learning!** 🚀🤖

---

*Última actualización: Diciembre 2025*
