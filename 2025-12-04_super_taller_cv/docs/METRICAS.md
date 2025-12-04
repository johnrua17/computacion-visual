# Metrics Documentation - Subsystem 5

## 📊 Performance Metrics Overview

This document provides comprehensive documentation of all performance metrics used in Subsystem 5 for model evaluation and comparison.

---

## 🎯 Primary Metrics

### 1. Accuracy

**Definition:** Proportion of correct predictions among total predictions.

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Range:** [0, 1] where 1 is perfect classification

**Interpretation:**
- **High (>0.9)**: Excellent model performance
- **Medium (0.7-0.9)**: Good performance
- **Low (<0.7)**: Poor performance, needs improvement

**Use Cases:**
- Overall model performance assessment
- Quick comparison between models
- Balanced datasets

**Limitations:**
- Misleading on imbalanced datasets
- Doesn't show per-class performance

**Example:**
```python
from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0]

accuracy = accuracy_score(y_true, y_pred)
# Output: 0.8 (80% correct)
```

---

### 2. Precision

**Definition:** Proportion of true positives among all positive predictions.

```
Precision = TP / (TP + FP)
```

**Range:** [0, 1] where 1 is no false positives

**Interpretation:**
- Answers: "Of all instances predicted as positive, how many are actually positive?"
- **High precision**: Few false positives
- **Low precision**: Many false positives

**Use Cases:**
- When false positives are costly
- Spam detection (don't want legit emails marked as spam)
- Medical diagnostics (avoid unnecessary treatments)

**Example:**
```python
from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 1, 0]

precision = precision_score(y_true, y_pred)
# Output: 1.0 (all positive predictions were correct)
```

---

### 3. Recall (Sensitivity)

**Definition:** Proportion of true positives among all actual positives.

```
Recall = TP / (TP + FN)
```

**Range:** [0, 1] where 1 is no false negatives

**Interpretation:**
- Answers: "Of all actual positive instances, how many did we correctly identify?"
- **High recall**: Few false negatives
- **Low recall**: Many false negatives

**Use Cases:**
- When false negatives are costly
- Disease detection (don't want to miss sick patients)
- Fraud detection (catch all fraudulent transactions)

**Example:**
```python
from sklearn.metrics import recall_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 1, 0]

recall = recall_score(y_true, y_pred)
# Output: 0.75 (found 3 out of 4 positive cases)
```

---

### 4. F1-Score

**Definition:** Harmonic mean of precision and recall.

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Range:** [0, 1] where 1 is perfect precision and recall

**Interpretation:**
- Balances precision and recall
- Single metric for model quality
- Useful when classes are imbalanced

**Example:**
```python
from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 1, 0]

f1 = f1_score(y_true, y_pred)
# Output: 0.857
```

---

### 5. AUC (Area Under ROC Curve)

**Definition:** Area under the Receiver Operating Characteristic curve.

**Range:** [0, 1] where:
- 1.0 = Perfect classifier
- 0.5 = Random classifier
- <0.5 = Worse than random

**Interpretation:**
- Probability that model ranks random positive higher than random negative
- Threshold-independent metric
- Good for comparing models

**ROC Curve:**
- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (TPR/Recall)

**Use Cases:**
- Model comparison
- Threshold selection
- Imbalanced datasets

**Example:**
```python
from sklearn.metrics import roc_auc_score

y_true = [0, 0, 1, 1]
y_scores = [0.1, 0.4, 0.35, 0.8]

auc = roc_auc_score(y_true, y_scores)
# Output: 0.75
```

---

### 6. Loss (Categorical Cross-Entropy)

**Definition:** Measure of dissimilarity between predicted and true distributions.

```
Loss = -Σ(y_true * log(y_pred))
```

**Range:** [0, ∞) where 0 is perfect prediction

**Interpretation:**
- **Lower is better**
- Penalizes confident wrong predictions heavily
- Guides training process

**Use Cases:**
- Model optimization
- Training progress monitoring
- Early stopping criteria

---

## 📈 Secondary Metrics

### Confusion Matrix

**Definition:** Table showing true positives, false positives, true negatives, and false negatives.

```
                Predicted
              Pos       Neg
Actual  Pos   TP        FN
        Neg   FP        TN
```

**Components:**
- **TP (True Positive)**: Correctly predicted positive
- **TN (True Negative)**: Correctly predicted negative
- **FP (False Positive)**: Incorrectly predicted positive (Type I error)
- **FN (False Negative)**: Incorrectly predicted negative (Type II error)

**Visualization:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
```

---

### Per-Class Metrics

**Classification Report:**
```
              precision    recall  f1-score   support

     class 0       0.95      0.92      0.93      1000
     class 1       0.89      0.94      0.91       900
     class 2       0.92      0.88      0.90      1100
     
    accuracy                           0.91      3000
   macro avg       0.92      0.91      0.91      3000
weighted avg       0.92      0.91      0.91      3000
```

**Definitions:**
- **Support**: Number of actual occurrences
- **Macro avg**: Unweighted mean (treats all classes equally)
- **Weighted avg**: Weighted by support (accounts for class imbalance)

---

## 🔄 Cross-Validation Metrics

### K-Fold Metrics

**Mean ± Standard Deviation:**
```python
{
    "mean_accuracy": 0.8234,
    "std_accuracy": 0.0156,
    "mean_precision": 0.8145,
    "mean_recall": 0.8267,
    "mean_auc": 0.9012
}
```

**Interpretation:**
- Mean: Expected performance
- Std: Consistency across folds
- **Low std**: Stable model
- **High std**: Variance in performance

---

## 📊 Comparative Metrics

### Model Comparison Table

```csv
Model,Accuracy,Precision,Recall,AUC,Loss
CNN SCRATCH,0.7234,0.7156,0.7298,0.8543,0.7821
RESNET50,0.8567,0.8489,0.8623,0.9234,0.4123
MOBILENETV2,0.8234,0.8123,0.8345,0.9012,0.4892
VGG16,0.8423,0.8334,0.8512,0.9156,0.4567
INCEPTIONV3,0.8689,0.8598,0.8789,0.9345,0.3856
```

### Performance Rankings

**Best Accuracy:**
1. InceptionV3: 0.8689
2. ResNet50: 0.8567
3. VGG16: 0.8423

**Best Precision:**
1. InceptionV3: 0.8598
2. ResNet50: 0.8489
3. VGG16: 0.8334

**Lowest Loss:**
1. InceptionV3: 0.3856
2. ResNet50: 0.4123
3. VGG16: 0.4567

---

## 🎨 Visualization Metrics

### 1. Training Curves

**Metrics Tracked:**
- Training accuracy/loss
- Validation accuracy/loss
- Training precision/recall
- Validation precision/recall

**Analysis:**
- **Overfitting**: Train accuracy high, val accuracy low
- **Underfitting**: Both train and val accuracy low
- **Good fit**: Both train and val accuracy high and close

### 2. Radar Chart

**Dimensions:**
- Accuracy
- Precision
- Recall
- AUC

**Use:** Quick visual comparison of multiple models

### 3. Precision-Recall Curve

**Components:**
- Precision vs Recall at different thresholds
- Shows trade-off between precision and recall

**Ideal:** Curve close to top-right corner

---

## 💡 Metric Selection Guide

### Choose metrics based on problem:

**Balanced Dataset:**
- ✅ Accuracy
- ✅ F1-Score
- ✅ AUC

**Imbalanced Dataset:**
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ AUC
- ❌ Accuracy (can be misleading)

**Cost-Sensitive:**
- **False positives costly**: Maximize precision
- **False negatives costly**: Maximize recall
- **Both costly**: Maximize F1-score

**Multi-Class:**
- ✅ Macro/Weighted F1
- ✅ Per-class precision/recall
- ✅ Confusion matrix

---

## 📋 Metrics Checklist

### Training Phase
- [ ] Track training loss
- [ ] Track validation loss
- [ ] Monitor overfitting (train vs val gap)
- [ ] Use early stopping
- [ ] Save best model checkpoint

### Evaluation Phase
- [ ] Calculate accuracy
- [ ] Calculate precision
- [ ] Calculate recall
- [ ] Calculate F1-score
- [ ] Calculate AUC
- [ ] Generate confusion matrix
- [ ] Generate classification report
- [ ] Plot ROC curves

### Comparison Phase
- [ ] Compare across all models
- [ ] Rank by each metric
- [ ] Analyze trade-offs
- [ ] Consider computational cost
- [ ] Consider inference time

---

## 🧮 Calculation Examples

### Binary Classification Example

```python
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
y_pred = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]

# Manual calculation
TP = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))  # 5
TN = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))  # 3
FP = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))  # 1
FN = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))  # 1

Accuracy = (TP + TN) / (TP + TN + FP + FN) = 8/10 = 0.8
Precision = TP / (TP + FP) = 5/6 = 0.833
Recall = TP / (TP + FN) = 5/6 = 0.833
F1 = 2 * (0.833 * 0.833) / (0.833 + 0.833) = 0.833
```

### Multi-Class Example

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 1, 2, 0, 1, 1]

print(classification_report(y_true, y_pred))
```

---

## 📊 Benchmarks

### CIFAR-10 Dataset Benchmarks

**State-of-the-art:**
- EfficientNet-B7: ~98% accuracy
- Vision Transformer: ~99% accuracy

**Typical Results:**
- Basic CNN: 60-75%
- ResNet50: 80-90%
- Fine-tuned ResNet50: 85-93%

**Expected Results (This Subsystem):**
- CNN from Scratch: 70-75%
- ResNet50 Fine-tuned: 85-90%
- MobileNetV2 Fine-tuned: 80-85%
- VGG16 Fine-tuned: 85-88%
- InceptionV3 Fine-tuned: 87-92%

---

## 🔍 Metrics Interpretation

### What makes a good model?

**Accuracy > 0.85:**
- Strong performance
- Suitable for deployment

**Precision/Recall > 0.80:**
- Balanced predictions
- Few misclassifications

**AUC > 0.90:**
- Excellent discriminative power
- Robust across thresholds

**Loss < 0.5:**
- Well-calibrated predictions
- Converged training

---

## 📝 Metrics Reporting Template

```markdown
## Model Performance Report

### Overall Metrics
- Accuracy: 0.XXXX
- Precision: 0.XXXX
- Recall: 0.XXXX
- F1-Score: 0.XXXX
- AUC: 0.XXXX
- Loss: 0.XXXX

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.XX      | 0.XX   | 0.XX     | XXX     |
| 1     | 0.XX      | 0.XX   | 0.XX     | XXX     |

### Cross-Validation Results
- Mean Accuracy: 0.XXXX ± 0.XXXX
- Mean Precision: 0.XXXX ± 0.XXXX
- Mean Recall: 0.XXXX ± 0.XXXX

### Visualizations
- Training curves
- Confusion matrix
- ROC curves
- Comparison charts
```

---

**Version:** 1.0  
**Last Updated:** December 2025  
**Reference:** Subsystem 5 - Deep Learning Training
