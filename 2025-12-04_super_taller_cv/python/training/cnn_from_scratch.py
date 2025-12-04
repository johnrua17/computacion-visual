"""
CNN Training from Scratch
Subsystem 5: Training and Model Comparison

This script implements a Convolutional Neural Network from scratch using TensorFlow/Keras
with cross-validation and comprehensive metrics analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
class Config:
    """Configuration parameters for CNN training"""
    
    # Data parameters
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 32
    NUM_CLASSES = 10  # Adjust based on your dataset
    
    # Training parameters
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    K_FOLDS = 5
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    
    # Model name
    MODEL_NAME = f'cnn_scratch_{datetime.now().strftime("%Y%m%d_%H%M%S")}'


class CNNArchitecture:
    """Define CNN architecture from scratch"""
    
    @staticmethod
    def build_model(input_shape, num_classes):
        """
        Build a CNN model with modern architecture
        
        Args:
            input_shape: Tuple of image dimensions (height, width, channels)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1'),
            layers.BatchNormalization(name='bn4_1'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_2'),
            layers.BatchNormalization(name='bn4_2'),
            layers.MaxPooling2D((2, 2), name='pool4'),
            layers.Dropout(0.25, name='dropout4'),
            
            # Fully connected layers
            layers.Flatten(name='flatten'),
            layers.Dense(512, activation='relu', name='fc1'),
            layers.BatchNormalization(name='bn_fc1'),
            layers.Dropout(0.5, name='dropout_fc1'),
            layers.Dense(256, activation='relu', name='fc2'),
            layers.BatchNormalization(name='bn_fc2'),
            layers.Dropout(0.5, name='dropout_fc2'),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        return model


class DataLoader:
    """Handle data loading and preprocessing"""
    
    @staticmethod
    def create_data_generators():
        """Create data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=Config.VALIDATION_SPLIT
        )
        
        # Validation data (no augmentation, only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=Config.VALIDATION_SPLIT
        )
        
        return train_datagen, val_datagen
    
    @staticmethod
    def load_cifar10_data():
        """
        Load CIFAR-10 dataset as example
        Replace this with your custom dataset loader
        """
        print("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Resize images if needed
        if Config.IMAGE_SIZE != (32, 32):
            x_train = tf.image.resize(x_train, Config.IMAGE_SIZE).numpy()
            x_test = tf.image.resize(x_test, Config.IMAGE_SIZE).numpy()
        
        # Convert labels to categorical
        y_train = keras.utils.to_categorical(y_train, Config.NUM_CLASSES)
        y_test = keras.utils.to_categorical(y_test, Config.NUM_CLASSES)
        
        print(f"Training samples: {len(x_train)}")
        print(f"Test samples: {len(x_test)}")
        print(f"Image shape: {x_train.shape[1:]}")
        
        return (x_train, y_train), (x_test, y_test)


class MetricsVisualizer:
    """Visualize training metrics and model performance"""
    
    @staticmethod
    def plot_training_history(history, fold=None):
        """Plot training and validation metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        fold_suffix = f'_fold{fold}' if fold is not None else ''
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, f'training_history{fold_suffix}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=None):
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - CNN from Scratch', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'confusion_matrix_cnn.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_roc_curves(y_true, y_pred_proba, num_classes):
        """Plot ROC curves for each class"""
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - CNN from Scratch', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.PLOTS_DIR, 'roc_curves_cnn.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


class CNNTrainer:
    """Handle CNN training with cross-validation"""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.history = None
        self.metrics = {}
    
    def train_with_cross_validation(self, x_data, y_data):
        """Train model using K-Fold cross-validation"""
        
        print(f"\n{'='*60}")
        print(f"Starting {Config.K_FOLDS}-Fold Cross-Validation")
        print(f"{'='*60}\n")
        
        kfold = KFold(n_splits=Config.K_FOLDS, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(x_data), 1):
            print(f"\n{'─'*60}")
            print(f"Training Fold {fold}/{Config.K_FOLDS}")
            print(f"{'─'*60}")
            
            # Split data
            x_train, x_val = x_data[train_idx], x_data[val_idx]
            y_train, y_val = y_data[train_idx], y_data[val_idx]
            
            # Build model
            input_shape = x_train.shape[1:]
            model = CNNArchitecture.build_model(input_shape, Config.NUM_CLASSES)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    os.path.join(Config.MODELS_DIR, f'cnn_fold{fold}_best.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model
            history = model.fit(
                x_train, y_train,
                batch_size=Config.BATCH_SIZE,
                epochs=Config.EPOCHS,
                validation_data=(x_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate on validation set
            val_loss, val_acc, val_prec, val_rec, val_auc = model.evaluate(
                x_val, y_val, verbose=0
            )
            
            fold_results.append({
                'fold': fold,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_precision': val_prec,
                'val_recall': val_rec,
                'val_auc': val_auc
            })
            
            print(f"\nFold {fold} Results:")
            print(f"  Validation Loss: {val_loss:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Validation Precision: {val_prec:.4f}")
            print(f"  Validation Recall: {val_rec:.4f}")
            print(f"  Validation AUC: {val_auc:.4f}")
            
            # Plot training history for this fold
            MetricsVisualizer.plot_training_history(history, fold=fold)
        
        # Calculate mean metrics across folds
        self.metrics['cross_validation'] = {
            'folds': fold_results,
            'mean_accuracy': np.mean([r['val_accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['val_accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['val_precision'] for r in fold_results]),
            'mean_recall': np.mean([r['val_recall'] for r in fold_results]),
            'mean_auc': np.mean([r['val_auc'] for r in fold_results])
        }
        
        print(f"\n{'='*60}")
        print("Cross-Validation Summary")
        print(f"{'='*60}")
        print(f"Mean Accuracy: {self.metrics['cross_validation']['mean_accuracy']:.4f} ± {self.metrics['cross_validation']['std_accuracy']:.4f}")
        print(f"Mean Precision: {self.metrics['cross_validation']['mean_precision']:.4f}")
        print(f"Mean Recall: {self.metrics['cross_validation']['mean_recall']:.4f}")
        print(f"Mean AUC: {self.metrics['cross_validation']['mean_auc']:.4f}")
        
        return fold_results
    
    def train_final_model(self, x_train, y_train, x_val, y_val):
        """Train final model on full training set"""
        
        print(f"\n{'='*60}")
        print("Training Final Model")
        print(f"{'='*60}\n")
        
        # Build model
        input_shape = x_train.shape[1:]
        self.model = CNNArchitecture.build_model(input_shape, Config.NUM_CLASSES)
        
        # Display model architecture
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(Config.MODELS_DIR, f'{Config.MODEL_NAME}_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            TensorBoard(
                log_dir=os.path.join(Config.RESULTS_DIR, 'logs', Config.MODEL_NAME),
                histogram_freq=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=Config.BATCH_SIZE,
            epochs=Config.EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history
        MetricsVisualizer.plot_training_history(self.history)
        
        # Save final model
        self.model.save(os.path.join(Config.MODELS_DIR, f'{Config.MODEL_NAME}_final.h5'))
        print(f"\n✓ Model saved: {Config.MODEL_NAME}_final.h5")
        
        return self.history
    
    def evaluate_model(self, x_test, y_test):
        """Comprehensive model evaluation"""
        
        print(f"\n{'='*60}")
        print("Model Evaluation")
        print(f"{'='*60}\n")
        
        # Predict
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_prec, test_rec, test_auc = self.model.evaluate(
            x_test, y_test, verbose=0
        )
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Plot confusion matrix
        class_names = [f'Class {i}' for i in range(Config.NUM_CLASSES)]
        MetricsVisualizer.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # Plot ROC curves
        MetricsVisualizer.plot_roc_curves(y_test, y_pred_proba, Config.NUM_CLASSES)
        
        # Store metrics
        self.metrics['test'] = {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'auc': float(test_auc)
        }
        
        # Save metrics to JSON
        with open(os.path.join(Config.METRICS_DIR, f'{Config.MODEL_NAME}_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        return self.metrics


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("CNN Training from Scratch - Subsystem 5")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.METRICS_DIR, exist_ok=True)
    
    # Load data (using CIFAR-10 as example)
    (x_train, y_train), (x_test, y_test) = DataLoader.load_cifar10_data()
    
    # Initialize trainer
    trainer = CNNTrainer()
    
    # Option 1: Cross-validation
    print("\n" + "─"*60)
    print("Option: Perform Cross-Validation? (y/n)")
    print("─"*60)
    choice = input("Enter choice: ").strip().lower()
    
    if choice == 'y':
        trainer.train_with_cross_validation(x_train, y_train)
    
    # Split training data for validation
    split_idx = int(len(x_train) * (1 - Config.VALIDATION_SPLIT))
    x_train_final = x_train[:split_idx]
    y_train_final = y_train[:split_idx]
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    
    # Train final model
    trainer.train_final_model(x_train_final, y_train_final, x_val, y_val)
    
    # Evaluate model
    metrics = trainer.evaluate_model(x_test, y_test)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nResults saved to: {Config.RESULTS_DIR}")
    print(f"  - Models: {Config.MODELS_DIR}")
    print(f"  - Plots: {Config.PLOTS_DIR}")
    print(f"  - Metrics: {Config.METRICS_DIR}")
    print("\n✓ CNN training pipeline completed successfully!")


if __name__ == "__main__":
    main()
