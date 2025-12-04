"""
Fine-Tuning with Pre-trained Models
Subsystem 5: Training and Model Comparison

This script implements transfer learning using pre-trained models (ResNet50, MobileNetV2)
with fine-tuning capabilities and comprehensive performance analysis.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, MobileNetV2, VGG16, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


class FineTuningConfig:
    """Configuration for fine-tuning experiments"""
    
    # Data parameters
    IMAGE_SIZE = (224, 224)  # Standard size for pre-trained models
    BATCH_SIZE = 32
    NUM_CLASSES = 10
    
    # Training parameters
    EPOCHS_FEATURE_EXTRACTION = 10  # Train only top layers
    EPOCHS_FINE_TUNING = 30  # Fine-tune entire model
    LEARNING_RATE_INITIAL = 0.001
    LEARNING_RATE_FINE_TUNE = 0.0001
    
    # Fine-tuning parameters
    UNFREEZE_LAYERS = 50  # Number of layers to unfreeze for fine-tuning
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
    PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
    METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
    
    # Model configurations
    MODELS_CONFIG = {
        'resnet50': {
            'class': ResNet50,
            'name': 'ResNet50',
            'preprocessing': tf.keras.applications.resnet50.preprocess_input
        },
        'mobilenetv2': {
            'class': MobileNetV2,
            'name': 'MobileNetV2',
            'preprocessing': tf.keras.applications.mobilenet_v2.preprocess_input
        },
        'vgg16': {
            'class': VGG16,
            'name': 'VGG16',
            'preprocessing': tf.keras.applications.vgg16.preprocess_input
        },
        'inceptionv3': {
            'class': InceptionV3,
            'name': 'InceptionV3',
            'preprocessing': tf.keras.applications.inception_v3.preprocess_input
        }
    }


class TransferLearningModel:
    """Build and manage transfer learning models"""
    
    def __init__(self, model_name='resnet50'):
        """
        Initialize transfer learning model
        
        Args:
            model_name: Name of pre-trained model ('resnet50', 'mobilenetv2', etc.)
        """
        self.model_name = model_name
        self.config = FineTuningConfig.MODELS_CONFIG[model_name]
        self.model = None
        self.history_feature_extraction = None
        self.history_fine_tuning = None
        self.metrics = {}
    
    def build_model(self, num_classes, trainable_base=False):
        """
        Build transfer learning model
        
        Args:
            num_classes: Number of output classes
            trainable_base: Whether base model is trainable
            
        Returns:
            Compiled Keras model
        """
        print(f"\nBuilding {self.config['name']} model...")
        
        # Load pre-trained base model
        base_model = self.config['class'](
            weights='imagenet',
            include_top=False,
            input_shape=(*FineTuningConfig.IMAGE_SIZE, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = trainable_base
        
        # Build complete model
        inputs = keras.Input(shape=(*FineTuningConfig.IMAGE_SIZE, 3))
        
        # Preprocessing
        x = inputs
        
        # Base model
        x = base_model(x, training=False)
        
        # Add custom top layers
        x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.Dense(256, activation='relu', name='fc2')(x)
        x = layers.Dropout(0.3, name='dropout2')(x)
        outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
        
        model = keras.Model(inputs, outputs, name=f'{self.model_name}_transfer')
        
        return model, base_model
    
    def compile_model(self, model, learning_rate):
        """Compile model with optimizer and metrics"""
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def feature_extraction_training(self, x_train, y_train, x_val, y_val):
        """
        Phase 1: Feature extraction (train only top layers)
        """
        print(f"\n{'='*60}")
        print(f"Phase 1: Feature Extraction - {self.config['name']}")
        print(f"{'='*60}\n")
        
        # Build model with frozen base
        self.model, base_model = self.build_model(
            FineTuningConfig.NUM_CLASSES,
            trainable_base=False
        )
        
        # Compile model
        self.model = self.compile_model(
            self.model,
            FineTuningConfig.LEARNING_RATE_INITIAL
        )
        
        print(f"Base model frozen: {not base_model.trainable}")
        print(f"Total parameters: {self.model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(FineTuningConfig.MODELS_DIR, 
                           f'{self.model_name}_feature_extraction_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history_feature_extraction = self.model.fit(
            x_train, y_train,
            batch_size=FineTuningConfig.BATCH_SIZE,
            epochs=FineTuningConfig.EPOCHS_FEATURE_EXTRACTION,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_auc = self.model.evaluate(
            x_val, y_val, verbose=0
        )
        
        print(f"\nFeature Extraction Results:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        self.metrics['feature_extraction'] = {
            'val_accuracy': float(val_acc),
            'val_loss': float(val_loss),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'val_auc': float(val_auc)
        }
        
        return self.history_feature_extraction
    
    def fine_tuning_training(self, x_train, y_train, x_val, y_val, base_model):
        """
        Phase 2: Fine-tuning (unfreeze and train entire model)
        """
        print(f"\n{'='*60}")
        print(f"Phase 2: Fine-Tuning - {self.config['name']}")
        print(f"{'='*60}\n")
        
        # Unfreeze base model layers
        base_model.trainable = True
        
        # Freeze early layers, unfreeze later layers
        num_layers = len(base_model.layers)
        freeze_until = max(0, num_layers - FineTuningConfig.UNFREEZE_LAYERS)
        
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False
        
        print(f"Total layers in base model: {num_layers}")
        print(f"Frozen layers: {freeze_until}")
        print(f"Trainable layers: {num_layers - freeze_until}")
        
        # Re-compile with lower learning rate
        self.model = self.compile_model(
            self.model,
            FineTuningConfig.LEARNING_RATE_FINE_TUNE
        )
        
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in self.model.trainable_weights]):,}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            ),
            ModelCheckpoint(
                os.path.join(FineTuningConfig.MODELS_DIR, 
                           f'{self.model_name}_fine_tuned_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        self.history_fine_tuning = self.model.fit(
            x_train, y_train,
            batch_size=FineTuningConfig.BATCH_SIZE,
            epochs=FineTuningConfig.EPOCHS_FINE_TUNING,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        val_loss, val_acc, val_prec, val_rec, val_auc = self.model.evaluate(
            x_val, y_val, verbose=0
        )
        
        print(f"\nFine-Tuning Results:")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        
        self.metrics['fine_tuning'] = {
            'val_accuracy': float(val_acc),
            'val_loss': float(val_loss),
            'val_precision': float(val_prec),
            'val_recall': float(val_rec),
            'val_auc': float(val_auc)
        }
        
        # Save final model
        model_path = os.path.join(FineTuningConfig.MODELS_DIR, 
                                 f'{self.model_name}_final.h5')
        self.model.save(model_path)
        print(f"\n✓ Model saved: {model_path}")
        
        return self.history_fine_tuning
    
    def evaluate_model(self, x_test, y_test):
        """Comprehensive evaluation on test set"""
        
        print(f"\n{'='*60}")
        print(f"Test Evaluation - {self.config['name']}")
        print(f"{'='*60}\n")
        
        # Predict
        y_pred_proba = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_acc, test_prec, test_rec, test_auc = self.model.evaluate(
            x_test, y_test, verbose=0
        )
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_rec:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        # Store metrics
        self.metrics['test'] = {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'auc': float(test_auc)
        }
        
        # Save metrics
        metrics_path = os.path.join(FineTuningConfig.METRICS_DIR, 
                                   f'{self.model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        return self.metrics, y_true, y_pred
    
    def plot_training_progress(self):
        """Plot training progress for both phases"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Combine histories
        if self.history_fine_tuning:
            combined_acc = (self.history_feature_extraction.history['accuracy'] + 
                          self.history_fine_tuning.history['accuracy'])
            combined_val_acc = (self.history_feature_extraction.history['val_accuracy'] + 
                              self.history_fine_tuning.history['val_accuracy'])
            combined_loss = (self.history_feature_extraction.history['loss'] + 
                           self.history_fine_tuning.history['loss'])
            combined_val_loss = (self.history_feature_extraction.history['val_loss'] + 
                               self.history_fine_tuning.history['val_loss'])
            
            fe_epochs = len(self.history_feature_extraction.history['accuracy'])
            total_epochs = len(combined_acc)
            
            # Plot accuracy
            axes[0, 0].plot(combined_acc, label='Train Accuracy', linewidth=2)
            axes[0, 0].plot(combined_val_acc, label='Val Accuracy', linewidth=2)
            axes[0, 0].axvline(x=fe_epochs, color='r', linestyle='--', 
                             label='Start Fine-Tuning', linewidth=2)
            axes[0, 0].set_title(f'{self.config["name"]} - Accuracy', 
                               fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot loss
            axes[0, 1].plot(combined_loss, label='Train Loss', linewidth=2)
            axes[0, 1].plot(combined_val_loss, label='Val Loss', linewidth=2)
            axes[0, 1].axvline(x=fe_epochs, color='r', linestyle='--', 
                             label='Start Fine-Tuning', linewidth=2)
            axes[0, 1].set_title(f'{self.config["name"]} - Loss', 
                               fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot precision
            combined_prec = (self.history_feature_extraction.history['precision'] + 
                           self.history_fine_tuning.history['precision'])
            combined_val_prec = (self.history_feature_extraction.history['val_precision'] + 
                               self.history_fine_tuning.history['val_precision'])
            
            axes[1, 0].plot(combined_prec, label='Train Precision', linewidth=2)
            axes[1, 0].plot(combined_val_prec, label='Val Precision', linewidth=2)
            axes[1, 0].axvline(x=fe_epochs, color='r', linestyle='--', 
                             label='Start Fine-Tuning', linewidth=2)
            axes[1, 0].set_title(f'{self.config["name"]} - Precision', 
                               fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot recall
            combined_rec = (self.history_feature_extraction.history['recall'] + 
                          self.history_fine_tuning.history['recall'])
            combined_val_rec = (self.history_feature_extraction.history['val_recall'] + 
                              self.history_fine_tuning.history['val_recall'])
            
            axes[1, 1].plot(combined_rec, label='Train Recall', linewidth=2)
            axes[1, 1].plot(combined_val_rec, label='Val Recall', linewidth=2)
            axes[1, 1].axvline(x=fe_epochs, color='r', linestyle='--', 
                             label='Start Fine-Tuning', linewidth=2)
            axes[1, 1].set_title(f'{self.config["name"]} - Recall', 
                               fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(FineTuningConfig.PLOTS_DIR, 
                                f'{self.model_name}_training_history.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def load_and_preprocess_data(model_name='resnet50'):
    """Load and preprocess data for transfer learning"""
    
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Resize images to required size
    x_train = tf.image.resize(x_train, FineTuningConfig.IMAGE_SIZE).numpy()
    x_test = tf.image.resize(x_test, FineTuningConfig.IMAGE_SIZE).numpy()
    
    # Apply model-specific preprocessing
    preprocess_fn = FineTuningConfig.MODELS_CONFIG[model_name]['preprocessing']
    x_train = preprocess_fn(x_train)
    x_test = preprocess_fn(x_test)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, FineTuningConfig.NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, FineTuningConfig.NUM_CLASSES)
    
    # Split training data for validation
    split_idx = int(len(x_train) * 0.8)
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def main():
    """Main fine-tuning pipeline"""
    
    print("\n" + "="*60)
    print("Fine-Tuning with Pre-trained Models - Subsystem 5")
    print("="*60 + "\n")
    
    # Create directories
    os.makedirs(FineTuningConfig.MODELS_DIR, exist_ok=True)
    os.makedirs(FineTuningConfig.PLOTS_DIR, exist_ok=True)
    os.makedirs(FineTuningConfig.METRICS_DIR, exist_ok=True)
    
    # Select model
    print("Available models:")
    for i, (key, config) in enumerate(FineTuningConfig.MODELS_CONFIG.items(), 1):
        print(f"  {i}. {config['name']} ({key})")
    
    print("\nEnter model numbers to train (comma-separated, e.g., 1,2):")
    print("Or press Enter to train all models")
    choice = input("Choice: ").strip()
    
    if choice:
        indices = [int(x.strip()) - 1 for x in choice.split(',')]
        selected_models = [list(FineTuningConfig.MODELS_CONFIG.keys())[i] for i in indices]
    else:
        selected_models = list(FineTuningConfig.MODELS_CONFIG.keys())
    
    # Train each selected model
    all_results = {}
    
    for model_name in selected_models:
        print(f"\n{'#'*60}")
        print(f"Training {FineTuningConfig.MODELS_CONFIG[model_name]['name']}")
        print(f"{'#'*60}\n")
        
        # Load data
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data(model_name)
        
        # Initialize model
        transfer_model = TransferLearningModel(model_name)
        
        # Phase 1: Feature extraction
        transfer_model.feature_extraction_training(x_train, y_train, x_val, y_val)
        
        # Phase 2: Fine-tuning
        _, base_model = transfer_model.build_model(FineTuningConfig.NUM_CLASSES)
        transfer_model.fine_tuning_training(x_train, y_train, x_val, y_val, base_model)
        
        # Evaluate
        metrics, y_true, y_pred = transfer_model.evaluate_model(x_test, y_test)
        
        # Plot training progress
        transfer_model.plot_training_progress()
        
        # Store results
        all_results[model_name] = metrics
    
    # Save combined results
    with open(os.path.join(FineTuningConfig.METRICS_DIR, 'all_models_comparison.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "="*60)
    print("Fine-Tuning Complete!")
    print("="*60)
    print(f"\nResults saved to: {FineTuningConfig.RESULTS_DIR}")
    print("\n✓ Fine-tuning pipeline completed successfully!")


if __name__ == "__main__":
    main()
