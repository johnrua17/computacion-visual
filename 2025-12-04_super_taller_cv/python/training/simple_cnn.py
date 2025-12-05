#!/usr/bin/env python
"""
SIMPLE CNN Training - Memory Optimized Version
Quick training script that works with limited RAM
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Memory optimization settings
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU'), True) if tf.config.list_physical_devices('GPU') else None

class SimpleConfig:
    """Simple configuration for memory-constrained training"""
    IMAGE_SIZE = (32, 32)  # Original CIFAR-10 size
    NUM_CLASSES = 10
    BATCH_SIZE = 16        # Small batch size
    EPOCHS = 5             # Quick training
    LEARNING_RATE = 0.001

def load_data():
    """Load CIFAR-10 with minimal preprocessing"""
    print("Loading CIFAR-10 dataset...")
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, SimpleConfig.NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, SimpleConfig.NUM_CLASSES)
    
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_simple_model():
    """Create a simple CNN model optimized for memory"""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(32, 32, 3)),
        
        # Block 1
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(SimpleConfig.NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=SimpleConfig.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Main training function"""
    print("="*60)
    print("  Simple CNN Training - Memory Optimized")
    print("="*60)
    
    # Create directories
    os.makedirs('../../../results/models', exist_ok=True)
    os.makedirs('../../../results/plots', exist_ok=True)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Create model
    print("\\nCreating model...")
    model = create_simple_model()
    model.summary()
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Train model
    print("\\nStarting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=SimpleConfig.BATCH_SIZE,
        epochs=SimpleConfig.EPOCHS,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"../../../results/models/simple_cnn_{timestamp}.h5"
    model.save(model_path)
    print(f"\\nModel saved: {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_path = f"../../../results/plots/simple_cnn_history_{timestamp}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Training plots saved: {plot_path}")
    
    # Classification report
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=['airplane', 'automobile', 'bird', 'cat', 'deer',
                                          'dog', 'frog', 'horse', 'ship', 'truck']))
    
    print("\\n" + "="*60)
    print(f"  Training completed successfully!")
    print(f"  Final test accuracy: {test_accuracy:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()