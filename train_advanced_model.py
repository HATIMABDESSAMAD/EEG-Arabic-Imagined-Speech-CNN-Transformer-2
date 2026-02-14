"""
Arabic Imagined Speech EEG Classification - ADVANCED VERSION
Architecture: Multi-Band CNN + Transformer with Advanced Techniques

Improvements over balanced version:
1. Multi-band filtering (Theta + Alpha + Beta)
2. Higher overlap (0.85) for more training data
3. Larger architecture (more capacity since low overfitting)
4. Spatial attention (channel attention)
5. Deeper transformer (2 layers)
6. Cosine annealing learning rate
7. Better augmentation with channel dropout
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
)

warnings.filterwarnings('ignore')

# ============================================================================
# REPRODUCIBILITY & GPU
# ============================================================================

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✓ Seeds fixed to {seed}")


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s)")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    else:
        print("⚠ No GPU found. Running on CPU.")
    return len(gpus) > 0


# ============================================================================
# CONSTANTS
# ============================================================================

EEG_CHANNELS = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
    'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
]

# Multi-band frequency ranges
FREQUENCY_BANDS = {
    'theta': (4, 8),    # Theta: 4-8 Hz
    'alpha': (8, 13),   # Alpha: 8-13 Hz
    'beta': (13, 30),   # Beta: 13-30 Hz
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_eeg_csv(filepath: str) -> Optional[np.ndarray]:
    try:
        df = pd.read_csv(filepath, skiprows=1)
        eeg_data = df[EEG_CHANNELS].values.T
        
        if np.any(np.isnan(eeg_data)):
            for i in range(eeg_data.shape[0]):
                mask = np.isnan(eeg_data[i])
                if np.any(mask):
                    eeg_data[i][mask] = np.interp(
                        np.where(mask)[0],
                        np.where(~mask)[0],
                        eeg_data[i][~mask]
                    )
        return eeg_data
    except Exception as e:
        return None


def discover_dataset(data_root: str) -> Dict[str, List[str]]:
    data_root = Path(data_root)
    dataset = {}
    
    for word_dir in data_root.iterdir():
        if word_dir.is_dir():
            word_name = word_dir.name
            csv_files = list(word_dir.glob('*.csv'))
            if csv_files:
                dataset[word_name] = [str(f) for f in csv_files]
    
    total_files = sum(len(files) for files in dataset.values())
    print(f"✓ Dataset: {len(dataset)} classes, {total_files} files")
    
    return dataset


# ============================================================================
# MULTI-BAND SIGNAL PROCESSING
# ============================================================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = max(lowcut / nyquist, 0.01)
    high = min(highcut / nyquist, 0.99)
    
    if low >= high:
        return data
    
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        try:
            filtered[i] = filtfilt(b, a, data[i])
        except:
            filtered[i] = data[i]
    return filtered


def extract_multiband_features(data: np.ndarray, fs: float = 128.0) -> np.ndarray:
    """
    Extract multi-band features: Theta + Alpha + Beta
    Output shape: (n_channels * 3, n_samples)
    """
    bands_data = []
    
    for band_name, (low, high) in FREQUENCY_BANDS.items():
        filtered = butter_bandpass_filter(data, low, high, fs)
        bands_data.append(filtered)
    
    # Concatenate along channel dimension
    # Shape: (n_channels * n_bands, n_samples)
    multi_band = np.concatenate(bands_data, axis=0)
    
    return multi_band


def segment_signal(data, epoch_length, overlap=0.0):
    n_channels, n_samples = data.shape
    
    if n_samples < epoch_length:
        return []
    
    step = int(epoch_length * (1 - overlap))
    epochs = []
    
    start = 0
    while start + epoch_length <= n_samples:
        epoch = data[:, start:start + epoch_length]
        epochs.append(epoch)
        start += step
    
    return epochs


# ============================================================================
# DATASET PREPARATION
# ============================================================================

class AdvancedEEGProcessor:
    """Process EEG with multi-band filtering."""
    
    def __init__(
        self,
        fs: float = 128.0,
        epoch_length: float = 1.0,
        overlap: float = 0.85,  # Higher overlap
        use_multiband: bool = True
    ):
        self.fs = fs
        self.epoch_length_samples = int(epoch_length * fs)
        self.overlap = overlap
        self.use_multiband = use_multiband
        self.mean = None
        self.std = None
        
        n_features = 14 * 3 if use_multiband else 14
        
        print(f"\n✓ Advanced EEG Processor:")
        print(f"  Multi-band: {use_multiband} (Theta+Alpha+Beta)")
        print(f"  Features per epoch: {n_features} channels × {self.epoch_length_samples} samples")
        print(f"  Overlap: {overlap * 100:.0f}%")
    
    def process_file(self, filepath: str) -> List[np.ndarray]:
        eeg_data = load_eeg_csv(filepath)
        if eeg_data is None:
            return []
        
        # Multi-band filtering
        if self.use_multiband:
            processed = extract_multiband_features(eeg_data, self.fs)
        else:
            processed = butter_bandpass_filter(eeg_data, 4, 30, self.fs)
        
        # Segment
        epochs = segment_signal(processed, self.epoch_length_samples, self.overlap)
        
        # Remove DC offset
        epochs = [ep - ep.mean(axis=1, keepdims=True) for ep in epochs]
        
        return epochs
    
    def fit_normalization(self, train_epochs: np.ndarray):
        self.mean = train_epochs.mean(axis=(0, 2), keepdims=False)
        self.std = train_epochs.std(axis=(0, 2), keepdims=False)
        self.std = np.maximum(self.std, 1e-8)
        print(f"✓ Normalization fitted on {len(train_epochs)} epochs")
    
    def normalize_epochs(self, epochs: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Must call fit_normalization() first")
        normalized = (epochs - self.mean[None, :, None]) / self.std[None, :, None]
        return np.clip(normalized, -5.0, 5.0)
    
    def save_stats(self, filepath: str):
        np.savez(filepath, mean=self.mean, std=self.std)


def prepare_dataset(dataset_dict, processor, dry_run=False):
    all_epochs = []
    all_labels = []
    
    label_encoder = LabelEncoder()
    label_encoder.fit(list(dataset_dict.keys()))
    
    print(f"\n{'='*60}")
    print("PROCESSING FILES (ADVANCED MODE)")
    print(f"{'='*60}")
    
    for class_name, file_list in sorted(dataset_dict.items()):
        class_epochs = []
        files_to_process = file_list[:1] if dry_run else file_list
        
        for filepath in files_to_process:
            epochs = processor.process_file(filepath)
            class_epochs.extend(epochs)
        
        if class_epochs:
            class_label = label_encoder.transform([class_name])[0]
            all_epochs.extend(class_epochs)
            all_labels.extend([class_label] * len(class_epochs))
            print(f"  {class_name:15s}: {len(files_to_process):3d} files → {len(class_epochs):5d} epochs")
    
    X = np.array(all_epochs, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    print(f"\n{'='*60}")
    print(f"Total: {len(X)} epochs, shape per epoch: {X.shape[1:]}")
    print(f"{'='*60}")
    
    return X, y, label_encoder


# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

@keras.saving.register_keras_serializable()
class AdvancedEEGAugmenter(layers.Layer):
    """Advanced augmentation with channel dropout."""
    
    def __init__(
        self,
        noise_std: float = 0.02,
        time_shift_max: int = 8,
        channel_dropout: float = 0.1,  # New: drop random channels
        **kwargs
    ):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.channel_dropout = channel_dropout
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_std': self.noise_std,
            'time_shift_max': self.time_shift_max,
            'channel_dropout': self.channel_dropout
        })
        return config
    
    def call(self, x, training=None):
        if not training:
            return x
        
        # 1. Gaussian noise
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std)
        x = x + noise
        
        # 2. Time shift
        if self.time_shift_max > 0:
            shift = tf.random.uniform([], -self.time_shift_max, self.time_shift_max + 1, dtype=tf.int32)
            x = tf.roll(x, shift, axis=1)
        
        # 3. Channel dropout (randomly zero some channels)
        if self.channel_dropout > 0:
            mask = tf.random.uniform([1, 1, tf.shape(x)[2]]) > self.channel_dropout
            mask = tf.cast(mask, x.dtype)
            x = x * mask / (1.0 - self.channel_dropout)  # Scale to maintain variance
        
        return x


# ============================================================================
# CHANNEL ATTENTION (SQUEEZE-EXCITATION)
# ============================================================================

@keras.saving.register_keras_serializable()
class ChannelAttention(layers.Layer):
    """Squeeze-and-Excitation attention for channels."""
    
    def __init__(self, reduction_ratio: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(channels // self.reduction_ratio, 4)
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc1 = layers.Dense(reduced, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
    
    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config
    
    def call(self, x):
        # Squeeze: global average pooling
        squeezed = self.global_pool(x)  # (batch, channels)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        excited = self.fc1(squeezed)
        excited = self.fc2(excited)  # (batch, channels)
        
        # Scale: multiply input by attention weights
        return x * tf.expand_dims(excited, axis=1)


# ============================================================================
# COSINE ANNEALING LEARNING RATE
# ============================================================================

class CosineAnnealingScheduler(Callback):
    """Cosine annealing with warm restarts."""
    
    def __init__(self, initial_lr, min_lr, epochs_per_cycle):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.epochs_per_cycle = epochs_per_cycle
    
    def on_epoch_begin(self, epoch, logs=None):
        cycle_position = epoch % self.epochs_per_cycle
        lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
             (1 + math.cos(math.pi * cycle_position / self.epochs_per_cycle))
        
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)


# ============================================================================
# ADVANCED MODEL ARCHITECTURE
# ============================================================================

def create_advanced_model(
    input_shape: Tuple[int, int] = (128, 42),  # 14 channels * 3 bands = 42
    num_classes: int = 16,
    cnn_filters: int = 72,      # Larger
    transformer_layers: int = 2, # Deeper
    num_heads: int = 6,
    ff_dim: int = 144,
    dropout: float = 0.25,       # Slightly lower since more data
    weight_decay: float = 2e-4
) -> keras.Model:
    """
    Advanced CNN + Transformer with:
    - Channel attention (SE blocks)
    - Deeper transformer (2 layers)
    - Larger capacity
    """
    time_steps, n_channels = input_shape
    
    inputs = layers.Input(shape=input_shape, name='eeg_input')
    x = inputs
    
    # Advanced augmentation
    x = AdvancedEEGAugmenter(
        noise_std=0.02,
        time_shift_max=8,
        channel_dropout=0.1
    )(x)
    
    # ========================================================================
    # CNN BLOCK with Channel Attention
    # ========================================================================
    
    # Channel projection
    x = layers.Conv1D(
        cnn_filters, kernel_size=1, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    # Temporal conv 1 + SE attention
    x = layers.Conv1D(
        cnn_filters, kernel_size=7, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = ChannelAttention(reduction_ratio=4)(x)  # SE block
    x = layers.Dropout(dropout)(x)
    
    # Temporal conv 2 + SE attention
    x = layers.Conv1D(
        cnn_filters, kernel_size=5, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = ChannelAttention(reduction_ratio=4)(x)  # SE block
    x = layers.Dropout(dropout)(x)
    
    # Temporal conv 3 (additional layer)
    x = layers.Conv1D(
        cnn_filters, kernel_size=3, padding='same',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(dropout)(x)
    
    # Downsampling
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)  # 128 -> 64
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)  # 64 -> 32
    
    # ========================================================================
    # DEEPER TRANSFORMER (2 layers)
    # ========================================================================
    
    seq_len = x.shape[1]
    embed_dim = x.shape[2]
    
    # Learned positional encoding
    pos_embedding = layers.Embedding(
        input_dim=seq_len, output_dim=embed_dim,
        embeddings_regularizer=regularizers.l2(weight_decay)
    )(tf.range(seq_len))
    x = x + pos_embedding
    x = layers.Dropout(dropout * 0.5)(x)
    
    # Transformer layers
    for i in range(transformer_layers):
        # Multi-head attention
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout * 0.5,
            kernel_regularizer=regularizers.l2(weight_decay)
        )(x, x)
        
        x = layers.Add()([x, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feedforward with expansion
        ff = layers.Dense(
            ff_dim * 2,  # Larger expansion
            activation='gelu',
            kernel_regularizer=regularizers.l2(weight_decay)
        )(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(
            embed_dim,
            kernel_regularizer=regularizers.l2(weight_decay)
        )(ff)
        ff = layers.Dropout(dropout)(ff)
        
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # ========================================================================
    # CLASSIFICATION HEAD
    # ========================================================================
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    
    # Two dense layers
    x = layers.Dense(
        128, activation='gelu',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(
        64, activation='gelu',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.Dropout(dropout * 0.5)(x)
    
    outputs = layers.Dense(
        num_classes, activation='softmax',
        dtype='float32',
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Advanced_MultiBand_CNN_Transformer')
    
    return model


def compile_model(model, learning_rate=1e-3, weight_decay=2e-4):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model, X_train, y_train, X_val, y_val,
    batch_size=32, epochs=250, output_dir='.'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(output_dir / 'best_model.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Cosine annealing
        CosineAnnealingScheduler(
            initial_lr=1e-3,
            min_lr=1e-5,
            epochs_per_cycle=50
        )
    ]
    
    print(f"\n{'='*60}")
    print("ADVANCED TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"Cosine annealing: ENABLED")
    print(f"{'='*60}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    # Save history
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Analysis
    final_train = history_dict['accuracy'][-1]
    final_val = history_dict['val_accuracy'][-1]
    gap = final_train - final_val
    
    print(f"\n{'='*60}")
    print("TRAINING ANALYSIS")
    print(f"{'='*60}")
    print(f"Final train: {final_train:.4f}")
    print(f"Final val:   {final_val:.4f}")
    print(f"Gap:         {gap:.4f} ({gap*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return history_dict


def evaluate_model(model, X_test, y_test, label_encoder, output_dir='.'):
    output_dir = Path(output_dir)
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"🎯 Test F1-Score: {f1_macro:.4f} ({f1_macro*100:.2f}%)")
    
    class_names = label_encoder.classes_
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix (ADVANCED)\nAccuracy: {accuracy:.4f}, F1: {f1_macro:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    
    # Save metrics
    metrics = {
        'test_accuracy': float(accuracy),
        'test_f1_macro': float(f1_macro),
        'class_names': class_names.tolist()
    }
    with open(output_dir / 'test_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to {output_dir}")
    
    return accuracy, f1_macro


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train ADVANCED Multi-Band CNN+Transformer')
    
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./outputs_advanced')
    parser.add_argument('--overlap', type=float, default=0.85)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--no_multiband', action='store_true', help='Disable multi-band')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("ADVANCED EEG CLASSIFIER")
    print("Multi-Band (Theta+Alpha+Beta) | Channel Attention | Deep Transformer")
    print(f"{'='*60}\n")
    
    set_seeds(args.seed)
    configure_gpu()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Discover dataset
    dataset_dict = discover_dataset(args.data_root)
    if not dataset_dict:
        print("❌ No data found.")
        return
    
    # Process with multi-band
    use_multiband = not args.no_multiband
    processor = AdvancedEEGProcessor(
        fs=128.0,
        epoch_length=1.0,
        overlap=args.overlap,
        use_multiband=use_multiband
    )
    
    X, y, label_encoder = prepare_dataset(dataset_dict, processor, dry_run=args.dry_run)
    
    if len(X) == 0:
        print("❌ No epochs generated.")
        return
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=args.seed
    )
    
    print(f"\nData split: Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")
    
    # Normalize
    processor.fit_normalization(X_train)
    X_train = processor.normalize_epochs(X_train)
    X_val = processor.normalize_epochs(X_val)
    X_test = processor.normalize_epochs(X_test)
    
    # Transpose to (batch, time, channels)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    
    processor.save_stats(output_dir / 'normalization_stats.npz')
    
    # Create model
    n_channels = 42 if use_multiband else 14
    
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE (ADVANCED)")
    print(f"{'='*60}")
    
    model = create_advanced_model(
        input_shape=(128, n_channels),
        num_classes=len(dataset_dict),
        cnn_filters=72,
        transformer_layers=2,
        num_heads=6,
        ff_dim=144,
        dropout=0.25,
        weight_decay=2e-4
    )
    
    model = compile_model(model, learning_rate=1e-3, weight_decay=2e-4)
    model.summary()
    
    # Train
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=output_dir
    )
    
    # Load best and evaluate
    model = keras.models.load_model(output_dir / 'best_model.keras')
    accuracy, f1 = evaluate_model(model, X_test, y_test, label_encoder, output_dir)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"📁 Outputs: {output_dir}")
    print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")
    print(f"🎯 Test F1-Score: {f1*100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
