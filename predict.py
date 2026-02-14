"""
Inference script - Use trained model to predict on new EEG data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt

# Register custom layer
@keras.saving.register_keras_serializable()
class EEGAugmenter(layers.Layer):
    """Data augmentation layer for EEG signals."""
    
    def __init__(self, noise_std: float = 0.01, time_shift_max: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'noise_std': self.noise_std,
            'time_shift_max': self.time_shift_max
        })
        return config
    
    def call(self, x, training=None):
        if not training:
            return x
        noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=self.noise_std)
        x = x + noise
        if self.time_shift_max > 0:
            shift = tf.random.uniform([], -self.time_shift_max, self.time_shift_max + 1, dtype=tf.int32)
            x = tf.roll(x, shift, axis=1)
        return x

# Same channel order as training
EEG_CHANNELS = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
    'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
]

# Arabic word classes
ARABIC_CLASSES = [
    'اختر', 'اسفل', 'اعلى', 'انذار', 'ايقاف تشغيل', 'تشغيل',
    'جوع', 'حذف', 'حمام', 'دواء', 'عطش', 'لا', 'مسافة', 'نعم', 'يسار', 'يمين'
]


def butter_bandpass_filter(data, lowcut=4.0, highcut=8.0, fs=128.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered[i] = filtfilt(b, a, data[i])
    
    return filtered


def load_and_preprocess_eeg(
    filepath: str,
    norm_stats_path: str,
    fs: float = 128.0,
    epoch_length: int = 128
):
    """
    Load and preprocess a single EEG file.
    
    Args:
        filepath: Path to CSV file
        norm_stats_path: Path to normalization statistics
        fs: Sampling frequency
        epoch_length: Length of each epoch in samples
        
    Returns:
        Preprocessed epochs ready for prediction (n_epochs, 128, 14)
    """
    # Load CSV
    df = pd.read_csv(filepath, skiprows=1)
    eeg_data = df[EEG_CHANNELS].values.T  # (14, n_samples)
    
    # Remove NaN
    for i in range(eeg_data.shape[0]):
        mask = np.isnan(eeg_data[i])
        if np.any(mask):
            eeg_data[i][mask] = np.interp(
                np.where(mask)[0],
                np.where(~mask)[0],
                eeg_data[i][~mask]
            )
    
    # Bandpass filter
    filtered = butter_bandpass_filter(eeg_data, lowcut=4.0, highcut=8.0, fs=fs)
    
    # Segment into epochs
    n_channels, n_samples = filtered.shape
    epochs = []
    
    start = 0
    while start + epoch_length <= n_samples:
        epoch = filtered[:, start:start + epoch_length]
        # Remove DC offset
        epoch = epoch - epoch.mean(axis=1, keepdims=True)
        epochs.append(epoch)
        start += epoch_length
    
    if not epochs:
        return None
    
    epochs = np.array(epochs, dtype=np.float32)  # (n_epochs, 14, 128)
    
    # Load normalization stats
    norm_stats = np.load(norm_stats_path)
    mean = norm_stats['mean']
    std = norm_stats['std']
    
    # Normalize
    epochs = (epochs - mean[None, :, None]) / std[None, :, None]
    epochs = np.clip(epochs, -5.0, 5.0)
    
    # Transpose: (n_epochs, 14, 128) → (n_epochs, 128, 14)
    epochs = np.transpose(epochs, (0, 2, 1))
    
    return epochs


def predict_on_file(
    model_path: str,
    norm_stats_path: str,
    eeg_file: str,
    top_k: int = 3
):
    """
    Predict word class from EEG file.
    
    Args:
        model_path: Path to trained model
        norm_stats_path: Path to normalization stats
        eeg_file: Path to EEG CSV file
        top_k: Number of top predictions to show
    """
    print(f"\n{'='*70}")
    print("EEG WORD PREDICTION")
    print(f"{'='*70}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Load and preprocess data
    print(f"\nLoading EEG file: {Path(eeg_file).name}")
    epochs = load_and_preprocess_eeg(eeg_file, norm_stats_path)
    
    if epochs is None:
        print("❌ No valid epochs found in file")
        return
    
    print(f"✓ Preprocessed {len(epochs)} epochs")
    
    # Predict
    print("\nPredicting...")
    predictions = model.predict(epochs, verbose=0)
    
    # Average predictions across all epochs
    avg_prediction = predictions.mean(axis=0)
    
    # Get top k predictions
    top_indices = np.argsort(avg_prediction)[::-1][:top_k]
    
    print(f"\n{'='*70}")
    print(f"TOP {top_k} PREDICTIONS")
    print(f"{'='*70}")
    
    for i, idx in enumerate(top_indices, 1):
        word = ARABIC_CLASSES[idx]
        confidence = avg_prediction[idx] * 100
        bar = '█' * int(confidence / 2)
        print(f"{i}. {word:20s} {confidence:5.2f}% {bar}")
    
    # Show per-epoch predictions
    print(f"\n{'='*70}")
    print(f"PER-EPOCH PREDICTIONS ({len(epochs)} epochs)")
    print(f"{'='*70}")
    
    for i, pred in enumerate(predictions):
        pred_idx = np.argmax(pred)
        pred_word = ARABIC_CLASSES[pred_idx]
        confidence = pred[pred_idx] * 100
        print(f"Epoch {i+1:2d}: {pred_word:20s} ({confidence:.2f}%)")
    
    # Consensus
    epoch_predictions = [ARABIC_CLASSES[np.argmax(p)] for p in predictions]
    from collections import Counter
    most_common = Counter(epoch_predictions).most_common(3)
    
    print(f"\n{'='*70}")
    print("CONSENSUS VOTING")
    print(f"{'='*70}")
    
    for word, count in most_common:
        percentage = count / len(epochs) * 100
        print(f"{word:20s}: {count}/{len(epochs)} epochs ({percentage:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"FINAL PREDICTION: {ARABIC_CLASSES[np.argmax(avg_prediction)]}")
    print(f"{'='*70}\n")


def predict_on_directory(
    model_path: str,
    norm_stats_path: str,
    input_dir: str,
    output_csv: str = None
):
    """
    Predict on all CSV files in a directory.
    
    Args:
        model_path: Path to trained model
        norm_stats_path: Path to normalization stats
        input_dir: Directory containing EEG CSV files
        output_csv: Optional path to save results as CSV
    """
    print(f"\n{'='*70}")
    print("BATCH PREDICTION")
    print(f"{'='*70}")
    
    # Load model
    print(f"\nLoading model...")
    model = tf.keras.models.load_model(model_path)
    print("✓ Model loaded")
    
    # Find all CSV files
    input_path = Path(input_dir)
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_dir}")
        return
    
    print(f"\n✓ Found {len(csv_files)} CSV files")
    
    # Process each file
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Processing {csv_file.name}...")
        
        try:
            epochs = load_and_preprocess_eeg(str(csv_file), norm_stats_path)
            
            if epochs is None:
                print("  ⚠ No valid epochs")
                continue
            
            predictions = model.predict(epochs, verbose=0)
            avg_prediction = predictions.mean(axis=0)
            
            pred_idx = np.argmax(avg_prediction)
            pred_word = ARABIC_CLASSES[pred_idx]
            confidence = avg_prediction[pred_idx] * 100
            
            print(f"  ✓ Predicted: {pred_word} ({confidence:.2f}%)")
            
            results.append({
                'filename': csv_file.name,
                'predicted_word': pred_word,
                'confidence': confidence,
                'n_epochs': len(epochs)
            })
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total files: {len(csv_files)}")
    print(f"Successful predictions: {len(results)}")
    
    # Save to CSV if requested
    if output_csv and results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n✓ Results saved to {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Predict Arabic words from EEG data using trained model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='./outputs/best_model.keras',
        help='Path to trained model'
    )
    parser.add_argument(
        '--norm_stats',
        type=str,
        default='./outputs/normalization_stats.npz',
        help='Path to normalization statistics'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Single EEG CSV file to predict'
    )
    parser.add_argument(
        '--directory',
        type=str,
        help='Directory containing EEG CSV files'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        help='Save batch results to CSV'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=3,
        help='Number of top predictions to show'
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("Please train a model first:")
        print("  python train_areeg_words_cnn_transformer.py")
        return
    
    if not Path(args.norm_stats).exists():
        print(f"❌ Normalization stats not found: {args.norm_stats}")
        return
    
    # Single file or batch
    if args.file:
        predict_on_file(args.model, args.norm_stats, args.file, args.top_k)
    elif args.directory:
        predict_on_directory(args.model, args.norm_stats, args.directory, args.output_csv)
    else:
        print("❌ Please specify --file or --directory")
        print("\nExamples:")
        print("  python predict.py --file ./data/select/recording.csv")
        print("  python predict.py --directory ./new_recordings --output_csv results.csv")


if __name__ == '__main__':
    main()
