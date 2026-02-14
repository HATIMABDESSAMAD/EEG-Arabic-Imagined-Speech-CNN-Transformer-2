"""
Streamlit Application for Arabic EEG Imagined Speech Classification
CNN + Transformer Model for 16 Arabic Words Classification
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt
from pathlib import Path
from collections import Counter
import os

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Arabic EEG Speech Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-word {
        font-size: 3rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .confidence-text {
        font-size: 1.5rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .word-table {
        direction: rtl;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Custom Layers (Required for model loading)
# ============================================================================

# Note: No decorators - using custom_objects dict for compatibility
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


class AdvancedEEGAugmenter(layers.Layer):
    """Advanced augmentation with channel dropout."""
    
    def __init__(
        self,
        noise_std: float = 0.02,
        time_shift_max: int = 8,
        channel_dropout: float = 0.1,
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
            x = x * mask / (1.0 - self.channel_dropout)
        
        return x


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
# Constants
# ============================================================================
EEG_CHANNELS = [
    'EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
    'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4'
]

ARABIC_CLASSES = [
    'اختر', 'اسفل', 'اعلى', 'انذار', 'ايقاف تشغيل', 'تشغيل',
    'جوع', 'حذف', 'حمام', 'دواء', 'عطش', 'لا', 'مسافة', 'نعم', 'يسار', 'يمين'
]

ENGLISH_TRANSLATIONS = {
    'اختر': 'Select', 'اسفل': 'Down', 'اعلى': 'Up', 'انذار': 'Alarm',
    'ايقاف تشغيل': 'Stop', 'تشغيل': 'Start', 'جوع': 'Hunger', 'حذف': 'Delete',
    'حمام': 'Bathroom', 'دواء': 'Medicine', 'عطش': 'Thirst', 'لا': 'No',
    'مسافة': 'Space', 'نعم': 'Yes', 'يسار': 'Left', 'يمين': 'Right'
}

# Multi-band frequency ranges (same as training)
FREQUENCY_BANDS = {
    'theta': (4, 8),    # Theta: 4-8 Hz
    'alpha': (8, 13),   # Alpha: 8-13 Hz
    'beta': (13, 30),   # Beta: 13-30 Hz
}

MODEL_PATH = "outputs_advanced/best_model.keras"
WEIGHTS_PATH = "outputs_advanced/model_weights.weights.h5"
POS_EMB_PATH = "outputs_advanced/position_embedding.npy"
NORM_STATS_PATH = "outputs_advanced/normalization_stats.npz"

# ============================================================================
# Processing Functions
# ============================================================================
def butter_bandpass_filter(data, lowcut, highcut, fs=128.0, order=4):
    """Apply Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
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


@st.cache_resource
def load_model():
    """Load the trained model by rebuilding architecture and loading weights."""
    # Check if weights file exists (Keras 3 compatible)
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = rebuild_model_architecture()
            model.load_weights(WEIGHTS_PATH)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.warning(f"Weight loading failed: {e}")
    
    # Fallback: try loading original model (only works with Keras 2)
    if os.path.exists(MODEL_PATH):
        custom_objects = {
            'EEGAugmenter': EEGAugmenter,
            'AdvancedEEGAugmenter': AdvancedEEGAugmenter,
            'ChannelAttention': ChannelAttention,
            'Custom>EEGAugmenter': EEGAugmenter,
            'Custom>AdvancedEEGAugmenter': AdvancedEEGAugmenter,
            'Custom>ChannelAttention': ChannelAttention,
        }
        try:
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return None
    
    st.error("No model files found!")
    return None


class PositionEmbeddingLayer(layers.Layer):
    """Custom layer for adding pre-trained position embeddings."""
    
    def __init__(self, position_embedding, **kwargs):
        super().__init__(**kwargs)
        self.position_embedding = tf.constant(position_embedding, dtype=tf.float32)
    
    def call(self, x):
        return x + self.position_embedding
    
    def get_config(self):
        config = super().get_config()
        return config


def rebuild_model_architecture():
    """Rebuild the exact model architecture for weight loading (Keras 3 compatible)."""
    from tensorflow.keras import regularizers
    
    input_shape = (128, 42)
    num_classes = 16
    cnn_filters = 72
    transformer_layers = 2
    num_heads = 6
    ff_dim = 144
    dropout = 0.25
    weight_decay = 2e-4
    
    time_steps, n_channels = input_shape
    
    # Load position embedding from file
    pos_emb = None
    if os.path.exists(POS_EMB_PATH):
        pos_emb = np.load(POS_EMB_PATH).astype(np.float32)
    else:
        # If no saved embedding, use zeros (will work but may affect accuracy)
        pos_emb = np.zeros((32, cnn_filters), dtype=np.float32)
    
    inputs = layers.Input(shape=input_shape, name='eeg_input')
    x = inputs
    
    # Advanced augmentation (training only, passthrough at inference)
    x = AdvancedEEGAugmenter(noise_std=0.02, time_shift_max=8, channel_dropout=0.1, name='advanced_eeg_augmenter')(x)
    
    # CNN Block with Channel Attention
    x = layers.Conv1D(cnn_filters, kernel_size=1, padding='same', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='conv1d')(x)
    x = layers.BatchNormalization(name='batch_normalization')(x)
    x = layers.Activation('gelu', name='activation')(x)
    
    x = layers.Conv1D(cnn_filters, kernel_size=7, padding='same', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='conv1d_1')(x)
    x = layers.BatchNormalization(name='batch_normalization_1')(x)
    x = layers.Activation('gelu', name='activation_1')(x)
    x = ChannelAttention(reduction_ratio=4, name='channel_attention')(x)
    x = layers.Dropout(dropout, name='dropout')(x)
    
    x = layers.Conv1D(cnn_filters, kernel_size=5, padding='same', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='conv1d_2')(x)
    x = layers.BatchNormalization(name='batch_normalization_2')(x)
    x = layers.Activation('gelu', name='activation_2')(x)
    x = ChannelAttention(reduction_ratio=4, name='channel_attention_1')(x)
    x = layers.Dropout(dropout, name='dropout_1')(x)
    
    x = layers.Conv1D(cnn_filters, kernel_size=3, padding='same', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='conv1d_3')(x)
    x = layers.BatchNormalization(name='batch_normalization_3')(x)
    x = layers.Activation('gelu', name='activation_3')(x)
    x = layers.Dropout(dropout, name='dropout_2')(x)
    
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='max_pooling1d')(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2, name='max_pooling1d_1')(x)
    
    # Position embedding - add constant tensor (compatible with Keras 3)
    pos_embedding_const = tf.constant(pos_emb, dtype=tf.float32)
    x = x + pos_embedding_const
    x = layers.Dropout(dropout * 0.5, name='dropout_3')(x)
    
    # Transformer blocks
    embed_dim = cnn_filters
    
    # Block 1
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, value_dim=embed_dim // num_heads,
        dropout=dropout * 0.5, kernel_regularizer=regularizers.l2(weight_decay),
        name='multi_head_attention'
    )(x, x)
    x = layers.Add(name='add')([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6, name='layer_normalization')(x)
    
    ff = layers.Dense(ff_dim * 2, activation='gelu', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='dense')(x)
    ff = layers.Dropout(dropout, name='dropout_4')(ff)
    ff = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay), name='dense_1')(ff)
    ff = layers.Dropout(dropout, name='dropout_5')(ff)
    x = layers.Add(name='add_1')([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name='layer_normalization_1')(x)
    
    # Block 2
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=embed_dim // num_heads, value_dim=embed_dim // num_heads,
        dropout=dropout * 0.5, kernel_regularizer=regularizers.l2(weight_decay),
        name='multi_head_attention_1'
    )(x, x)
    x = layers.Add(name='add_2')([x, attn_output])
    x = layers.LayerNormalization(epsilon=1e-6, name='layer_normalization_2')(x)
    
    ff = layers.Dense(ff_dim * 2, activation='gelu', 
                      kernel_regularizer=regularizers.l2(weight_decay), name='dense_2')(x)
    ff = layers.Dropout(dropout, name='dropout_6')(ff)
    ff = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(weight_decay), name='dense_3')(ff)
    ff = layers.Dropout(dropout, name='dropout_7')(ff)
    x = layers.Add(name='add_3')([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6, name='layer_normalization_3')(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling1D(name='global_average_pooling1d')(x)
    x = layers.Dropout(dropout, name='dropout_8')(x)
    x = layers.Dense(128, activation='gelu', 
                     kernel_regularizer=regularizers.l2(weight_decay), name='dense_4')(x)
    x = layers.Dropout(dropout, name='dropout_9')(x)
    x = layers.Dense(64, activation='gelu', 
                     kernel_regularizer=regularizers.l2(weight_decay), name='dense_5')(x)
    x = layers.Dropout(dropout * 0.5, name='dropout_10')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', 
                          kernel_regularizer=regularizers.l2(weight_decay), name='dense_6')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='Advanced_MultiBand_CNN_Transformer')
    return model


@st.cache_resource
def load_norm_stats():
    """Load normalization statistics (cached)."""
    if not os.path.exists(NORM_STATS_PATH):
        return None, None
    norm_stats = np.load(NORM_STATS_PATH)
    return norm_stats['mean'], norm_stats['std']


def preprocess_eeg(df, mean, std, fs=128.0, epoch_length=128):
    """
    Preprocess EEG data from DataFrame with multi-band filtering.
    
    Returns:
        epochs: Preprocessed epochs (n_epochs, 128, 42) - 14 channels × 3 bands
        raw_data: Raw EEG data for visualization (14, n_samples)
    """
    # Extract EEG channels
    eeg_data = df[EEG_CHANNELS].values.T  # (14, n_samples)
    raw_data = eeg_data.copy()
    
    # Handle NaN values
    for i in range(eeg_data.shape[0]):
        mask = np.isnan(eeg_data[i])
        if np.any(mask):
            if np.all(mask):
                eeg_data[i] = 0
            else:
                eeg_data[i][mask] = np.interp(
                    np.where(mask)[0],
                    np.where(~mask)[0],
                    eeg_data[i][~mask]
                )
    
    # Multi-band filtering (Theta + Alpha + Beta)
    processed = extract_multiband_features(eeg_data, fs)  # (42, n_samples)
    
    # Segment into epochs
    n_channels, n_samples = processed.shape
    epochs = []
    
    start = 0
    while start + epoch_length <= n_samples:
        epoch = processed[:, start:start + epoch_length]
        # Remove DC offset
        epoch = epoch - epoch.mean(axis=1, keepdims=True)
        epochs.append(epoch)
        start += epoch_length
    
    if not epochs:
        return None, raw_data
    
    epochs = np.array(epochs, dtype=np.float32)  # (n_epochs, 42, 128)
    
    # Normalize
    epochs = (epochs - mean[None, :, None]) / std[None, :, None]
    epochs = np.clip(epochs, -5.0, 5.0)
    
    # Transpose: (n_epochs, 42, 128) → (n_epochs, 128, 42)
    epochs = np.transpose(epochs, (0, 2, 1))
    
    return epochs, raw_data


def predict_eeg(model, epochs):
    """Run prediction on preprocessed epochs."""
    predictions = model.predict(epochs, verbose=0)
    avg_prediction = predictions.mean(axis=0)
    return predictions, avg_prediction


# ============================================================================
# Visualization Functions
# ============================================================================
def plot_eeg_signals(raw_data, fs=128.0, n_seconds=5):
    """Plot EEG signals."""
    n_samples = min(int(n_seconds * fs), raw_data.shape[1])
    time = np.arange(n_samples) / fs
    
    fig = make_subplots(
        rows=7, cols=2,
        subplot_titles=[ch.replace('EEG.', '') for ch in EEG_CHANNELS],
        vertical_spacing=0.03,
        horizontal_spacing=0.05
    )
    
    colors = px.colors.qualitative.Set3
    
    for i, channel in enumerate(EEG_CHANNELS):
        row = (i % 7) + 1
        col = (i // 7) + 1
        
        fig.add_trace(
            go.Scatter(
                x=time,
                y=raw_data[i, :n_samples],
                mode='lines',
                name=channel,
                line=dict(color=colors[i % len(colors)], width=1),
                showlegend=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        title_text="EEG Signals (First 5 seconds)",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Time (s)", row=7, col=1)
    fig.update_xaxes(title_text="Time (s)", row=7, col=2)
    
    return fig


def plot_prediction_bars(avg_prediction):
    """Plot prediction probabilities as horizontal bar chart."""
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Arabic': ARABIC_CLASSES,
        'English': [ENGLISH_TRANSLATIONS[w] for w in ARABIC_CLASSES],
        'Probability': avg_prediction * 100
    })
    df = df.sort_values('Probability', ascending=True)
    
    # Create labels combining Arabic and English
    df['Label'] = df['Arabic'] + ' (' + df['English'] + ')'
    
    fig = go.Figure()
    
    # Color gradient based on probability
    colors = [f'rgba(102, 126, 234, {0.3 + 0.7 * p / 100})' for p in df['Probability']]
    
    fig.add_trace(go.Bar(
        y=df['Label'],
        x=df['Probability'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(102, 126, 234, 1)', width=1)
        ),
        text=[f'{p:.1f}%' for p in df['Probability']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Prediction Probabilities by Class",
        xaxis_title="Probability (%)",
        yaxis_title="Word",
        height=600,
        margin=dict(l=150),
        xaxis=dict(range=[0, max(df['Probability']) * 1.2])
    )
    
    return fig


def plot_epoch_predictions(predictions):
    """Plot per-epoch predictions as heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=predictions.T * 100,
        x=[f'Epoch {i+1}' for i in range(len(predictions))],
        y=[f'{w} ({ENGLISH_TRANSLATIONS[w]})' for w in ARABIC_CLASSES],
        colorscale='Viridis',
        colorbar=dict(title='Probability (%)')
    ))
    
    fig.update_layout(
        title="Per-Epoch Prediction Heatmap",
        xaxis_title="Epoch",
        yaxis_title="Word",
        height=500
    )
    
    return fig


def plot_consensus_pie(epoch_predictions):
    """Plot consensus voting as pie chart."""
    counter = Counter(epoch_predictions)
    
    labels = [f'{w} ({ENGLISH_TRANSLATIONS[w]})' for w in counter.keys()]
    values = list(counter.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set2)
    )])
    
    fig.update_layout(
        title="Consensus Voting Distribution",
        height=400
    )
    
    return fig


# ============================================================================
# Sidebar
# ============================================================================
def render_sidebar():
    """Render sidebar content."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/brain.png", width=80)
        st.title("🧠 ArEEG Classifier")
        
        st.markdown("---")
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This application uses a **CNN + Transformer** 
        deep learning model to classify **16 Arabic words** 
        from EEG brain signals.
        """)
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Info")
        st.markdown("""
        - **Architecture**: CNN + Transformer
        - **Channels**: 14 EEG channels
        - **Sampling Rate**: 128 Hz
        """)
        
        st.markdown("---")
        
        st.markdown("### 📝 Word Classes")
        
        word_df = pd.DataFrame({
            'Arabic': ARABIC_CLASSES,
            'English': [ENGLISH_TRANSLATIONS[w] for w in ARABIC_CLASSES]
        })
        st.dataframe(word_df, hide_index=True, use_container_width=True)


# ============================================================================
# Main Application
# ============================================================================
def main():
    """Main application entry point."""
    render_sidebar()
    
    # Header
    st.markdown('<p class="main-header">🧠 Arabic EEG Imagined Speech Classifier</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Decode imagined Arabic words from brain signals using Deep Learning</p>', unsafe_allow_html=True)
    
    # Load model and normalization stats
    model = load_model()
    mean, std = load_norm_stats()
    
    if model is None or mean is None:
        st.error("⚠️ Model files not found. Please ensure `outputs_advanced/best_model.keras` and `outputs_advanced/normalization_stats.npz` exist.")
        st.info("Run `python train_advanced_model.py` to train the model first.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload & Predict", "📈 Sample Analysis"])
    
    # ========================================================================
    # Tab 1: Upload & Predict
    # ========================================================================
    with tab1:
        st.markdown("### Upload EEG Recording")
        st.markdown("Upload a CSV file from the Emotiv EPOC X device to classify the imagined Arabic word.")
        
        uploaded_file = st.file_uploader(
            "Choose an EEG CSV file",
            type=['csv'],
            help="Upload a CSV file containing EEG data with the 14 channel columns"
        )
        
        if uploaded_file is not None:
            # Read the file
            try:
                df = pd.read_csv(uploaded_file, skiprows=1)
                
                # Validate columns
                missing_cols = [col for col in EEG_CHANNELS if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing EEG channels: {missing_cols}")
                    st.info("Required channels: " + ", ".join(EEG_CHANNELS))
                    return
                
                st.success(f"✅ File loaded: {uploaded_file.name}")
                st.info(f"📊 Data shape: {df.shape[0]} samples × {df.shape[1]} columns")
                
                # Preprocess and predict
                with st.spinner("🔄 Processing EEG data..."):
                    epochs, raw_data = preprocess_eeg(df, mean, std)
                
                if epochs is None:
                    st.error("❌ Could not extract valid epochs from the data.")
                    return
                
                st.success(f"✅ Extracted {len(epochs)} epochs")
                
                # Run prediction
                with st.spinner("🧠 Running prediction..."):
                    predictions, avg_prediction = predict_eeg(model, epochs)
                
                # Results
                st.markdown("---")
                st.markdown("## 🎯 Prediction Results")
                
                # Main prediction card
                top_idx = np.argmax(avg_prediction)
                top_word = ARABIC_CLASSES[top_idx]
                top_confidence = avg_prediction[top_idx] * 100
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <div style="font-size: 1rem; opacity: 0.8;">Predicted Word</div>
                        <div class="prediction-word">{top_word}</div>
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">({ENGLISH_TRANSLATIONS[top_word]})</div>
                        <div class="confidence-text">Confidence: {top_confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top 3 predictions
                st.markdown("### 🏆 Top 3 Predictions")
                top3_indices = np.argsort(avg_prediction)[::-1][:3]
                
                cols = st.columns(3)
                medals = ["🥇", "🥈", "🥉"]
                for i, idx in enumerate(top3_indices):
                    with cols[i]:
                        word = ARABIC_CLASSES[idx]
                        conf = avg_prediction[idx] * 100
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="font-size: 2rem;">{medals[i]}</div>
                            <div style="font-size: 1.5rem; font-weight: bold;">{word}</div>
                            <div style="color: #666;">{ENGLISH_TRANSLATIONS[word]}</div>
                            <div style="font-size: 1.2rem; color: #667eea;">{conf:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualizations
                st.markdown("---")
                st.markdown("### 📊 Detailed Analysis")
                
                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                    "Probabilities", "Epoch Heatmap", "Consensus", "EEG Signals"
                ])
                
                with viz_tab1:
                    fig_bars = plot_prediction_bars(avg_prediction)
                    st.plotly_chart(fig_bars, use_container_width=True)
                
                with viz_tab2:
                    fig_heatmap = plot_epoch_predictions(predictions)
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                
                with viz_tab3:
                    epoch_preds = [ARABIC_CLASSES[np.argmax(p)] for p in predictions]
                    fig_pie = plot_consensus_pie(epoch_preds)
                    st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # Consensus table
                    counter = Counter(epoch_preds)
                    consensus_df = pd.DataFrame([
                        {
                            'Word': f'{w} ({ENGLISH_TRANSLATIONS[w]})',
                            'Votes': c,
                            'Percentage': f'{c/len(epoch_preds)*100:.1f}%'
                        }
                        for w, c in counter.most_common()
                    ])
                    st.dataframe(consensus_df, hide_index=True, use_container_width=True)
                
                with viz_tab4:
                    fig_eeg = plot_eeg_signals(raw_data)
                    st.plotly_chart(fig_eeg, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.exception(e)
    
    # ========================================================================
    # Tab 2: Sample Analysis
    # ========================================================================
    with tab2:
        st.markdown("### 🔬 Analyze Sample Data")
        st.markdown("Test the model on one random sample from each word class.")
        
        # Find available samples
        data_dir = Path("data")
        if data_dir.exists():
            word_folders = [f for f in data_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
            
            if word_folders:
                if st.button("🔍 Analyze All Classes", type="primary"):
                    import random
                    
                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, word_folder in enumerate(word_folders):
                        word_name = word_folder.name
                        status_text.text(f"Analyzing: {word_name} ({ENGLISH_TRANSLATIONS.get(word_name, '')})")
                        
                        csv_files = list(word_folder.glob("*.csv"))
                        if csv_files:
                            # Pick a random sample
                            sample_file = random.choice(csv_files)
                            
                            try:
                                df = pd.read_csv(sample_file, skiprows=1)
                                epochs, raw_data = preprocess_eeg(df, mean, std)
                                
                                if epochs is not None:
                                    predictions, avg_prediction = predict_eeg(model, epochs)
                                    
                                    top_idx = np.argmax(avg_prediction)
                                    predicted_word = ARABIC_CLASSES[top_idx]
                                    confidence = avg_prediction[top_idx] * 100
                                    is_correct = predicted_word == word_name
                                    
                                    results.append({
                                        'expected_ar': word_name,
                                        'expected_en': ENGLISH_TRANSLATIONS.get(word_name, ''),
                                        'predicted_ar': predicted_word,
                                        'predicted_en': ENGLISH_TRANSLATIONS.get(predicted_word, ''),
                                        'confidence': confidence,
                                        'is_correct': is_correct,
                                        'file': sample_file.name
                                    })
                            except Exception as e:
                                results.append({
                                    'expected_ar': word_name,
                                    'expected_en': ENGLISH_TRANSLATIONS.get(word_name, ''),
                                    'predicted_ar': 'Error',
                                    'predicted_en': str(e)[:20],
                                    'confidence': 0,
                                    'is_correct': False,
                                    'file': sample_file.name if csv_files else 'N/A'
                                })
                        
                        progress_bar.progress((i + 1) / len(word_folders))
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    if results:
                        # Calculate accuracy
                        correct_count = sum(1 for r in results if r['is_correct'])
                        total_count = len(results)
                        accuracy = correct_count / total_count * 100
                        
                        # Summary metrics
                        st.markdown("---")
                        st.markdown("### 📊 Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Classes", total_count)
                        with col2:
                            st.metric("Correct", correct_count)
                        with col3:
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                        
                        # Results grid
                        st.markdown("---")
                        st.markdown("### 📋 Per-Class Results")
                        
                        # Display as cards in a grid
                        cols_per_row = 4
                        for row_start in range(0, len(results), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for col_idx, col in enumerate(cols):
                                result_idx = row_start + col_idx
                                if result_idx < len(results):
                                    r = results[result_idx]
                                    bg_color = '#d4edda' if r['is_correct'] else '#f8d7da'
                                    icon = '✅' if r['is_correct'] else '❌'
                                    
                                    with col:
                                        st.markdown(f"""
                                        <div style="background: {bg_color}; padding: 0.8rem; 
                                                    border-radius: 10px; text-align: center; margin-bottom: 1rem;">
                                            <div style="font-size: 1.5rem;">{icon}</div>
                                            <div style="font-size: 1.2rem; font-weight: bold; margin: 0.3rem 0;">
                                                {r['expected_ar']}
                                            </div>
                                            <div style="font-size: 0.8rem; color: #666;">
                                                {r['expected_en']}
                                            </div>
                                            <hr style="margin: 0.5rem 0; border-color: #ccc;">
                                            <div style="font-size: 0.75rem; color: #333;">
                                                Predicted: <b>{r['predicted_ar']}</b>
                                            </div>
                                            <div style="font-size: 0.85rem; color: #667eea; font-weight: bold;">
                                                {r['confidence']:.1f}%
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                        
                        # Detailed table
                        st.markdown("---")
                        st.markdown("### 📊 Detailed Table")
                        
                        results_df = pd.DataFrame([
                            {
                                'Expected (AR)': r['expected_ar'],
                                'Expected (EN)': r['expected_en'],
                                'Predicted (AR)': r['predicted_ar'],
                                'Predicted (EN)': r['predicted_en'],
                                'Confidence': f"{r['confidence']:.1f}%",
                                'Result': '✅' if r['is_correct'] else '❌'
                            }
                            for r in results
                        ])
                        st.dataframe(results_df, hide_index=True, use_container_width=True)
                        
                        # Confusion visualization
                        st.markdown("---")
                        st.markdown("### 🎯 Prediction Distribution")
                        
                        # Create a simple bar chart showing correct vs incorrect per class
                        fig = go.Figure()
                        
                        expected_words = [r['expected_ar'] for r in results]
                        confidences = [r['confidence'] for r in results]
                        colors = ['#28a745' if r['is_correct'] else '#dc3545' for r in results]
                        
                        fig.add_trace(go.Bar(
                            x=[f"{r['expected_ar']}\n({r['expected_en']})" for r in results],
                            y=confidences,
                            marker_color=colors,
                            text=[f"{c:.1f}%" for c in confidences],
                            textposition='outside'
                        ))
                        
                        fig.update_layout(
                            title="Prediction Confidence per Class (Green=Correct, Red=Incorrect)",
                            xaxis_title="Word Class",
                            yaxis_title="Confidence (%)",
                            height=500,
                            yaxis=dict(range=[0, 110])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🔍 Analyze Single Sample")
                st.markdown("Or select a specific file to analyze:")
                
                selected_word = st.selectbox(
                    "Select a word category:",
                    options=[f.name for f in word_folders],
                    format_func=lambda x: f"{x} ({ENGLISH_TRANSLATIONS.get(x, 'Unknown')})"
                )
                
                word_path = data_dir / selected_word
                csv_files = list(word_path.glob("*.csv"))
                
                if csv_files:
                    selected_file = st.selectbox(
                        "Select a sample file:",
                        options=csv_files,
                        format_func=lambda x: x.name[:50] + "..." if len(x.name) > 50 else x.name
                    )
                    
                    if st.button("🔍 Analyze This Sample"):
                        try:
                            df = pd.read_csv(selected_file, skiprows=1)
                            epochs, raw_data = preprocess_eeg(df, mean, std)
                            
                            if epochs is not None:
                                predictions, avg_prediction = predict_eeg(model, epochs)
                                
                                top_idx = np.argmax(avg_prediction)
                                predicted_word = ARABIC_CLASSES[top_idx]
                                confidence = avg_prediction[top_idx] * 100
                                
                                # Check if prediction matches folder
                                is_correct = predicted_word == selected_word
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown(f"""
                                    <div style="background: {'#d4edda' if is_correct else '#f8d7da'}; 
                                                padding: 1rem; border-radius: 10px; text-align: center;">
                                        <div style="font-size: 0.9rem; color: #666;">Expected</div>
                                        <div style="font-size: 2rem; font-weight: bold;">{selected_word}</div>
                                        <div>{ENGLISH_TRANSLATIONS.get(selected_word, '')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                with col2:
                                    st.markdown(f"""
                                    <div style="background: {'#d4edda' if is_correct else '#f8d7da'}; 
                                                padding: 1rem; border-radius: 10px; text-align: center;">
                                        <div style="font-size: 0.9rem; color: #666;">Predicted</div>
                                        <div style="font-size: 2rem; font-weight: bold;">{predicted_word}</div>
                                        <div>{ENGLISH_TRANSLATIONS.get(predicted_word, '')} ({confidence:.1f}%)</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                
                                if is_correct:
                                    st.success("✅ Correct prediction!")
                                else:
                                    st.warning("⚠️ Incorrect prediction")
                                
                                # Show probabilities
                                fig = plot_prediction_bars(avg_prediction)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.error("Could not process the sample file.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                else:
                    st.warning(f"No CSV files found in {word_path}")
            else:
                st.info("No word folders found in the data directory.")
        else:
            st.info("Data directory not found. Please ensure the dataset is in the `data/` folder.")


if __name__ == "__main__":
    main()
