"""
Export model weights for Keras 3 compatibility.
Run this locally where TensorFlow 2.x with Keras 2 is installed.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")

# Custom layers (same as in the model)
class EEGAugmenter(layers.Layer):
    def __init__(self, noise_std=0.01, time_shift_max=5, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
    
    def get_config(self):
        config = super().get_config()
        config.update({'noise_std': self.noise_std, 'time_shift_max': self.time_shift_max})
        return config
    
    def call(self, x, training=None):
        return x if not training else x


class AdvancedEEGAugmenter(layers.Layer):
    def __init__(self, noise_std=0.02, time_shift_max=8, channel_dropout=0.1, **kwargs):
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
        return x if not training else x


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=4, **kwargs):
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
        squeezed = self.global_pool(x)
        excited = self.fc1(squeezed)
        excited = self.fc2(excited)
        return x * tf.expand_dims(excited, axis=1)


# Load the model
MODEL_PATH = "outputs_advanced/best_model.keras"

custom_objects = {
    'EEGAugmenter': EEGAugmenter,
    'AdvancedEEGAugmenter': AdvancedEEGAugmenter,
    'ChannelAttention': ChannelAttention,
    'Custom>EEGAugmenter': EEGAugmenter,
    'Custom>AdvancedEEGAugmenter': AdvancedEEGAugmenter,
    'Custom>ChannelAttention': ChannelAttention,
}

print(f"Loading model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
print("Model loaded successfully!")
print(f"Model name: {model.name}")

# Extract and save weights
weights_dict = {}
for layer in model.layers:
    layer_weights = layer.get_weights()
    if layer_weights:
        weights_dict[layer.name] = layer_weights
        print(f"Layer: {layer.name}, #weights: {len(layer_weights)}, shapes: {[w.shape for w in layer_weights]}")

# Save all weights to a single file
output_path = "outputs_advanced/model_weights.npz"
np.savez(output_path, **{f"{name}_{i}": w for name, weights in weights_dict.items() for i, w in enumerate(weights)})
print(f"\nWeights saved to {output_path}")

# Also save the weights using Keras built-in method (weights only, no architecture)
weights_h5_path = "outputs_advanced/model_weights.weights.h5"
model.save_weights(weights_h5_path)
print(f"Weights also saved to {weights_h5_path}")

print("\n=== EXPORT COMPLETE ===")
print("Now you can use these weights with the Keras 3 compatible architecture.")
