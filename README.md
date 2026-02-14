---
title: Arabic EEG Imagined Speech Classifier
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Arabic EEG Imagined Speech Classifier

CNN + Transformer model for classifying imagined Arabic speech from EEG signals.

## Features
- 16 Arabic words classification
- Multi-band EEG feature extraction (Theta, Alpha, Beta)
- Deep learning with CNN + Transformer architecture
- Real-time visualization with Plotly

## Model
- Input: 128 time samples × 14 EEG channels
- Output: 16 Arabic word classes
- Architecture: CNN with Channel Attention + Transformer

## Usage
Upload an EEG CSV file to classify the imagined Arabic word.
