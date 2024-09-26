# Emotion Detection Project

This project aims to build a system that detects emotions from audio files using machine learning techniques. The system processes audio data and classifies emotions based on predefined categories using neural networks and audio features.

## Features
- Audio data processing using MFCC, RMSE, and ZRC features.
- Visualizations such as Wave Signal, Spectrogram, and Chroma.
- Emotion classification using CNN model.
- Dataset support for CREMA-D, RAVDESS, SAVEE, and TESS.
- Custom metrics for model evaluation (Dice coefficient, Precision, Sensitivity, Specificity).

## Project Structure

- `models/`: Contains trained models and their weights (e.g., CNN_model.json, CNN_model_weights.h5).
- `audio_features/`: Scripts for extracting MFCC, RMSE, and ZRC features from audio.
- `visualizations/`: Visualization techniques (wave signals, spectrograms, etc.).
- `notebooks/`: Jupyter notebooks for model training and evaluation.
- `data/`: Directory for audio datasets (CREMA-D, RAVDESS, SAVEE, TESS).
- `results/`: Contains results of the model evaluations and metrics.
  
## Installation

To get started, clone the repository:

```bash
git clone https://github.com/abdellah20015/emotion_detection_pfe.git
cd emotion_detection_pfe
pip install Flask
python app.py
