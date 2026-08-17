# 👁️ Drowsiness & Eye State Detection CNN Model

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-97.2%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

Lightweight, high-precision Convolutional Neural Network (CNN) architecture designed to detect eye states (Open vs. Closed) for Driver Drowsiness Detection Systems. Built with **TensorFlow 2.20** and trained on the **MRL Eye Dataset**.

---

## 📌 Key Highlights & Results

- **Validation Accuracy**: `97.20%`
- **Validation Loss**: `0.0764`
- **Precision / Recall**: `97.20% / 97.20%`
- **Model Size**: `101,665 parameters (~397 KB)` — *Optimized for real-time edge deployment!*

---

## 🏗️ Architecture Overview

The model employs a custom lightweight 2D-CNN pipeline using **Global Average Pooling (GAP)** instead of heavy fully connected layers to prevent overfitting and guarantee real-time speed:

1. **Input Layer**: `64x64` Grayscale images.
2. **Data Augmentation**: Integrated `RandomRotation(0.08)` and `RandomZoom(0.08)` layers.
3. **Feature Extraction Blocks (x3)**:
   - `Conv2D(32/64/128, 3x3, padding='same')` + `BatchNormalization` + `ReLU` + `MaxPooling2D(2x2)`
4. **Classification Head**:
   - `GlobalAveragePooling2D`
   - `Dense(64, ReLU)` + `Dropout(0.3)`
   - `Dense(1, Sigmoid)` for binary classification.

---

## 📊 Dataset Information

- **Dataset Source**: [MRL Eye Open/Close Dataset](https://www.kaggle.com/datasets/rameezakther/mrl-eye-open-or-close-dataset) (via `kagglehub`).
- **Train Set**: 20,000 images (Balanced Open/Closed).
- **Test Set**: 1,000 images.
- **Pipeline**: Automated in-memory caching and prefetching via `tf.data.AUTOTUNE`.

---

## 🚀 Training Strategy & Optimization

- **Optimizer**: Adam (`initial_lr = 1e-3`)
- **Loss Function**: `BinaryCrossentropy`
- **Callbacks**:
  - `ModelCheckpoint`: Saved best `.keras` model based on `val_loss`.
  - `ReduceLROnPlateau`: Reduced learning rate dynamically upon plateau (`factor=0.3`).
  - `EarlyStopping`: Restored best weights automatically.

---

## 📂 Project Structure

```text
├── eye_state_classifier.ipynb    # Training notebook with visualizations
├── saved_models/
│   └── best_eye_model.keras      # Saved trained TensorFlow model
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
