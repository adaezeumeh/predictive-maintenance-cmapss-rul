# Artificial Intelligence in Predictive Maintenance for Industrial Equipment

**MSc Computer Science Dissertation — University of East London**  
**Author:** Adaeze Maryrose Umeh

## Overview
Unplanned equipment failures in capital-intensive industries lead to severe operational downtime and maintenance costs. This research presents an end-to-end machine learning and deep sequence modeling framework to predict **Remaining Useful Life (RUL)** and early fault onset using multivariate industrial sensor data (**NASA C-MAPSS FD001 & FD002**, 20,631 samples across 100 turbofan engines, evaluated over **217 experimental runs**).

---

## End-to-End Pipeline
1. **Signal Preprocessing & Denoising:**
   * Linear interpolation for missing sensor logs (0.21%).
   * **Isolation Forest** anomaly detection + winsorization (1st/99th percentiles) for sensor outliers (1.7%).
   * Min-Max normalization, improving average Signal-to-Noise Ratio (SNR) from **18.3 dB to 24.7 dB**.
2. **Feature Engineering & Selection:**
   * Extracted **142 candidate features** (temporal slopes, rate-of-change derivatives, FFT spectral peaks, rolling statistics, and domain-derived ratios such as *Fan Efficiency* and *Fuel-Air Ratio*).
   * Applied **Recursive Feature Elimination with Cross-Validation (RFE-CV)** to select an optimal subset of **22 features**—reducing dimensionality by **84.5%** while retaining **96.2%** of mutual information.
3. **Bayesian Hyperparameter Optimization:**
   * Conducted 50 Bayesian optimization iterations per model across **Random Forest (RF)**, **Artificial Neural Networks (ANN)**, and **Long Short-Term Memory (LSTM)** architectures.

---

## Key Benchmark Results (Held-Out Test Set)

### 1. Remaining Useful Life (RUL) Regression Performance
The optimized **LSTM** architecture (`[64, 32]` units, sequence length `30`, dropout `0.3`) achieved a **48.2% MAE reduction** over the linear baseline:

| Model | MAE (Cycles) | RMSE (Cycles) | $R^2$ Score | MAPE | Inference Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Linear Baseline | 28.64 | 34.72 | 0.721 | 35.6% | 2.1 ms |
| Random Forest (RF) | 19.42 | 25.31 | 0.861 | 23.1% | **8.3 ms** |
| Artificial Neural Network (ANN) | 17.15 | 23.07 | 0.884 | 19.8% | 12.1 ms |
| **LSTM (Optimized)** | **14.83** | **19.41** | **0.918** | **16.4%** | 15.7 ms |

### 2. Early Fault Detection (Binary Classification: Failure Within 30 Cycles)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Random Forest (RF) | 91.8% | 89.2% | 88.5% | 88.8% | 0.926 |
| Artificial Neural Network (ANN) | 94.6% | 93.1% | 92.7% | 92.9% | 0.961 |
| **LSTM (Optimized)** | **96.3%** | **95.4%** | **94.9%** | **95.1%** | **0.978** |

### 3. Prognostic Horizon ($\alpha$-$\lambda$ Accuracy)
* **LSTM** maintained **0.937 accuracy at λ = 125 cycles** before failure (average α-λ accuracy: **0.887** vs. **0.853** for ANN and **0.816** for RF).

---

## Visual Highlights
* `images/figure4_feature_importance.png` — Top engineered temporal & spectral features
* `images/figure5_training_curves.png` — Convergence & validation loss across RF, ANN, and LSTM
* `images/figure6_roc_curves.png` — Fault detection ROC comparison (LSTM AUC = 0.978)

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow / Keras, SciPy, Matplotlib, Seaborn

## Contact
* **LinkedIn:** https://www.linkedin.com/in/adaeze-u
* **Email:** lady.ada22@gmail.com

