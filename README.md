# 🛡️ FraudShield: Real-Time Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-green.svg)

An end-to-end Machine Learning pipeline designed to detect fraudulent credit card transactions in real-time. This project handles extreme data imbalance (0.17% fraud rate) and provides a live monitoring console for transaction security.

## 📊 Project Overview
Credit card fraud costs the global economy billions annually. This project uses a **LightGBM** gradient-boosted decision tree model to identify the subtle mathematical fingerprints left by fraudsters.

### Key Features:
* **Data Resampling:** Addressed heavy class imbalance using cost-sensitive learning (`scale_pos_weight`).
* **Real-Time Inference:** A live terminal-based monitor that processes transactions and flags risks instantly.
* **Visual Insights:** Comprehensive Exploratory Data Analysis (EDA) showcasing the correlation between hidden features and fraudulent behavior.

## 📁 Project Structure
* `data/`: Contains the Credit Card Fraud detection dataset (PCA-transformed).
* `models/`: Stores the trained `.pkl` model file.
* `notebooks/`: Jupyter Notebooks containing visual data analysis and plots.
* `src/`: Production-ready Python scripts for training and real-time monitoring.

## 🚀 How It Works

### 1. Training
The model was trained on 284,807 transactions. By adjusting the class weights, the model achieved a high **Recall (~81%)**, ensuring that most fraudulent "test charges" are caught before they scale.

### 2. Live Monitoring
The system simulates a live bank feed. As transactions arrive, the AI calculates a probability score.
* **Green:** Secure transaction (Risk < 50%)
* **Red:** ALERT! Potential fraud detected (Risk > 50%)

## 🛠️ Setup & Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/Shad1573/Fraud-Detection-Realtime.git](https://github.com/Shad1573/Fraud-Detection-Realtime.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Live Monitor:
    ```bash
    python src/3_app.py
    ```

## 📈 Visualizations
You can view the detailed data analysis in the [Exploration Notebook](./notebooks/Exploration.ipynb), featuring:
* Correlation Heatmaps
* Transaction Amount Distributions
* Feature Anomaly Detection (Boxplots)

---
*Created as a 3-day AI Development Intensive.*
