# news-to-finance-dl-models
Predicting financial markets (BTC, Gold, VIX) from geopolitical news using hybrid deep learning models (BLSTM-Attention, GRU).

# Predicting Financial Markets from Geopolitical News

This project explores the correlation between global geopolitical events and financial market fluctuations (BTC, Gold, VIX) using hybrid deep learning models.

## 📊 Dataset
The dataset consists of 6 years of geopolitical news and financial data, queried and numerically transformed via **Google BigQuery**. 
* **Full Dataset:** https://www.kaggle.com/datasets/kozasalih/geopolitical-news-and-financial-markets-numeric-data

## 🤖 Models Implemented
I have developed and compared 5 different architectures to evaluate their predictive performance on complex financial time series:

1. **MLP (Multi-Layer Perceptron):** Baseline neural network approach.
2. **SVR (Support Vector Regression):** Traditional machine learning model.
3. **LSTM (Long Short-Term Memory):** Standard RNN for capturing temporal dependencies.
4. **Hybrid BLSTM-Attention:** A Bidirectional LSTM integrated with an **Attention Mechanism** to focus on the most impactful news events.
5. **Hybrid BLSTM-GRU:** A robust architecture combining Bidirectional LSTM with **Gated Recurrent Units** for efficient sequence modeling.

## 🚀 Key Features
- **Data Engineering:** Automated extraction and transformation of raw news data into numerical sentiment and impact features.
- **Hybrid Architectures:** Custom-built models focusing on market volatility and asset price directions.
- **Explainable AI (XAI):** Utilizing SHAP values to interpret model decisions and feature importance.

## 📈 Results and Visualizations
The models were evaluated using R², MAE, RMSE, and directional accuracy (MDA). Complete training histories, learning curves, and SHAP analyses are generated dynamically within the model scripts.
