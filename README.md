# 📊 AI Portfolio Analysis & Stock Risk Prediction System

## 📌 Project Overview

This project is an intelligent **Portfolio Analysis and Stock Prediction System** that helps users make informed investment decisions using machine learning and AI.

The system collects real-time stock data from Yahoo Finance, performs data preprocessing and analysis, predicts stock trends for the next 10 days, and classifies investment risk levels. It also includes a **RAG-based chatbot** that allows users to interact with the system and gain insights from predictions.

---

## 🚀 Key Features

- 📥 Fetch real-time stock data from Yahoo Finance
- 🧹 Data preprocessing and feature engineering
- 📊 Exploratory data analysis (EDA)
- 🤖 Stock price prediction for the next 10 days
- ⚠️ Risk classification: **Low / Medium / High**
- 🏆 Top 10 company recommendations (from 50 companies)
- 💬 AI Chatbot with RAG (Retrieval-Augmented Generation)
- 📈 Portfolio decision support system

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Data Source:** Yahoo Finance API (`yfinance`)  
- **Libraries:**  
  - Pandas, NumPy  
  - Scikit-learn  
  - Random Forest  
  - LightGBM  
  - Matplotlib, Seaborn  
- **AI / NLP:** RAG (Retrieval-Augmented Generation)  
- **Environment:** Jupyter Notebook / Python IDE  

## ⚙️ Workflow
1 Data Collection
Fetch stock data using Yahoo Finance API.
2 Data Preprocessing
Clean data, handle missing values, and engineer features.
3 EDA (Exploratory Data Analysis)
Analyze trends, correlations, and stock behavior.
4 Model Training
Train Random Forest and LightGBM models.
5 Prediction
Predict stock prices for the next 10 days.
6 Risk Classification
Categorize stocks into risk levels.
7 Top 10 Selection
Identify best companies for investment.
8 RAG Chatbot
Enable interactive querying using prediction results.



## 💬 RAG-Based Chatbot

The system includes an intelligent chatbot powered by **Retrieval-Augmented Generation (RAG)**, designed to make portfolio insights easier to understand through natural language interaction.

### Features

Users can ask questions such as:

- **Which stock is safest to invest in?**
- **Why is this company classified as high risk?**
- **What are the top 10 companies for investment?**

### How It Works

The chatbot:

- Retrieves stock prediction results and risk classification outputs
- Uses those results as context for answering user queries
- Generates accurate, relevant, and easy-to-understand responses

### Purpose

This feature helps users:

- Understand prediction results more clearly
- Compare companies based on risk and performance
- Make more informed investment decisions through conversational AI


## ⚠️ Risk Classification

Each stock is classified into different risk levels based on multiple factors derived from data analysis and model predictions.

### Criteria Used

- **Volatility** → How much the stock price fluctuates over time  
- **Price Fluctuations** → Short-term instability in stock movement  
- **Model Prediction Uncertainty** → Confidence level of the ML model predictions  

### Risk Levels

- 🟢 **Low Risk**  
  Stable stocks with minimal fluctuations and high prediction confidence  

- 🟡 **Medium Risk**  
  Moderate fluctuations with some uncertainty in predictions  

- 🔴 **High Risk**  
  Highly volatile stocks with unpredictable behavior and lower model confidence  



## 📂 Project Structure

```bash
investment-portfolio-analysis/
│
├── backend/
│   ├── models/
│   │   ├── train_model.py
│   │   ├── train_model1.ipynb
│   │
│   ├── preprocess_data/
│   │   # Data cleaning and transformation logic
│   │
│   ├── rag_agent/
│   │   # RAG chatbot logic and retrieval system
│   │
│   ├── download_csv.py
│   │   # Fetch stock data from Yahoo Finance
│   │
│   ├── features.py
│   │   # Feature engineering for ML models
│   │
│   ├── preprocess.py
│   │   # Data preprocessing pipeline
│   │
│   ├── risk.py
│   │   # Risk classification logic (Low/Medium/High)
│
├── data/
│   # Raw stock data
│
├── processed/
│   # Cleaned and transformed data
│
├── processed-rag/
│   # Data used for RAG system
│
├── scripts/
│   # Utility scripts
│
├── frontend/
│   # Frontend UI (React or other)
│
├── .env
├── requirements.txt
├── README.md