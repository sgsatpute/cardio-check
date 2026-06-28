# 🫀 CardioCheck — Heart Disease Risk Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-98.4%25_Acc-brightgreen?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)

**A production-ready machine learning web application for real-time heart disease risk assessment.**

[🌐 Live Demo](https://cardiocheck-saurav.streamlit.app) · [📖 Project Explanation](PROJECT_EXPLANATION.md) · [🐛 Report Bug](https://github.com/sgsatpute/cardio-check/issues)

</div>

---

## 📌 Overview

Cardiovascular disease is the **leading cause of death globally**, accounting for over 17.9 million lives per year (WHO). Early detection is critical — yet access to quick, data-driven screening tools remains limited.

**CardioCheck** addresses this by providing an intelligent, accessible risk assessment tool built on two machine learning classifiers trained on 918 real patient records. Users input 11 clinical parameters and receive an instant risk prediction with probability confidence, powered by XGBoost (98.4% accuracy) and KNN (85.9% accuracy).

> ⚕️ This tool is for educational and research purposes only. It does not substitute clinical diagnosis.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **Dual ML Models** | Switch between XGBoost and KNN in real time |
| 📊 **Probability Gauge** | Interactive speedometer showing exact risk % |
| 🎯 **Confidence Score** | Model certainty displayed alongside each prediction |
| 📈 **EDA Dashboard** | 4 interactive Plotly charts for data exploration |
| 🏆 **Model Performance** | Accuracy, F1 score, confusion matrix comparison |
| 🎨 **Clean UI** | Inputs grouped into logical clinical sections |
| 📱 **Responsive** | Works seamlessly on desktop and mobile |

---

## 🧠 Machine Learning

### Model Comparison

| Metric | XGBoost | KNN |
|---|---|---|
| **Accuracy** | 98.4% | 85.9% |
| **F1 Score** | 98.7% | 89.2% |
| **Preprocessing** | None (tree-based) | StandardScaler |
| **Key Hyperparameters** | n_estimators=100, max_depth=4, lr=0.1 | n_neighbors=11 |
| **Inference Speed** | Fast | Fast |

### Input Features

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | Age | Numerical | Patient age in years (18–100) |
| 2 | Biological Sex | Categorical | Male / Female |
| 3 | Chest Pain Type | Categorical | ATA, NAP, TA, ASY |
| 4 | Resting Blood Pressure | Numerical | mm Hg (80–200) |
| 5 | Cholesterol | Numerical | mg/dL (100–600) |
| 6 | Fasting Blood Sugar | Binary | > 120 mg/dL: Yes / No |
| 7 | Resting ECG | Categorical | Normal, ST abnormality, LVH |
| 8 | Max Heart Rate | Numerical | Beats per minute (60–220) |
| 9 | Exercise-Induced Angina | Binary | Yes / No |
| 10 | Oldpeak | Numerical | ST depression value (0.0–6.2) |
| 11 | ST Slope | Categorical | Up / Flat / Down |

### Pipeline

```
Raw Input → One-Hot Encoding → StandardScaler (KNN only) → Model → Probability → Risk Label
```

---

## 🖥️ App Structure

```
🔍 Risk Assessment       →  Input form + model selector + result banner + gauge
📊 Model Performance     →  Metrics cards + bar chart + confusion matrix
📈 EDA                   →  Age dist + chest pain breakdown + scatter + heatmap
```

---

## 📦 Project Structure

```
cardio-check/
├── app.py                    # Main Streamlit application (3 tabs)
├── knn_heart_model.pkl       # Trained KNN classifier (n_neighbors=11)
├── xgb_heart_model.pkl       # Trained XGBoost classifier
├── heart_scaler.pkl          # Fitted StandardScaler for KNN pipeline
├── heart_columns.pkl         # One-hot encoded feature column order
├── model_metrics.json        # Accuracy, F1 scores, confusion matrices
├── requirements.txt          # Python dependencies
├── PROJECT_EXPLANATION.md    # Full technical write-up
├── .streamlit/
│   └── config.toml           # Theme (red palette) and server config
└── README.md
```

---

## 🚀 Quick Start

### Run Locally

**Prerequisites:** Python 3.10+, Git

```bash
# Clone the repo
git clone https://github.com/sgsatpute/cardio-check.git
cd cardio-check

# Install dependencies
pip install -r requirements.txt

# Launch
streamlit run app.py
```

Opens at → `http://localhost:8501`

### Deploy to Streamlit Cloud (Free)

1. Fork this repository
2. Visit [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **"Create app"** → fill in:
   - **Repository:** `your-username/cardio-check`
   - **Branch:** `main`
   - **Main file:** `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Custom CSS |
| ML Models | XGBoost, scikit-learn (KNN) |
| Data Processing | Pandas, NumPy |
| Serialisation | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 📄 Documentation

For a full technical explanation of the project — including problem statement, data preprocessing steps, model training rationale, feature engineering, and results analysis — see **[PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md)**.

---

## ⚕️ Disclaimer

This application is intended for **educational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. Predictions are probabilistic and based on a statistical model — not a clinical evaluation. Always consult a qualified healthcare professional for any medical concerns.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  Made with ❤️ | <a href="https://cardiocheck-saurav.streamlit.app">cardiocheck-saurav.streamlit.app</a>
</div>