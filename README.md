# 🫀 CardioCheck — Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-98.4%25_Accuracy-brightgreen?logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A clinical-grade machine learning web application that predicts heart disease risk from patient vitals and diagnostic indicators. Built with Streamlit, the app features two trained classifiers (XGBoost and KNN), interactive EDA visualisations, and a real-time probability gauge.

🌐 **Live Demo:** [cardiocheck-saurav.streamlit.app](https://cardiocheck-saurav.streamlit.app)

---

## 📌 Overview

Cardiovascular disease is the leading cause of death globally. CardioCheck leverages two ML models trained on 918 patient records to assess heart disease risk based on 11 clinical features — providing a fast, accessible screening tool for educational and research purposes.

---

## 🖼️ Features

- **Dual Model Support** — switch between XGBoost (98.4% accuracy) and KNN (85.9% accuracy)
- **Instant Risk Assessment** — predicts High or Low risk in real time
- **Probability Gauge** — live speedometer showing exact risk percentage
- **Model Performance Tab** — accuracy, F1 score, confusion matrix, and model comparison chart
- **EDA Tab** — 4 interactive charts exploring feature distributions and correlations
- **Clean UI** — inputs grouped into Demographics, Vitals, and Clinical Indicators
- **Human-readable Labels** — chest pain types and ECG readings shown with full descriptions
- **Medical Disclaimer** — clearly scoped as an educational tool

---

## 🧠 Model Details

| Property | XGBoost | KNN |
|---|---|---|
| Accuracy | 98.4% | 85.9% |
| F1 Score | 98.7% | 89.2% |
| Preprocessing | None (tree-based) | StandardScaler |
| Hyperparameters | n_estimators=100, max_depth=4, lr=0.1 | n_neighbors=11 |

### Input Features

| Feature | Description |
|---|---|
| Age | Patient age in years |
| Biological Sex | Male / Female |
| Chest Pain Type | ATA, NAP, TA, ASY |
| Resting Blood Pressure | mm Hg |
| Cholesterol | mg/dL |
| Fasting Blood Sugar | > 120 mg/dL (Yes/No) |
| Resting ECG | Normal, ST abnormality, LVH |
| Max Heart Rate | Beats per minute |
| Exercise-Induced Angina | Yes / No |
| Oldpeak | ST depression value |
| ST Slope | Up / Flat / Down |

---

## 📊 App Tabs

| Tab | Contents |
|---|---|
| 🔍 Risk Assessment | Input form, model selector, result banner, probability gauge |
| 📊 Model Performance | Accuracy/F1 metrics, bar chart comparison, confusion matrix |
| 📈 EDA | Age distribution, chest pain breakdown, MaxHR vs Age scatter, correlation heatmap |

---

## 📦 Project Structure

```
cardio-check/
├── app.py                  # Main Streamlit application
├── knn_heart_model.pkl     # Trained KNN classifier
├── xgb_heart_model.pkl     # Trained XGBoost classifier
├── heart_scaler.pkl        # Fitted StandardScaler (for KNN)
├── heart_columns.pkl       # Expected feature column order
├── model_metrics.json      # Accuracy, F1, confusion matrices
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Theme and server configuration
└── README.md
```

---

## 🚀 Run Locally

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/sgsatpute/cardio-check.git
cd cardio-check

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## ☁️ Deploy on Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"Create app"** and fill in:
   - **Repository:** `your-username/cardio-check`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy** — live in ~2 minutes

---

## ⚕️ Disclaimer

This application is intended for **educational and research purposes only**. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for any medical concerns.

---

## 📄 License

This project is licensed under the MIT License.