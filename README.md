# 🫀 CardioCheck — Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A clinical-grade machine learning web application that predicts heart disease risk from patient vitals and diagnostic indicators. Built with Streamlit and a K-Nearest Neighbours classifier, CardioCheck delivers instant risk assessments through a clean, intuitive interface.

🌐 **Live Demo:** [cardiocheck-saurav.streamlit.app](https://cardiocheck-saurav.streamlit.app)

---

## 📌 Overview

Cardiovascular disease is the leading cause of death globally. CardioCheck leverages a trained KNN model to assess a patient's likelihood of heart disease based on 11 clinical features — providing a fast, accessible screening tool for educational and research purposes.

---

## 🖼️ Features

- **Instant Risk Assessment** — predicts High or Low risk in real time
- **Confidence Score** — displays model prediction confidence (%)
- **Clean UI** — inputs grouped into Demographics, Vitals, and Clinical Indicators
- **Human-readable Labels** — chest pain types and ECG readings shown with full descriptions
- **Medical Disclaimer** — clearly scoped as an educational tool
- **Mobile Responsive** — works across devices

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | K-Nearest Neighbours (KNN) |
| Preprocessing | StandardScaler |
| Input Features | 11 clinical features |
| Output | Binary — High Risk / Low Risk |
| Training Data | UCI Heart Failure dataset |

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

## 📦 Project Structure

```
cardio-check/
├── app.py                  # Main Streamlit application
├── knn_heart_model.pkl     # Serialised KNN classifier
├── heart_scaler.pkl        # Fitted StandardScaler
├── heart_columns.pkl       # Expected feature column order
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