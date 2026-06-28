# 🫀 CardioCheck — Heart Disease Risk Predictor

A machine learning web app that predicts heart disease risk using a K-Nearest Neighbours (KNN) classifier trained on clinical data.

## 🚀 Deploy on Streamlit Community Cloud (Free)

1. Push this project to a **public GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**
4. Set:
   - **Repository**: your GitHub repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **Deploy** — your app will be live in ~2 minutes at `yourapp.streamlit.app`

## 🖥️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Project Structure

```
├── app.py                  # Streamlit app
├── knn_heart_model.pkl     # Trained KNN model
├── heart_scaler.pkl        # StandardScaler
├── heart_columns.pkl       # Expected feature columns
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── config.toml         # Theme & server config
```

## 🔍 Features

- Age, sex, chest pain type, resting BP, cholesterol
- Fasting blood sugar, resting ECG, max heart rate
- Exercise-induced angina, ST depression (oldpeak), ST slope
- Shows prediction confidence alongside High / Low risk result
- Medical disclaimer footer
