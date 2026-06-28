import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck — Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 60%, #e8573f 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem 1.6rem;
        margin-bottom: 1.8rem;
        color: white;
    }
    .hero h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem; }
    .hero p  { font-size: 0.95rem; opacity: 0.88; margin: 0; }

    /* Section card */
    .section-card {
        background: #fafafa;
        border: 1px solid #ebebeb;
        border-radius: 12px;
        padding: 1.2rem 1.4rem 0.6rem;
        margin-bottom: 1.2rem;
    }
    .section-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.8rem;
    }

    /* Result banner */
    .result-high {
        background: #fff0f0;
        border: 2px solid #e74c3c;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-low {
        background: #f0fff4;
        border: 2px solid #27ae60;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        margin-top: 1rem;
    }
    .result-high h2 { color: #c0392b; font-size: 1.3rem; margin: 0 0 0.4rem; }
    .result-low  h2 { color: #1e8449;  font-size: 1.3rem; margin: 0 0 0.4rem; }
    .result-high p, .result-low p { font-size: 0.88rem; color: #555; margin: 0; }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.88; }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.76rem;
        color: #999;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model           = joblib.load("knn_heart_model.pkl")
    scaler          = joblib.load("heart_scaler.pkl")
    expected_cols   = joblib.load("heart_columns.pkl")
    return model, scaler, expected_cols

model, scaler, expected_columns = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫀 CardioCheck</h1>
  <p>Fill in your clinical details below to get an instant heart disease risk assessment powered by a K-Nearest Neighbours classifier.</p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────

# — Demographics —
st.markdown('<div class="section-card"><div class="section-label">Demographics</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 45)
with col2:
    sex = st.selectbox("Biological Sex", ["Male", "Female"])
st.markdown('</div>', unsafe_allow_html=True)

# — Vitals —
st.markdown('<div class="section-card"><div class="section-label">Vitals</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    resting_bp   = st.number_input("Resting BP (mm Hg)",  min_value=80,  max_value=200, value=120)
with col2:
    cholesterol  = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
with col3:
    max_hr       = st.slider("Max Heart Rate", 60, 220, 150)
st.markdown('</div>', unsafe_allow_html=True)

# — Clinical Indicators —
st.markdown('<div class="section-card"><div class="section-label">Clinical Indicators</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    chest_pain      = st.selectbox("Chest Pain Type", ["ATA — Atypical Angina", "NAP — Non-Anginal Pain", "TA — Typical Angina", "ASY — Asymptomatic"])
    resting_ecg     = st.selectbox("Resting ECG", ["Normal", "ST — ST-T wave abnormality", "LVH — Left Ventricular Hypertrophy"])
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
with col2:
    fasting_bs  = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    st_slope    = st.selectbox("ST Slope", ["Up", "Flat", "Down"])
    oldpeak     = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
st.markdown('</div>', unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("Assess My Risk"):

    # Map display labels back to model tokens
    sex_token           = "M" if sex == "Male" else "F"
    cp_token            = chest_pain.split(" — ")[0]
    ecg_token           = resting_ecg.split(" — ")[0]
    angina_token        = "Y" if exercise_angina == "Yes" else "N"
    fasting_token       = 1   if fasting_bs == "Yes" else 0

    raw_input = {
        "Age":        age,
        "RestingBP":  resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS":  fasting_token,
        "MaxHR":      max_hr,
        "Oldpeak":    oldpeak,
        f"Sex_{sex_token}": 1,
        f"ChestPainType_{cp_token}": 1,
        f"RestingECG_{ecg_token}": 1,
        f"ExerciseAngina_{angina_token}": 1,
        f"ST_Slope_{st_slope}": 1,
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df     = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    proba        = model.predict_proba(scaled_input)[0]
    confidence   = round(proba[int(prediction)] * 100, 1)

    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
          <h2>⚠️ Elevated Risk Detected</h2>
          <p>The model indicates a <strong>high likelihood of heart disease</strong> based on the values you provided (model confidence: {confidence}%).<br>
          Please consult a cardiologist or healthcare professional for a full clinical evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
          <h2>✅ Lower Risk Indicated</h2>
          <p>The model suggests a <strong>lower risk of heart disease</strong> based on the values you provided (model confidence: {confidence}%).<br>
          Continue maintaining a heart-healthy lifestyle and schedule regular check-ups.</p>
        </div>
        """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚕️ This tool is for educational purposes only and does not constitute medical advice.<br>
  Always consult a qualified healthcare professional for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
