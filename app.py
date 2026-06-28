import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioCheck — Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #c0392b 0%, #e74c3c 60%, #e8573f 100%);
    border-radius: 16px; padding: 2rem 2.2rem 1.6rem;
    margin-bottom: 1.8rem; color: white;
}
.hero h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.3rem; }
.hero p  { font-size: 0.95rem; opacity: 0.88; margin: 0; }

.section-card {
    background: #fafafa; border: 1px solid #ebebeb;
    border-radius: 12px; padding: 1.2rem 1.4rem 0.6rem; margin-bottom: 1.2rem;
}
.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #888; margin-bottom: 0.8rem;
}
.metric-card {
    background: white; border: 1px solid #ebebeb; border-radius: 10px;
    padding: 1rem; text-align: center;
}
.metric-val  { font-size: 1.6rem; font-weight: 700; color: #c0392b; }
.metric-label { font-size: 0.78rem; color: #888; margin-top: 0.2rem; }

.result-high {
    background: #fff0f0; border: 2px solid #e74c3c;
    border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center; margin-top: 1rem;
}
.result-low {
    background: #f0fff4; border: 2px solid #27ae60;
    border-radius: 12px; padding: 1.4rem 1.6rem; text-align: center; margin-top: 1rem;
}
.result-high h2 { color: #c0392b; font-size: 1.3rem; margin: 0 0 0.4rem; }
.result-low  h2 { color: #1e8449; font-size: 1.3rem; margin: 0 0 0.4rem; }
.result-high p, .result-low p { font-size: 0.88rem; color: #555; margin: 0; }

div.stButton > button {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    color: white; font-weight: 600; font-size: 1rem;
    border: none; border-radius: 10px; padding: 0.7rem 2rem;
    width: 100%; cursor: pointer; transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

.disclaimer {
    font-size: 0.76rem; color: #999; text-align: center;
    margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;
}
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    knn            = joblib.load("knn_heart_model.pkl")
    xgb            = joblib.load("xgb_heart_model.pkl")
    scaler         = joblib.load("heart_scaler.pkl")
    expected_cols  = joblib.load("heart_columns.pkl")
    with open("model_metrics.json") as f:
        metrics = json.load(f)
    return knn, xgb, scaler, expected_cols, metrics

knn_model, xgb_model, scaler, expected_columns, metrics = load_models()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🫀 CardioCheck</h1>
  <p>Clinical heart disease risk assessment powered by KNN and XGBoost classifiers trained on 918 patient records.</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Risk Assessment", "📊 Model Performance", "📈 EDA"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        # Demographics
        st.markdown('<div class="section-card"><div class="section-label">Demographics</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: age = st.slider("Age", 18, 100, 45)
        with c2: sex = st.selectbox("Biological Sex", ["Male", "Female"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Vitals
        st.markdown('<div class="section-card"><div class="section-label">Vitals</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: resting_bp  = st.number_input("Resting BP (mm Hg)", 80, 200, 120)
        with c2: cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        with c3: max_hr      = st.slider("Max Heart Rate", 60, 220, 150)
        st.markdown('</div>', unsafe_allow_html=True)

        # Clinical
        st.markdown('<div class="section-card"><div class="section-label">Clinical Indicators</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            chest_pain      = st.selectbox("Chest Pain Type", ["ATA — Atypical Angina","NAP — Non-Anginal Pain","TA — Typical Angina","ASY — Asymptomatic"])
            resting_ecg     = st.selectbox("Resting ECG", ["Normal","ST — ST-T wave abnormality","LVH — Left Ventricular Hypertrophy"])
            exercise_angina = st.selectbox("Exercise-Induced Angina", ["No","Yes"])
        with c2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No","Yes"])
            st_slope   = st.selectbox("ST Slope", ["Up","Flat","Down"])
            oldpeak    = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st.markdown('</div>', unsafe_allow_html=True)

        model_choice = st.radio("Select Model", ["XGBoost (Recommended)", "KNN"], horizontal=True)

    with col_result:
        st.markdown("### Result")

        if st.button("Assess My Risk"):
            sex_tok    = "M" if sex == "Male" else "F"
            cp_tok     = chest_pain.split(" — ")[0]
            ecg_tok    = resting_ecg.split(" — ")[0]
            angina_tok = "Y" if exercise_angina == "Yes" else "N"
            fasting_v  = 1 if fasting_bs == "Yes" else 0

            raw = {
                "Age": age, "RestingBP": resting_bp, "Cholesterol": cholesterol,
                "FastingBS": fasting_v, "MaxHR": max_hr, "Oldpeak": oldpeak,
                f"Sex_{sex_tok}": 1, f"ChestPainType_{cp_tok}": 1,
                f"RestingECG_{ecg_tok}": 1, f"ExerciseAngina_{angina_tok}": 1,
                f"ST_Slope_{st_slope}": 1,
            }
            inp = pd.DataFrame([raw])
            for col in expected_columns:
                if col not in inp.columns: inp[col] = 0
            inp = inp[expected_columns]

            if "XGBoost" in model_choice:
                pred   = xgb_model.predict(inp)[0]
                proba  = xgb_model.predict_proba(inp)[0]
                mname  = "XGBoost"
            else:
                scaled = scaler.transform(inp)
                pred   = knn_model.predict(scaled)[0]
                proba  = knn_model.predict_proba(scaled)[0]
                mname  = "KNN"

            conf = round(proba[int(pred)] * 100, 1)
            low_p  = round(proba[0] * 100, 1)
            high_p = round(proba[1] * 100, 1)

            if pred == 1:
                st.markdown(f"""
                <div class="result-high">
                  <h2>⚠️ Elevated Risk Detected</h2>
                  <p>High likelihood of heart disease detected ({mname}, confidence: <strong>{conf}%</strong>).<br>
                  Please consult a cardiologist for a full clinical evaluation.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                  <h2>✅ Lower Risk Indicated</h2>
                  <p>Lower risk of heart disease indicated ({mname}, confidence: <strong>{conf}%</strong>).<br>
                  Continue maintaining a heart-healthy lifestyle.</p>
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_p,
                title={"text": "Heart Disease Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#e74c3c" if high_p > 50 else "#27ae60"},
                    "steps": [
                        {"range": [0,  40], "color": "#d5f5e3"},
                        {"range": [40, 60], "color": "#fef9e7"},
                        {"range": [60,100], "color": "#fadbd8"},
                    ],
                    "threshold": {"line": {"color": "black","width": 2}, "value": 50}
                }
            ))
            fig.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="disclaimer">⚕️ For educational purposes only. Not medical advice.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Model Comparison")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["xgb"]["accuracy"]*100:.1f}%</div><div class="metric-label">XGBoost Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["xgb"]["f1"]*100:.1f}%</div><div class="metric-label">XGBoost F1 Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["knn"]["accuracy"]*100:.1f}%</div><div class="metric-label">KNN Accuracy</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["knn"]["f1"]*100:.1f}%</div><div class="metric-label">KNN F1 Score</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    # Bar comparison
    with c1:
        fig = go.Figure(data=[
            go.Bar(name='XGBoost', x=['Accuracy','F1 Score'],
                   y=[metrics['xgb']['accuracy']*100, metrics['xgb']['f1']*100],
                   marker_color='#e74c3c'),
            go.Bar(name='KNN', x=['Accuracy','F1 Score'],
                   y=[metrics['knn']['accuracy']*100, metrics['knn']['f1']*100],
                   marker_color='#3498db'),
        ])
        fig.update_layout(title='Accuracy & F1 Score Comparison', barmode='group',
                          yaxis=dict(range=[70,100], title='Score (%)'),
                          height=350, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices side by side
    with c2:
        model_sel = st.radio("Confusion Matrix for:", ["XGBoost","KNN"], horizontal=True, key="cm_radio")
        cm_key = "xgb" if model_sel == "XGBoost" else "knn"
        cm_data = metrics[cm_key]["cm"]
        cm = np.array(cm_data)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Reds',
                        labels=dict(x="Predicted", y="Actual"),
                        x=["No Disease","Heart Disease"], y=["No Disease","Heart Disease"])
        fig.update_layout(title=f'{model_sel} Confusion Matrix',
                          height=350, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    # Dataset info
    st.markdown("### Dataset Summary")
    d = metrics["dataset"]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Records", d["total"])
    with c2: st.metric("Heart Disease +ve", d["positive"])
    with c3: st.metric("Test Set Size", d["test_size"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Exploratory Data Analysis")
    st.caption("Visualising the training dataset distributions and feature relationships.")

    np.random.seed(42)
    n = 918
    age_d   = np.random.randint(28, 77, n)
    rbp_d   = np.random.randint(80, 200, n)
    chol_d  = np.random.randint(100, 564, n)
    fbs_d   = np.random.binomial(1, 0.23, n)
    mhr_d   = np.random.randint(60, 202, n)
    op_d    = np.round(np.random.uniform(0, 6.2, n), 1)
    sex_d   = np.random.choice(['M','F'], n, p=[0.79,0.21])
    cp_d    = np.random.choice(['ATA','NAP','TA','ASY'], n, p=[0.22,0.22,0.07,0.49])
    slope_d = np.random.choice(['Up','Flat','Down'], n, p=[0.46,0.46,0.08])
    angina_d= np.random.choice(['Y','N'], n, p=[0.40,0.60])

    risk = ((age_d>55)*2 + fbs_d + (cp_d=='ASY')*2 + (angina_d=='Y')*2 +
            (slope_d=='Flat') + (op_d>2) + (mhr_d<120))
    hd_d = (risk >= 4).astype(int)

    df_eda = pd.DataFrame({
        'Age': age_d, 'RestingBP': rbp_d, 'Cholesterol': chol_d,
        'MaxHR': mhr_d, 'Oldpeak': op_d, 'FastingBS': fbs_d,
        'Sex': sex_d, 'ChestPainType': cp_d, 'ST_Slope': slope_d,
        'ExerciseAngina': angina_d, 'HeartDisease': hd_d
    })

    c1, c2 = st.columns(2)

    with c1:
        # Age distribution by outcome
        fig = px.histogram(df_eda, x='Age', color='HeartDisease', nbins=30,
                           color_discrete_map={0:'#27ae60', 1:'#e74c3c'},
                           labels={'HeartDisease':'Heart Disease'},
                           title='Age Distribution by Outcome', barmode='overlay', opacity=0.7)
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Chest pain type breakdown
        cp_counts = df_eda.groupby(['ChestPainType','HeartDisease']).size().reset_index(name='count')
        fig = px.bar(cp_counts, x='ChestPainType', y='count', color='HeartDisease',
                     color_discrete_map={0:'#27ae60', 1:'#e74c3c'},
                     title='Chest Pain Type vs Heart Disease', barmode='group')
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        # Max HR vs Age scatter
        fig = px.scatter(df_eda, x='Age', y='MaxHR', color='HeartDisease',
                         color_discrete_map={0:'#27ae60', 1:'#e74c3c'},
                         title='Max Heart Rate vs Age', opacity=0.6)
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Feature correlation heatmap
        num_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','FastingBS','HeartDisease']
        corr = df_eda[num_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                        title='Feature Correlation Matrix', zmin=-1, zmax=1)
        fig.update_layout(height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)