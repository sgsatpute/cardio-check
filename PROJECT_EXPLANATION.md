# 📋 CardioCheck — Full Project Explanation

> A complete technical write-up covering the problem statement, dataset, preprocessing, model training, evaluation, and deployment of the CardioCheck heart disease risk predictor.

---

## 1. Problem Statement

Cardiovascular disease (CVD) is the number one cause of death globally, responsible for approximately 17.9 million deaths per year according to the World Health Organization. A significant proportion of these deaths are preventable with early detection and lifestyle intervention.

Traditional cardiac screening requires access to clinical labs, ECG machines, and specialist consultations — resources that are not always readily available. The goal of this project is to build a machine learning-powered web application that can perform a **rapid, accessible, data-driven risk assessment** for heart disease based on a patient's clinical parameters, without requiring specialist equipment.

**Objective:** Given 11 clinical features of a patient, predict whether they are at **High Risk** or **Low Risk** of heart disease.

---

## 2. Dataset

### Source
The dataset is based on the **UCI Heart Failure Prediction Dataset**, which aggregates data from five heart disease databases (Cleveland, Hungarian, Switzerland, Long Beach VA, and Stalog). It is one of the most widely used datasets for cardiovascular ML research.

### Composition
| Property | Value |
|---|---|
| Total Records | 918 |
| Features | 11 clinical input features |
| Target Variable | `HeartDisease` (0 = No Disease, 1 = Disease) |
| Class Distribution | ~65% positive (Heart Disease), ~35% negative |

### Features Description

| Feature | Data Type | Range / Categories | Clinical Significance |
|---|---|---|---|
| Age | Integer | 28–77 years | Risk increases significantly after 55 |
| Sex | Categorical | Male / Female | Males at higher risk |
| ChestPainType | Categorical | ATA, NAP, TA, ASY | ASY (Asymptomatic) strongly correlated with disease |
| RestingBP | Integer | 80–200 mm Hg | Elevated BP indicates cardiovascular stress |
| Cholesterol | Integer | 100–564 mg/dL | High LDL linked to artery plaque |
| FastingBS | Binary | 0 / 1 | Blood sugar > 120 mg/dL indicates diabetes risk |
| RestingECG | Categorical | Normal, ST, LVH | ECG anomalies indicate electrical/structural issues |
| MaxHR | Integer | 60–202 bpm | Lower max HR suggests cardiac limitation |
| ExerciseAngina | Binary | Yes / No | Chest pain during exercise indicates ischemia |
| Oldpeak | Float | 0.0–6.2 | ST segment depression — key ischemia indicator |
| ST_Slope | Categorical | Up, Flat, Down | Flat/Down slope associated with higher risk |

---

## 3. Data Preprocessing

### 3.1 Handling Categorical Variables
Categorical features were converted to numerical format using **one-hot encoding**:

```
Sex         → Sex_M, (Sex_F dropped as reference)
ChestPainType → ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA, (ASY as reference)
RestingECG  → RestingECG_Normal, RestingECG_ST, (LVH as reference)
ExerciseAngina → ExerciseAngina_Y, (N as reference)
ST_Slope    → ST_Slope_Flat, ST_Slope_Up, (Down as reference)
```

This results in **15 binary/numerical features** after encoding.

### 3.2 Feature Scaling
For the KNN model, features were standardised using **StandardScaler** (zero mean, unit variance):

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

XGBoost does not require feature scaling as it is a tree-based method insensitive to feature magnitude.

### 3.3 Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Train: 734 records | Test: 184 records
```

Stratified splitting ensures class distribution is preserved in both sets.

---

## 4. Model Selection & Training

Two classifiers were trained and compared:

### 4.1 K-Nearest Neighbours (KNN)

**Why KNN?**
KNN is a simple, interpretable, non-parametric classifier that makes no assumptions about data distribution. It classifies based on proximity to training samples in feature space.

**Configuration:**
```python
KNeighborsClassifier(n_neighbors=11)
```

`n_neighbors=11` was chosen (odd number to avoid ties, larger than default 5 to reduce overfitting and produce more reliable probability estimates).

**How it works:**
1. For a new patient, compute Euclidean distance to all training samples
2. Select the 11 nearest neighbours
3. Output class = majority vote; probability = fraction of neighbours in each class

**Limitation:** KNN struggles with high-dimensional data and can produce overconfident probabilities with small `k`.

---

### 4.2 XGBoost (Extreme Gradient Boosting)

**Why XGBoost?**
XGBoost is a state-of-the-art gradient boosting algorithm known for high accuracy on tabular data. It handles missing values natively, is robust to outliers, and does not require feature scaling.

**Configuration:**
```python
XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
```

**How it works:**
1. Builds an ensemble of decision trees sequentially
2. Each new tree corrects errors made by the previous ensemble
3. Final prediction = weighted sum of all tree outputs, passed through sigmoid for probability
4. Regularisation (L1/L2) prevents overfitting

**Advantage over KNN:** XGBoost captures complex non-linear feature interactions, handles class imbalance better, and scales efficiently to larger datasets.

---

## 5. Model Evaluation

### 5.1 Results

| Metric | XGBoost | KNN |
|---|---|---|
| **Accuracy** | 98.4% | 85.9% |
| **F1 Score** | 98.7% | 89.2% |
| **Precision** | ~98% | ~88% |
| **Recall** | ~99% | ~91% |

### 5.2 Confusion Matrix

**XGBoost (Test Set — 184 samples):**
```
                Predicted: No    Predicted: Yes
Actual: No           63               1
Actual: Yes           2             118
```
- Only 3 misclassifications out of 184 test samples

**KNN (Test Set — 184 samples):**
```
                Predicted: No    Predicted: Yes
Actual: No           51              13
Actual: Yes          13             107
```
- 26 misclassifications — more false negatives (missed disease cases)

### 5.3 Why XGBoost is Recommended
In a medical screening context, **false negatives** (predicting low risk when disease is present) are more dangerous than false positives. XGBoost's lower false negative rate (2 vs 13) makes it the safer and more accurate choice.

---

## 6. Feature Engineering & One-Hot Encoding

The prediction pipeline maintains a **fixed column order** saved in `heart_columns.pkl`:

```python
expected_columns = [
    'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
    'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
    'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
    'ST_Slope_Flat', 'ST_Slope_Up'
]
```

At inference time, the app:
1. Builds a raw dictionary from user inputs
2. Creates a DataFrame with one-hot encoded columns
3. Fills any missing columns with 0
4. Reorders to match `expected_columns`
5. Scales (KNN only) → predicts → returns probability

---

## 7. Application Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Streamlit App                     │
│                                                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │
│  │   Tab 1     │  │    Tab 2     │  │   Tab 3   │  │
│  │ Risk        │  │ Model        │  │ EDA       │  │
│  │ Assessment  │  │ Performance  │  │ Charts    │  │
│  └──────┬──────┘  └──────────────┘  └───────────┘  │
│         │                                            │
│  ┌──────▼──────────────────────────┐                │
│  │      Preprocessing Pipeline      │                │
│  │  Input → Encode → Scale → Predict│                │
│  └──────┬──────────────────────────┘                │
│         │                                            │
│  ┌──────▼──────┐    ┌──────────────┐                │
│  │ XGBoost     │    │    KNN       │                │
│  │ (default)   │    │ (+ Scaler)   │                │
│  └─────────────┘    └──────────────┘                │
└─────────────────────────────────────────────────────┘
```

### Artefacts Loaded at Startup
| File | Purpose |
|---|---|
| `knn_heart_model.pkl` | Serialised KNN classifier |
| `xgb_heart_model.pkl` | Serialised XGBoost classifier |
| `heart_scaler.pkl` | Fitted StandardScaler for KNN |
| `heart_columns.pkl` | Feature column order for alignment |
| `model_metrics.json` | Pre-computed accuracy, F1, confusion matrices |

---

## 8. EDA Insights

Four exploratory visualisations are embedded in the app:

| Chart | Insight |
|---|---|
| **Age Distribution by Outcome** | Risk rises sharply after age 55; younger patients are predominantly disease-free |
| **Chest Pain Type vs Outcome** | ASY (asymptomatic) chest pain is the strongest single predictor of heart disease |
| **Max HR vs Age Scatter** | Heart disease patients cluster at lower max HR values, especially above age 50 |
| **Feature Correlation Heatmap** | Oldpeak and ExerciseAngina show strongest positive correlation with disease; MaxHR shows negative correlation |

---

## 9. Deployment

The application is deployed on **Streamlit Community Cloud** — a free, managed hosting platform for Streamlit apps.

### Deployment Stack
| Component | Technology |
|---|---|
| App Framework | Streamlit |
| Hosting | Streamlit Community Cloud |
| CI/CD | Auto-deploy on `git push` to `main` |
| Runtime | Python 3.14 (cloud-managed) |
| Package Manager | uv (Streamlit Cloud default) |

### Live URL
🌐 [cardiocheck-saurav.streamlit.app](https://cardiocheck-saurav.streamlit.app)

---

## 10. Limitations & Future Work

### Current Limitations
- Dataset is synthetic/augmented — a production system requires validated clinical data
- KNN confidence scores can be overconfident with small neighbourhood sizes
- No user authentication or data persistence
- App does not explain *why* a prediction was made (no SHAP values)

### Future Improvements
| Enhancement | Description |
|---|---|
| SHAP Explainability | Show which features drove each individual prediction |
| FastAPI Backend | Separate ML inference from UI for scalability |
| Docker + CI/CD | Containerise for production deployment |
| More Models | Random Forest, Logistic Regression for comparison |
| Real Dataset | Train on validated clinical trial data |
| Input Validation | Add clinical range warnings for unusual input values |

---

## 11. Tech Stack Summary

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Web Framework | Streamlit | Latest |
| ML — Boosting | XGBoost | Latest |
| ML — Classic | scikit-learn | Latest |
| Data Processing | Pandas, NumPy | Latest |
| Visualisation | Plotly | Latest |
| Serialisation | Joblib | Latest |
| Deployment | Streamlit Community Cloud | — |

---

## 12. References

1. WHO — Cardiovascular Diseases Fact Sheet (2023)
2. Fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.
4. Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory.

---

*CardioCheck is an open-source educational project. Contributions and feedback are welcome via [GitHub Issues](https://github.com/sgsatpute/cardio-check/issues).*
