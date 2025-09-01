import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import os

# ==============================
# 1. Load Dataset
# ==============================
df = pd.read_csv("data/hospital_readmissions_30k.csv")

# ==============================
# 2. Feature Engineering
# ==============================
df[['systolic_bp', 'diastolic_bp']] = df['blood_pressure'].str.split('/', expand=True).astype(float)
df.drop(['blood_pressure', 'patient_id'], axis=1, inplace=True)

# Encode binary yes/no
for col in ['diabetes', 'hypertension']:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Encode categorical
for col in ['gender', 'discharge_destination']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Encode target
df['readmitted_30_days'] = df['readmitted_30_days'].map({'Yes': 1, 'No': 0})

# New features
df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
df['high_bp_flag'] = ((df['systolic_bp'] > 140) | (df['diastolic_bp'] > 90)).astype(int)
df['comorbidity_index'] = df['diabetes'] + df['hypertension']

df['bmi_category'] = pd.cut(
    df['bmi'], bins=[0,18.5,24.9,29.9,100],
    labels=['Underweight','Normal','Overweight','Obese']
)
df['bmi_category'] = LabelEncoder().fit_transform(df['bmi_category'])

df['discharge_risk'] = df['discharge_destination']
df['meds_per_day'] = df['medication_count'] / (1 + df['length_of_stay'])
df['meds_per_comorbidity'] = df['medication_count'] / (1 + df['comorbidity_index'])

print("✅ Feature Engineering Done!")

# ==============================
# 3. Features & Target
# ==============================
X = df.drop("readmitted_30_days", axis=1)
y = df["readmitted_30_days"]

# ==============================
# 4. Scale + Handle Imbalance
# ==============================
scaler = StandardScaler()
X_scaled = X.copy()
num_cols = X_scaled.select_dtypes(include=["int64", "float64"]).columns
X_scaled[num_cols] = scaler.fit_transform(X_scaled[num_cols])

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("✅ Preprocessing Done!")

# ==============================
# 5. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

# ==============================
# 6. Define Models
# ==============================
models = {
    
    "lightgbm": LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42
    )
}

# ==============================
# 7. Train & Evaluate Models
# ==============================
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    results.append((name, acc, auc))

    print("="*40)
    print(f"📌 {name.upper()} Results")
    print("Accuracy:", acc)
    print("ROC-AUC:", auc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 8. Save All Models + Scaler + Features + Num Cols
# ==============================
os.makedirs("models", exist_ok=True)
for name, model in models.items():
    joblib.dump(model, f"models/{name}_model.pkl")
    print(f"💾 Saved {name} model as models/{name}_model.pkl")

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/feature_names.pkl")
joblib.dump(num_cols.tolist(), "models/num_cols.pkl")

print("✅ All models, Scaler, Features, and Num Cols saved in models/")
from sklearn.preprocessing import LabelEncoder

def fit_with_other(values):
    le = LabelEncoder()
    unique_vals = list(set(values))
    if "Other" not in unique_vals:
        unique_vals.append("Other")
    le.fit(unique_vals)
    return le

# Example:
le_gender = fit_with_other(df["gender"])
joblib.dump(le_gender, "models/le_gender.pkl")
