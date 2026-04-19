
import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Athlete Injury Risk Assessment", page_icon="🏃", layout="wide")

MODEL_PATHS = ["injury_model.pkl", "/mnt/data/injury_model.pkl"]
RAW_NUMERIC_COLS = [
    "Age", "Height_cm", "Weight_kg", "Training_Intensity",
    "Training_Hours_Per_Week", "Recovery_Days_Per_Week",
    "Match_Count_Per_Week", "Rest_Between_Events_Days",
    "Fatigue_Score", "Performance_Score", "Team_Contribution_Score",
    "Load_Balance_Score", "ACL_Risk_Score"
]
EXPECTED_MODEL_COLUMNS = RAW_NUMERIC_COLS + ["Gender_Male", "Position_Forward", "Position_Guard"]
RAW_REQUIRED_COLUMNS = RAW_NUMERIC_COLS + ["Gender", "Position"]

def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    raise FileNotFoundError("Could not find injury_model.pkl")

@st.cache_data(show_spinner=False)
def fit_scaler_from_training_csv(df: pd.DataFrame):
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")
    scaler = StandardScaler()
    scaler.fit(df[RAW_NUMERIC_COLS])
    return scaler

def build_feature_frame(raw_df: pd.DataFrame, scaler: StandardScaler | None):
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {', '.join(missing)}")

    work = raw_df.copy()

    # Normalize category text a little.
    work["Gender"] = work["Gender"].astype(str).str.strip().str.title()
    work["Position"] = work["Position"].astype(str).str.strip().str.title()

    encoded = pd.get_dummies(work, columns=["Gender", "Position"], drop_first=True)

    for col in EXPECTED_MODEL_COLUMNS:
        if col not in encoded.columns:
            encoded[col] = 0

    encoded = encoded[EXPECTED_MODEL_COLUMNS]

    if scaler is not None:
        encoded[RAW_NUMERIC_COLS] = scaler.transform(encoded[RAW_NUMERIC_COLS])

    return encoded

def risk_band(prob: float) -> str:
    if prob < 0.33:
        return "Low Risk"
    if prob < 0.66:
        return "Medium Risk"
    return "High Risk"

def get_template_df():
    return pd.DataFrame([{
        "Age": 20,
        "Height_cm": 180,
        "Weight_kg": 75,
        "Training_Intensity": 5,
        "Training_Hours_Per_Week": 8,
        "Recovery_Days_Per_Week": 3,
        "Match_Count_Per_Week": 1,
        "Rest_Between_Events_Days": 2,
        "Fatigue_Score": 3,
        "Performance_Score": 85,
        "Team_Contribution_Score": 80,
        "Load_Balance_Score": 80,
        "ACL_Risk_Score": 50,
        "Gender": "Male",
        "Position": "Forward"
    }])

model, model_path = load_model()
feature_names = list(getattr(model, "feature_names_in_", EXPECTED_MODEL_COLUMNS))
model_cols_ok = feature_names == EXPECTED_MODEL_COLUMNS

st.title("Athlete Injury Risk Assessment")
st.caption("Streamlit app built around your saved Logistic Regression `.pkl` model.")

with st.sidebar:
    st.subheader("Model")
    st.write(f"Loaded model from: `{model_path}`")
    st.write(f"Model type: `{type(model).__name__}`")
    st.write(f"Expected features: `{len(feature_names)}`")
    if not model_cols_ok:
        st.warning("The model's saved feature names differ from the expected notebook schema. Predictions may fail unless the training CSV matches the model.")
    st.divider()
    st.subheader("Optional: training CSV")
    st.write("Upload the original training CSV to rebuild the scaler from your notebook workflow. This is the most reliable way to use the `.pkl` correctly.")
    train_file = st.file_uploader("Training CSV", type=["csv"], key="training_csv")
    use_identity = st.checkbox("Allow predictions without training CSV", value=False)
    st.caption("Without the original training CSV, the model can still load, but numeric fields will not be scaled the same way they were during training.")

scaler = None
scaler_ready = False

if train_file is not None:
    try:
        train_df = pd.read_csv(train_file)
        scaler = fit_scaler_from_training_csv(train_df)
        scaler_ready = True
        st.success("Training CSV loaded. Scaler recreated from raw numeric columns.")
    except Exception as e:
        st.error(f"Could not use training CSV: {e}")
elif use_identity:
    st.warning("Using the model without the original scaler. Predictions may be unreliable.")
else:
    st.info("Upload the original training CSV for notebook-consistent predictions, or enable the fallback checkbox in the sidebar.")

tab1, tab2, tab3 = st.tabs(["Single Athlete", "Batch Prediction", "How It Works"])

with tab1:
    st.subheader("Single-athlete prediction")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        height = st.number_input("Height (cm)", min_value=120, max_value=230, value=180)
        weight = st.number_input("Weight (kg)", min_value=35, max_value=180, value=75)
        training_intensity = st.slider("Training Intensity", 1, 10, 5)
        training_hours = st.number_input("Training Hours / Week", min_value=0, max_value=40, value=8)

    with c2:
        recovery_days = st.number_input("Recovery Days / Week", min_value=0, max_value=7, value=3)
        match_count = st.number_input("Match Count / Week", min_value=0, max_value=14, value=1)
        rest_days = st.number_input("Rest Between Events (Days)", min_value=0, max_value=14, value=2)
        fatigue = st.slider("Fatigue Score", 1, 10, 3)
        performance = st.slider("Performance Score", 0, 100, 85)

    with c3:
        team_contribution = st.slider("Team Contribution Score", 0, 100, 80)
        load_balance = st.slider("Load Balance Score", 0, 100, 80)
        acl_risk = st.slider("ACL Risk Score", 0, 100, 50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        position = st.selectbox("Position", ["Forward", "Guard", "Center"])

    raw_single = pd.DataFrame([{
        "Age": age,
        "Height_cm": height,
        "Weight_kg": weight,
        "Training_Intensity": training_intensity,
        "Training_Hours_Per_Week": training_hours,
        "Recovery_Days_Per_Week": recovery_days,
        "Match_Count_Per_Week": match_count,
        "Rest_Between_Events_Days": rest_days,
        "Fatigue_Score": fatigue,
        "Performance_Score": performance,
        "Team_Contribution_Score": team_contribution,
        "Load_Balance_Score": load_balance,
        "ACL_Risk_Score": acl_risk,
        "Gender": gender,
        "Position": position
    }])

    can_predict = scaler_ready or use_identity
    if st.button("Predict injury risk", type="primary", disabled=not can_predict):
        try:
            features = build_feature_frame(raw_single, scaler if scaler_ready else None)
            prob = float(model.predict_proba(features)[:, 1][0])
            pred = int(model.predict(features)[0])
            band = risk_band(prob)

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted class", "Injured" if pred == 1 else "Not Injured")
            m2.metric("Injury probability", f"{prob:.1%}")
            m3.metric("Risk band", band)

            st.dataframe(features, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV with the raw notebook columns:")
    st.code(", ".join(RAW_REQUIRED_COLUMNS), language="text")

    template_df = get_template_df()
    st.download_button(
        "Download example CSV template",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="athlete_prediction_template.csv",
        mime="text/csv"
    )

    batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch_csv")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            features = build_feature_frame(batch_df, scaler if scaler_ready else None)
            probs = model.predict_proba(features)[:, 1]
            preds = model.predict(features)

            results = batch_df.copy()
            results["Predicted_Class"] = np.where(preds == 1, "Injured", "Not Injured")
            results["Injury_Probability"] = probs
            results["Risk_Band"] = [risk_band(p) for p in probs]

            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="athlete_injury_predictions.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

with tab3:
    st.subheader("How this version works")
    st.markdown(
        """
        - This app loads your saved `injury_model.pkl`.
        - Your notebook trained the logistic regression model on **scaled numeric features** plus dummy-encoded categorical variables.
        - The `.pkl` you uploaded contains the trained model, but **not** the fitted scaler.
        - Because of that, the most accurate setup is:
          1. keep `injury_model.pkl` in the repo  
          2. upload the original training CSV in the sidebar  
          3. let the app rebuild the scaler before predicting
        - If you use the fallback mode without the training CSV, the app can still run, but the numeric features will not match the notebook preprocessing exactly.
        """
    )

    st.write("Expected model columns:")
    st.json(feature_names)
