import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Athlete Injury Risk Demo", page_icon="🏃", layout="centered")

MODEL_PATH = "injury_model.pkl"
SCALER_PATH = "injury_scaler.pkl"

NUMERIC_COLS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "Training_Intensity",
    "Training_Hours_Per_Week",
    "Recovery_Days_Per_Week",
    "Match_Count_Per_Week",
    "Rest_Between_Events_Days",
    "Fatigue_Score",
    "Performance_Score",
    "Team_Contribution_Score",
    "Load_Balance_Score",
    "ACL_Risk_Score",
]

EXPECTED_COLS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "Training_Intensity",
    "Training_Hours_Per_Week",
    "Recovery_Days_Per_Week",
    "Match_Count_Per_Week",
    "Rest_Between_Events_Days",
    "Fatigue_Score",
    "Performance_Score",
    "Team_Contribution_Score",
    "Load_Balance_Score",
    "ACL_Risk_Score",
    "Gender_Male",
    "Position_Forward",
    "Position_Guard",
]


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(SCALER_PATH):
        st.error(
            f"Scaler file not found: {SCALER_PATH}\n\n"
            "Save the fitted scaler from your notebook with:\n"
            'joblib.dump(scaler, "injury_scaler.pkl")'
        )
        st.stop()

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    model_cols = list(getattr(model, "feature_names_in_", EXPECTED_COLS))
    return model, scaler, model_cols


def build_input_frame(raw_inputs: dict, model_columns: list[str]) -> pd.DataFrame:
    row = {
        "Age": raw_inputs["Age"],
        "Height_cm": raw_inputs["Height_cm"],
        "Weight_kg": raw_inputs["Weight_kg"],
        "Training_Intensity": raw_inputs["Training_Intensity"],
        "Training_Hours_Per_Week": raw_inputs["Training_Hours_Per_Week"],
        "Recovery_Days_Per_Week": raw_inputs["Recovery_Days_Per_Week"],
        "Match_Count_Per_Week": raw_inputs["Match_Count_Per_Week"],
        "Rest_Between_Events_Days": raw_inputs["Rest_Between_Events_Days"],
        "Fatigue_Score": raw_inputs["Fatigue_Score"],
        "Performance_Score": raw_inputs["Performance_Score"],
        "Team_Contribution_Score": raw_inputs["Team_Contribution_Score"],
        "Load_Balance_Score": raw_inputs["Load_Balance_Score"],
        "ACL_Risk_Score": raw_inputs["ACL_Risk_Score"],
        "Gender_Male": 1 if raw_inputs["Gender"] == "Male" else 0,
        "Position_Forward": 1 if raw_inputs["Position"] == "Forward" else 0,
        "Position_Guard": 1 if raw_inputs["Position"] == "Guard" else 0,
    }

    df = pd.DataFrame([row])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    return df


def risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low Risk"
    if probability < 0.66:
        return "Medium Risk"
    return "High Risk"


st.title("Athlete Injury Risk Assessment")
st.caption("This app uses a pre-trained machine learning model plus the fitted scaler used during training.")

with st.expander("Files needed in your GitHub repo", expanded=False):
    st.code(
        "app.py\ninjury_model.pkl\ninjury_scaler.pkl\nrequirements.txt",
        language="text",
    )

model, scaler, model_columns = load_artifacts()

with st.form("injury_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=22)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=180.0)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
        gender = st.selectbox("Gender", ["Female", "Male"])
        position = st.selectbox("Position", ["Center", "Forward", "Guard"])
        training_intensity = st.slider("Training Intensity", min_value=1, max_value=10, value=6)
        training_hours = st.number_input("Training Hours Per Week", min_value=0.0, max_value=60.0, value=10.0)

    with col2:
        recovery_days = st.number_input("Recovery Days Per Week", min_value=0.0, max_value=7.0, value=2.0)
        match_count = st.number_input("Match Count Per Week", min_value=0.0, max_value=14.0, value=2.0)
        rest_days = st.number_input("Rest Between Events (days)", min_value=0.0, max_value=30.0, value=2.0)
        fatigue = st.slider("Fatigue Score", min_value=1, max_value=10, value=5)
        performance = st.slider("Performance Score", min_value=1, max_value=10, value=7)
        team_contribution = st.slider("Team Contribution Score", min_value=1, max_value=10, value=7)
        load_balance = st.slider("Load Balance Score", min_value=1, max_value=10, value=6)
        acl_risk = st.slider("ACL Risk Score", min_value=1, max_value=10, value=4)

    submitted = st.form_submit_button("Predict Injury Risk")

if submitted:
    raw_inputs = {
        "Age": age,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Gender": gender,
        "Position": position,
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
    }

    input_df = build_input_frame(raw_inputs, model_columns)

    scaled_df = input_df.copy()
    numeric_cols_to_scale = [col for col in NUMERIC_COLS if col in scaled_df.columns]
    scaled_df[numeric_cols_to_scale] = scaler.transform(scaled_df[numeric_cols_to_scale])

    predicted_class = int(model.predict(scaled_df)[0])
    probabilities = model.predict_proba(scaled_df)[0]
    injury_probability = float(probabilities[1])
    band = risk_band(injury_probability)

    st.subheader("Prediction Result")

    if predicted_class == 1:
        st.error(f"Predicted outcome: Injured ({band})")
    else:
        st.success(f"Predicted outcome: Not Injured ({band})")

    st.metric("Estimated injury probability", f"{injury_probability * 100:.2f}%")
    st.write(f"Class 0 probability: {probabilities[0] * 100:.2f}%")
    st.write(f"Class 1 probability: {probabilities[1] * 100:.2f}%")

    with st.expander("Preview of encoded + scaled model input"):
        st.dataframe(scaled_df, use_container_width=True)

    st.info(
        "This version applies the same scaler used during training before making a prediction, "
        "which fixes the mismatch from the earlier app."
    )
