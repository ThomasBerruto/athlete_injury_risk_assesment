
import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Athlete Injury Risk Assessment", page_icon="🏀", layout="centered")

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

# Based on the notebook's training columns
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

# Based on notebook describe() output
NOTEBOOK_RANGES = {
    "Age": (18, 24),
    "Height_cm": (160.0, 199.0),
    "Weight_kg": (55.0, 99.0),
    "Training_Intensity": (1, 9),
    "Training_Hours_Per_Week": (5.0, 19.0),
    "Recovery_Days_Per_Week": (1.0, 3.0),
    "Match_Count_Per_Week": (1.0, 4.0),
    "Rest_Between_Events_Days": (1.0, 3.0),
    "Fatigue_Score": (1, 9),
    "Performance_Score": (50, 99),
    "Team_Contribution_Score": (50, 99),
    "Load_Balance_Score": (62, 100),
    "ACL_Risk_Score": (2, 100),
}

DEFAULTS = {
    "Age": 20,
    "Height_cm": 180.0,
    "Weight_kg": 75.0,
    "Training_Intensity": 5,
    "Training_Hours_Per_Week": 8.0,
    "Recovery_Days_Per_Week": 3.0,
    "Match_Count_Per_Week": 1.0,
    "Rest_Between_Events_Days": 2.0,
    "Fatigue_Score": 3,
    "Performance_Score": 85,
    "Team_Contribution_Score": 80,
    "Load_Balance_Score": 80,
    "ACL_Risk_Score": 50,
    "Gender": "Female",
    "Position": "Center",
}

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

    return df[model_columns]

def risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low Risk"
    if probability < 0.66:
        return "Medium Risk"
    return "High Risk"

def validate_ranges(raw_inputs: dict):
    warnings = []
    for key, (min_v, max_v) in NOTEBOOK_RANGES.items():
        value = raw_inputs[key]
        if value < min_v or value > max_v:
            warnings.append(f"{key}: {value} is outside the notebook training range [{min_v}, {max_v}]")
    return warnings

st.title("Athlete Injury Risk Assessment")
st.caption("This Injury predictor model is a Logisitical Regression model that is already trained and can provide an accurate predictions of whether or not you will be injured based on the scores you provide for each feature. If you look at my Github, it utilizes the 'injury_model.pkl' and 'scaler.pkl'")

model, scaler, model_columns = load_artifacts()


with st.form("injury_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=40, value=DEFAULTS["Age"])
        height_cm = st.number_input("Height (cm)", min_value=150.0, max_value=199.0, value=DEFAULTS["Height_cm"])
        weight_kg = st.number_input("Weight (kg)", min_value=55.0, max_value=100.0, value=DEFAULTS["Weight_kg"])
        gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        position = st.selectbox("Position", ["Center", "Forward", "Guard"], index=0)
        training_intensity = st.slider("Training Intensity", min_value=0, max_value=10, value=DEFAULTS["Training_Intensity"])
        training_hours = st.number_input("Training Hours Per Week", min_value=0.0, max_value=40.0, value=DEFAULTS["Training_Hours_Per_Week"])

    with col2:
        recovery_days = st.number_input("Recovery Days Per Week", min_value=0.0, max_value=7.0, value=DEFAULTS["Recovery_Days_Per_Week"])
        match_count = st.number_input("Match Count Per Week", min_value=0.0, max_value=7.0, value=DEFAULTS["Match_Count_Per_Week"])
        rest_days = st.number_input("Rest Between Events (days)", min_value=0.0, max_value=7.0, value=DEFAULTS["Rest_Between_Events_Days"])
        fatigue = st.slider("Fatigue Score", min_value=0, max_value=10, value=DEFAULTS["Fatigue_Score"])
        performance = st.slider("Performance Score", min_value=0, max_value=100, value=DEFAULTS["Performance_Score"])
        team_contribution = st.slider("Team Contribution Score", min_value=0, max_value=100, value=DEFAULTS["Team_Contribution_Score"])
        load_balance = st.slider("Load Balance Score", min_value=0, max_value=100, value=DEFAULTS["Load_Balance_Score"])
        acl_risk = st.slider("ACL Risk Score", min_value=0, max_value=100, value=DEFAULTS["ACL_Risk_Score"])

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

    range_warnings = validate_ranges(raw_inputs)

    input_df = build_input_frame(raw_inputs, model_columns)
    scaled_df = input_df.copy()
    numeric_cols_to_scale = [col for col in NUMERIC_COLS if col in scaled_df.columns]
    scaled_df[numeric_cols_to_scale] = scaler.transform(scaled_df[numeric_cols_to_scale])

    predicted_class = int(model.predict(scaled_df)[0])
    probabilities = model.predict_proba(scaled_df)[0]
    injury_probability = float(probabilities[1])

    st.subheader("Prediction Result")
    band = risk_band(injury_probability)

    if predicted_class == 1:
        st.error(f"Predicted outcome: Injured ({band})")
    else:
        st.success(f"Predicted outcome: Not Injured ({band})")

    st.metric("Estimated injury probability", f"{injury_probability * 100:.2f}%")
    st.write(f"Class 0 probability: {probabilities[0] * 100:.2f}%")
    st.write(f"Class 1 probability: {probabilities[1] * 100:.2f}%")

    if range_warnings:
        st.warning("Some inputs are outside the ranges shown in the notebook training data:")
        for warning in range_warnings:
            st.write(f"- {warning}")

    with st.expander("Encoded + scaled model input"):
        st.dataframe(scaled_df, use_container_width=True)

    
