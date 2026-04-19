import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Athlete Injury Predictor", page_icon="🏃", layout="centered")

MODEL_PATH = "injury_model(2).pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

FEATURE_ORDER = [
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

st.title("Athlete Injury Predictor")
st.write(
    "This app uses a pre-trained machine learning model (`injury_model(2).pkl`) "
    "to predict whether an athlete is likely to be injured based on their input data."
)

with st.expander("Model details", expanded=False):
    st.write("Loaded model type:", type(model).__name__)
    if hasattr(model, "feature_names_in_"):
        st.write("Expected features:", list(model.feature_names_in_))
    else:
        st.write("Expected features:", FEATURE_ORDER)

with st.form("prediction_form"):
    st.subheader("Enter athlete data")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=10, max_value=60, value=22, step=1)
        height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=250.0, value=178.0, step=0.1)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        training_intensity = st.slider("Training Intensity", min_value=1, max_value=10, value=6)
        training_hours = st.number_input("Training Hours Per Week", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
        recovery_days = st.number_input("Recovery Days Per Week", min_value=0.0, max_value=7.0, value=2.0, step=0.5)
        match_count = st.number_input("Match Count Per Week", min_value=0.0, max_value=14.0, value=1.0, step=0.5)
        rest_days = st.number_input("Rest Between Events (Days)", min_value=0.0, max_value=14.0, value=2.0, step=0.5)

    with col2:
        fatigue_score = st.slider("Fatigue Score", min_value=0, max_value=100, value=40)
        performance_score = st.slider("Performance Score", min_value=0, max_value=100, value=70)
        team_contribution = st.slider("Team Contribution Score", min_value=0, max_value=100, value=65)
        load_balance = st.slider("Load Balance Score", min_value=0, max_value=100, value=60)
        acl_risk = st.slider("ACL Risk Score", min_value=0, max_value=100, value=30)

        gender = st.selectbox("Gender", ["Female", "Male"])
        position = st.selectbox("Position", ["Center/Other", "Forward", "Guard"])

    submitted = st.form_submit_button("Predict Injury Risk")

if submitted:
    input_dict = {
        "Age": age,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Training_Intensity": training_intensity,
        "Training_Hours_Per_Week": training_hours,
        "Recovery_Days_Per_Week": recovery_days,
        "Match_Count_Per_Week": match_count,
        "Rest_Between_Events_Days": rest_days,
        "Fatigue_Score": fatigue_score,
        "Performance_Score": performance_score,
        "Team_Contribution_Score": team_contribution,
        "Load_Balance_Score": load_balance,
        "ACL_Risk_Score": acl_risk,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Position_Forward": 1 if position == "Forward" else 0,
        "Position_Guard": 1 if position == "Guard" else 0,
    }

    input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        not_injured_prob = float(probabilities[0]) * 100
        injured_prob = float(probabilities[1]) * 100

        st.metric("Probability of Injury", f"{injured_prob:.2f}%")
        st.metric("Probability of No Injury", f"{not_injured_prob:.2f}%")

    if prediction == 1:
        st.error("Prediction: Athlete is likely to be injured.")
    else:
        st.success("Prediction: Athlete is not likely to be injured.")

    st.write("Input data used for prediction:")
    st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.caption(
    "This Streamlit app uses a previously trained `.pkl` model file. "
    "Make sure `injury_model(2).pkl`, `app.py`, and `requirements.txt` are in the same project folder."
)
