import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

st.set_page_config(page_title="Injury Risk Assessment", layout="wide")

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

DEFAULT_CSV_PATHS = [
    "collegiate_athlete_injury_dataset.csv",
    "data/collegiate_athlete_injury_dataset.csv",
]

DEFAULT_PKL_PATHS = [
    "injury_model.pkl",
    "models/injury_model.pkl",
]

LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66


def risk_label(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "Low Risk"
    if prob < HIGH_THRESHOLD:
        return "Medium Risk"
    return "High Risk"


def find_first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def load_default_csv():
    path = find_first_existing(DEFAULT_CSV_PATHS)
    if path:
        return pd.read_csv(path), path
    return None, None


def load_default_model():
    path = find_first_existing(DEFAULT_PKL_PATHS)
    if path:
        return joblib.load(path), path
    return None, None


def preprocess_training_df(df: pd.DataFrame):
    df = df.copy()
    if "Injury_Indicator" not in df.columns:
        raise ValueError("The dataset must contain an 'Injury_Indicator' column.")
    df_encoded = pd.get_dummies(df, columns=["Gender", "Position"], drop_first=True)
    X = df_encoded.drop("Injury_Indicator", axis=1)
    y = df_encoded["Injury_Indicator"]

    numeric_present = [c for c in NUMERIC_COLS if c in X.columns]
    scaler = StandardScaler()
    X_scaled = X.copy()
    if numeric_present:
        X_scaled[numeric_present] = scaler.fit_transform(X_scaled[numeric_present])

    return X_scaled, y, scaler, list(X.columns), numeric_present


def train_model_from_csv(df: pd.DataFrame):
    X_scaled, y, scaler, training_columns, numeric_present = preprocess_training_df(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        C=0.01,
        class_weight="balanced",
        max_iter=1000,
        penalty="l2",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": float(accuracy_score(y_test, preds)),
        "Precision": float(precision_score(y_test, preds, zero_division=0)),
        "Recall": float(recall_score(y_test, preds, zero_division=0)),
        "Confusion Matrix": confusion_matrix(y_test, preds),
        "Average Predicted Risk": float(np.mean(probs)),
    }

    artifact = {
        "model": model,
        "scaler": scaler,
        "training_columns": training_columns,
        "numeric_cols": numeric_present,
    }
    return artifact, metrics


def build_user_input():
    st.subheader("Athlete Profile Input")

    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Male", "Female"])
        position = st.selectbox(
            "Position",
            ["Forward", "Midfielder", "Defender", "Goalkeeper", "Guard", "Center", "Other"],
        )
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0)
        weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0)
        training_intensity = st.slider("Training Intensity", 1, 10, 6)
        training_hours = st.number_input("Training Hours per Week", min_value=0.0, max_value=60.0, value=12.0)

    with right:
        recovery_days = st.number_input("Recovery Days per Week", min_value=0.0, max_value=7.0, value=2.0)
        match_count = st.number_input("Match Count per Week", min_value=0.0, max_value=14.0, value=1.0)
        rest_between = st.number_input("Rest Between Events (Days)", min_value=0.0, max_value=14.0, value=2.0)
        fatigue_score = st.slider("Fatigue Score", 1, 10, 5)
        performance_score = st.slider("Performance Score", 1, 10, 6)
        team_contribution = st.slider("Team Contribution Score", 1, 10, 6)
        load_balance = st.slider("Load Balance Score", 1, 10, 5)
        acl_risk = st.slider("ACL Risk Score", 1, 10, 4)

    athlete_data = {
        "Gender": gender,
        "Position": position,
        "Age": age,
        "Height_cm": height_cm,
        "Weight_kg": weight_kg,
        "Training_Intensity": training_intensity,
        "Training_Hours_Per_Week": training_hours,
        "Recovery_Days_Per_Week": recovery_days,
        "Match_Count_Per_Week": match_count,
        "Rest_Between_Events_Days": rest_between,
        "Fatigue_Score": fatigue_score,
        "Performance_Score": performance_score,
        "Team_Contribution_Score": team_contribution,
        "Load_Balance_Score": load_balance,
        "ACL_Risk_Score": acl_risk,
    }
    return athlete_data


def prepare_input_from_dict(athlete_data, training_columns, scaler, numeric_cols):
    athlete_df = pd.DataFrame([athlete_data])
    athlete_df_encoded = pd.get_dummies(athlete_df)

    for col in training_columns:
        if col not in athlete_df_encoded.columns:
            athlete_df_encoded[col] = 0

    athlete_df_encoded = athlete_df_encoded[training_columns]

    if numeric_cols:
        athlete_df_encoded[numeric_cols] = scaler.transform(athlete_df_encoded[numeric_cols])

    return athlete_df_encoded


def predict_with_full_artifact(artifact, athlete_data):
    prepared = prepare_input_from_dict(
        athlete_data,
        artifact["training_columns"],
        artifact["scaler"],
        artifact["numeric_cols"],
    )
    prob = float(artifact["model"].predict_proba(prepared)[:, 1][0])
    return prob, risk_label(prob)


def predict_with_pkl_only(model, athlete_data):
    st.warning(
        "This section uses the saved .pkl file directly. "
        "Because the .pkl file contains the trained model but not the original preprocessing pipeline, "
        "the prediction below uses the raw feature order expected by the saved model. "
        "This is useful for demonstrating a previously trained model, but the CSV-trained version is the more complete workflow."
    )

    direct_df = pd.DataFrame([{
        "Age": athlete_data["Age"],
        "Height_cm": athlete_data["Height_cm"],
        "Weight_kg": athlete_data["Weight_kg"],
        "Training_Intensity": athlete_data["Training_Intensity"],
        "Training_Hours_Per_Week": athlete_data["Training_Hours_Per_Week"],
        "Recovery_Days_Per_Week": athlete_data["Recovery_Days_Per_Week"],
        "Match_Count_Per_Week": athlete_data["Match_Count_Per_Week"],
        "Rest_Between_Events_Days": athlete_data["Rest_Between_Events_Days"],
        "Fatigue_Score": athlete_data["Fatigue_Score"],
        "Performance_Score": athlete_data["Performance_Score"],
        "Team_Contribution_Score": athlete_data["Team_Contribution_Score"],
        "Load_Balance_Score": athlete_data["Load_Balance_Score"],
        "ACL_Risk_Score": athlete_data["ACL_Risk_Score"],
        "Gender_Male": 1 if athlete_data["Gender"] == "Male" else 0,
        "Position_Forward": 1 if athlete_data["Position"] == "Forward" else 0,
        "Position_Guard": 1 if athlete_data["Position"] == "Guard" else 0,
        "Position_Midfielder": 1 if athlete_data["Position"] == "Midfielder" else 0,
        "Position_Other": 1 if athlete_data["Position"] == "Other" else 0,
    }])

    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        for col in expected:
            if col not in direct_df.columns:
                direct_df[col] = 0
        direct_df = direct_df[list(expected)]

    prob = float(model.predict_proba(direct_df)[:, 1][0])
    return prob, risk_label(prob)


def show_risk_explanation():
    st.markdown(
        f"""
### Risk Categories
- **Low Risk:** below **{LOW_THRESHOLD:.0%}**
- **Medium Risk:** **{LOW_THRESHOLD:.0%}** to **{HIGH_THRESHOLD:.0%}**
- **High Risk:** **{HIGH_THRESHOLD:.0%}** and above

These categories are based on the model's predicted probability of the outcome.  
They are intended as a simple interpretation layer:

- a lower percentage suggests lower estimated risk,
- a middle percentage suggests moderate estimated risk,
- a higher percentage suggests greater estimated risk relative to the other categories.

This is a decision-support style output and should be interpreted as a model estimate rather than a clinical determination.
"""
    )


st.title("Athlete Injury Risk Assessment")
st.caption("Generic demonstration app for comparing a previously saved model with a newly trained model.")

show_risk_explanation()

default_model, default_model_path = load_default_model()
default_csv, default_csv_path = load_default_csv()

tab1, tab2, tab3 = st.tabs([
    "Use Saved .pkl Model",
    "Train from CSV and Predict",
    "Compare Both Approaches",
])

with st.sidebar:
    st.header("Available Project Files")
    if default_model_path:
        st.success(f"Saved model found: {default_model_path}")
    else:
        st.info("No default .pkl model found in the repo root.")
    if default_csv_path:
        st.success(f"CSV found: {default_csv_path}")
    else:
        st.info("No default CSV found in the repo root.")
    st.markdown(
        """
**App Modes**
- **Saved .pkl model:** uses a previously trained model artifact
- **Train from CSV:** fits a fresh model using the dataset
- **Compare:** shows both results side by side
"""
    )

athlete_input = build_user_input()

with tab1:
    st.subheader("Prediction from Previously Trained Model (.pkl)")
    st.write(
        "This section pulls from an already trained data model saved as a `.pkl` file. "
        "It is useful when you want to demonstrate prediction with an existing model without retraining in the app."
    )

    uploaded_pkl = st.file_uploader("Optional: upload a .pkl model", type=["pkl"], key="pkl_only")
    model_for_tab1 = None

    if uploaded_pkl is not None:
        model_for_tab1 = joblib.load(uploaded_pkl)
        st.success("Uploaded .pkl model loaded.")
    elif default_model is not None:
        model_for_tab1 = default_model
        st.info(f"Using saved model from: {default_model_path}")

    if model_for_tab1 is not None:
        if st.button("Predict with Saved .pkl Model"):
            prob, label = predict_with_pkl_only(model_for_tab1, athlete_input)
            st.metric("Predicted Risk", f"{prob:.1%}")
            st.metric("Risk Category", label)
    else:
        st.warning("Add `injury_model.pkl` to the repo root or upload one to use this section.")

with tab2:
    st.subheader("Train a New Model from CSV and Predict")
    st.write(
        "This section trains a model from the source dataset and uses the full preprocessing workflow before making a prediction."
    )

    uploaded_csv = st.file_uploader("Optional: upload the source CSV", type=["csv"], key="csv_train")
    df_for_training = None

    if uploaded_csv is not None:
        df_for_training = pd.read_csv(uploaded_csv)
        st.success("Uploaded CSV loaded.")
    elif default_csv is not None:
        df_for_training = default_csv
        st.info(f"Using dataset from: {default_csv_path}")

    if df_for_training is not None:
        if st.button("Train Model from CSV and Predict"):
            artifact, metrics = train_model_from_csv(df_for_training)
            st.session_state["trained_artifact"] = artifact
            st.session_state["train_metrics"] = metrics

            prob, label = predict_with_full_artifact(artifact, athlete_input)
            st.metric("Predicted Risk", f"{prob:.1%}")
            st.metric("Risk Category", label)

            c1, c2, c3 = st.columns(3)
            c1.metric("Accuracy", f'{metrics["Accuracy"]:.3f}')
            c2.metric("Precision", f'{metrics["Precision"]:.3f}')
            c3.metric("Recall", f'{metrics["Recall"]:.3f}')
            st.write("Confusion Matrix")
            st.dataframe(pd.DataFrame(metrics["Confusion Matrix"]))

            buffer = io.BytesIO()
            joblib.dump(artifact, buffer)
            buffer.seek(0)
            st.download_button(
                "Download trained model bundle",
                data=buffer,
                file_name="injury_model_bundle.pkl",
                mime="application/octet-stream",
            )
    else:
        st.warning("Add the CSV to the repo root or upload it to train a new model.")

with tab3:
    st.subheader("Compare Saved Model vs CSV-Trained Model")
    st.write(
        "This comparison view shows a prediction from the previously saved `.pkl` model alongside a prediction from a model trained from the CSV."
    )

    compare_pkl = None
    compare_df = None

    uploaded_compare_pkl = st.file_uploader("Optional: upload a .pkl model for comparison", type=["pkl"], key="compare_pkl")
    uploaded_compare_csv = st.file_uploader("Optional: upload a CSV for comparison training", type=["csv"], key="compare_csv")

    if uploaded_compare_pkl is not None:
        compare_pkl = joblib.load(uploaded_compare_pkl)
    elif default_model is not None:
        compare_pkl = default_model

    if uploaded_compare_csv is not None:
        compare_df = pd.read_csv(uploaded_compare_csv)
    elif default_csv is not None:
        compare_df = default_csv

    if st.button("Run Side-by-Side Comparison"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Saved .pkl Model")
            if compare_pkl is not None:
                pkl_prob, pkl_label = predict_with_pkl_only(compare_pkl, athlete_input)
                st.metric("Predicted Risk", f"{pkl_prob:.1%}")
                st.metric("Risk Category", pkl_label)
                st.caption("Uses a previously trained model artifact.")
            else:
                st.warning("No .pkl model available.")

        with col2:
            st.markdown("#### Newly Trained CSV Model")
            if compare_df is not None:
                artifact, metrics = train_model_from_csv(compare_df)
                csv_prob, csv_label = predict_with_full_artifact(artifact, athlete_input)
                st.metric("Predicted Risk", f"{csv_prob:.1%}")
                st.metric("Risk Category", csv_label)
                st.caption("Uses the CSV to train a fresh model in the app.")
                with st.expander("Training Metrics"):
                    st.write({k: v for k, v in metrics.items() if k != "Confusion Matrix"})
                    st.dataframe(pd.DataFrame(metrics["Confusion Matrix"]))
            else:
                st.warning("No CSV available for training.")
