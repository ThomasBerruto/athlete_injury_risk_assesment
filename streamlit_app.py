
import io
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Athlete Injury Risk Assessment", page_icon="🏃", layout="wide")

TARGET_COL = "Injury_Indicator"
DROP_COLS = ["Athlete_ID"]
CATEGORICAL_COLS = ["Gender", "Position"]
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
DEFAULT_GENDERS = ["Male", "Female", "Other"]
DEFAULT_POSITIONS = ["Forward", "Guard", "Center", "Midfielder", "Defender", "Goalkeeper", "Pitcher", "Catcher"]

REQUIRED_COLUMNS = DROP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]


@dataclass
class TrainingArtifacts:
    model: object
    scaler: StandardScaler
    training_columns: list[str]
    metrics: dict
    feature_importance: pd.Series


def validate_dataset(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def prepare_training_data(df: pd.DataFrame):
    df = df.copy()

    if "Athlete_ID" in df.columns:
        df = df.drop(columns=["Athlete_ID"])

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna()

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)

    X = df_encoded.drop(columns=[TARGET_COL])
    y = df_encoded[TARGET_COL].astype(int)

    scaler = StandardScaler()
    X.loc[:, NUMERIC_COLS] = scaler.fit_transform(X[NUMERIC_COLS])

    return X, y, scaler


def train_model(df: pd.DataFrame) -> TrainingArtifacts:
    X, y, scaler = prepare_training_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid={"C": [0.01, 0.1, 1, 10, 100]},
        cv=10,
        scoring="recall",
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)

    metrics = {
        "best_c": grid.best_params_["C"],
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "train_accuracy": best_model.score(X_train, y_train),
        "test_accuracy": best_model.score(X_test, y_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    feature_importance = pd.Series(best_model.coef_[0], index=X_train.columns).sort_values(ascending=False)

    return TrainingArtifacts(
        model=best_model,
        scaler=scaler,
        training_columns=list(X_train.columns),
        metrics=metrics,
        feature_importance=feature_importance,
    )


def risk_bucket(prob: float) -> str:
    if prob < 0.33:
        return "Low Risk"
    if prob < 0.66:
        return "Medium Risk"
    return "High Risk"


def encode_single_athlete(athlete_df: pd.DataFrame, training_columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    athlete_df = athlete_df.copy()
    athlete_encoded = pd.get_dummies(athlete_df, columns=CATEGORICAL_COLS, drop_first=True)

    for col in training_columns:
        if col not in athlete_encoded.columns:
            athlete_encoded[col] = 0

    athlete_encoded = athlete_encoded[training_columns]
    athlete_encoded.loc[:, NUMERIC_COLS] = scaler.transform(athlete_encoded[NUMERIC_COLS])
    return athlete_encoded


def predict_single_athlete(artifacts: TrainingArtifacts, athlete_data: dict) -> tuple[float, str]:
    athlete_df = pd.DataFrame([athlete_data])
    athlete_encoded = encode_single_athlete(athlete_df, artifacts.training_columns, artifacts.scaler)
    prob = float(artifacts.model.predict_proba(athlete_encoded)[:, 1][0])
    return prob, risk_bucket(prob)


def build_input_form():
    with st.form("single_prediction_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", min_value=10, max_value=60, value=20)
            height = st.number_input("Height (cm)", min_value=100.0, max_value=240.0, value=180.0)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0)
            training_intensity = st.slider("Training Intensity", 1, 10, 5)
            training_hours = st.number_input("Training Hours Per Week", min_value=0.0, max_value=50.0, value=8.0)
            recovery_days = st.number_input("Recovery Days Per Week", min_value=0.0, max_value=7.0, value=3.0)
            match_count = st.number_input("Match Count Per Week", min_value=0.0, max_value=14.0, value=1.0)

        with right:
            rest_days = st.number_input("Rest Between Events (days)", min_value=0.0, max_value=14.0, value=2.0)
            fatigue = st.slider("Fatigue Score", 1, 10, 3)
            performance = st.slider("Performance Score", 0, 100, 85)
            team_contribution = st.slider("Team Contribution Score", 0, 100, 80)
            load_balance = st.slider("Load Balance Score", 0, 100, 80)
            acl_risk = st.slider("ACL Risk Score", 0, 100, 50)
            gender = st.selectbox("Gender", DEFAULT_GENDERS, index=0)
            position = st.selectbox("Position", DEFAULT_POSITIONS, index=0)

        submitted = st.form_submit_button("Predict injury risk", use_container_width=True)

    athlete_data = {
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
        "Position": position,
    }
    return submitted, athlete_data


st.title("🏃 Athlete Injury Risk Assessment")
st.caption("A cleaned-up Streamlit version of your notebook with upload → train → predict workflow.")

tab1, tab2, tab3 = st.tabs(["Overview", "Train Model", "Single Prediction"])

with tab1:
    st.subheader("Project summary")
    st.write(
        """
        This app takes the core workflow from the notebook and turns it into a cleaner project:
        - upload a dataset
        - preprocess numeric and categorical features
        - train a balanced logistic regression model with grid search
        - predict injury risk for an individual athlete
        """
    )

    st.subheader("Expected dataset columns")
    st.code("\n".join(REQUIRED_COLUMNS), language="text")

    st.info(
        "Tip: this version avoids committing a `.pkl` model file to GitHub. "
        "Instead, train the model from your dataset inside the app, then optionally download artifacts."
    )

with tab2:
    st.subheader("Upload dataset and train")
    uploaded_file = st.file_uploader("Upload your athlete injury CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head(), use_container_width=True)

        missing_cols = validate_dataset(df)
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            if st.button("Train model", type="primary", use_container_width=True):
                with st.spinner("Training model..."):
                    artifacts = train_model(df)
                    st.session_state["artifacts"] = artifacts

                st.success("Model trained successfully.")

    artifacts = st.session_state.get("artifacts")
    if artifacts:
        st.subheader("Evaluation metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{artifacts.metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{artifacts.metrics['precision']:.3f}")
        m3.metric("Recall", f"{artifacts.metrics['recall']:.3f}")
        m4.metric("F1 Score", f"{artifacts.metrics['f1']:.3f}")

        a1, a2, a3 = st.columns(3)
        a1.metric("Best C", f"{artifacts.metrics['best_c']}")
        a2.metric("Train Accuracy", f"{artifacts.metrics['train_accuracy']:.3f}")
        a3.metric("Test Accuracy", f"{artifacts.metrics['test_accuracy']:.3f}")

        st.subheader("Confusion matrix")
        cm = artifacts.metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        st.dataframe(cm_df, use_container_width=False)

        st.subheader("Top positive feature signals")
        st.dataframe(
            artifacts.feature_importance.reset_index().rename(columns={"index": "Feature", 0: "Coefficient"}),
            use_container_width=True,
            hide_index=True,
        )

        buffer = io.BytesIO()
        joblib.dump(
            {
                "model": artifacts.model,
                "scaler": artifacts.scaler,
                "training_columns": artifacts.training_columns,
                "numeric_cols": NUMERIC_COLS,
            },
            buffer,
        )
        st.download_button(
            "Download trained artifacts (.joblib)",
            data=buffer.getvalue(),
            file_name="injury_risk_artifacts.joblib",
            mime="application/octet-stream",
            use_container_width=True,
        )

with tab3:
    st.subheader("Predict one athlete")
    artifacts = st.session_state.get("artifacts")

    if not artifacts:
        st.warning("Train a model in the 'Train Model' tab first.")
    else:
        submitted, athlete_data = build_input_form()
        if submitted:
            prob, risk = predict_single_athlete(artifacts, athlete_data)

            st.metric("Predicted Injury Probability", f"{prob:.1%}")
            if risk == "High Risk":
                st.error(f"Risk level: {risk}")
            elif risk == "Medium Risk":
                st.warning(f"Risk level: {risk}")
            else:
                st.success(f"Risk level: {risk}")

            st.write("Athlete data used for prediction:")
            st.json(athlete_data)
