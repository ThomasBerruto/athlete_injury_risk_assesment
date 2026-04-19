import io
import os
from dataclasses import dataclass
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Athlete Risk Assessment", page_icon="📊", layout="wide")

# ----------------------------
# Shared configuration
# ----------------------------
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
RAW_REQUIRED_COLUMNS = NUMERIC_COLS + CATEGORICAL_COLS
REQUIRED_COLUMNS = DROP_COLS + CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]
MODEL_PATHS = ["injury_model.pkl", "/mnt/data/injury_model.pkl", "/mnt/data/injury_model(1).pkl"]
DEFAULT_GENDERS = ["Male", "Female", "Other"]
DEFAULT_POSITIONS = ["Forward", "Guard", "Center", "Midfielder", "Defender", "Goalkeeper", "Pitcher", "Catcher"]
EXPECTED_MODEL_COLUMNS = NUMERIC_COLS + ["Gender_Male", "Position_Forward", "Position_Guard"]


@dataclass
class TrainingArtifacts:
    model: object
    scaler: StandardScaler
    training_columns: list[str]
    metrics: dict
    feature_importance: pd.Series


def risk_bucket(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def risk_explanation(prob: float) -> str:
    band = risk_bucket(prob)
    if band == "Low":
        return "Low risk: below 33%. This indicates the model is currently estimating a lower relative likelihood of injury based on the available inputs."
    if band == "Medium":
        return "Medium risk: from 33% up to 65.9%. This indicates a moderate estimated likelihood of injury and may suggest closer monitoring."
    return "High risk: 66% or above. This indicates the model is estimating a comparatively higher likelihood of injury based on the available inputs."


def validate_dataset(df: pd.DataFrame) -> list[str]:
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


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


def load_pkl_model() -> tuple[Optional[object], Optional[str], Optional[str]]:
    for path in MODEL_PATHS:
        if os.path.exists(path):
            try:
                loaded = joblib.load(path)
                if isinstance(loaded, dict):
                    # Supports full artifact bundle if present.
                    model = loaded.get("model", loaded)
                else:
                    model = loaded
                return model, path, None
            except Exception as e:
                return None, path, str(e)
    return None, None, None


@st.cache_data(show_spinner=False)
def fit_scaler_from_training_csv(df: pd.DataFrame):
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")
    scaler = StandardScaler()
    scaler.fit(df[NUMERIC_COLS])
    return scaler


def align_encoded_columns(df_encoded: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    aligned = df_encoded.copy()
    for col in target_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[target_columns]
    return aligned


def encode_for_trained_model(athlete_df: pd.DataFrame, training_columns: list[str], scaler: StandardScaler) -> pd.DataFrame:
    athlete_encoded = pd.get_dummies(athlete_df.copy(), columns=CATEGORICAL_COLS, drop_first=True)
    athlete_encoded = align_encoded_columns(athlete_encoded, training_columns)
    athlete_encoded.loc[:, NUMERIC_COLS] = scaler.transform(athlete_encoded[NUMERIC_COLS])
    return athlete_encoded


def encode_for_pkl_model(raw_df: pd.DataFrame, scaler: Optional[StandardScaler], feature_names: list[str]) -> pd.DataFrame:
    work = raw_df.copy()
    work["Gender"] = work["Gender"].astype(str).str.strip().str.title()
    work["Position"] = work["Position"].astype(str).str.strip().str.title()
    encoded = pd.get_dummies(work, columns=CATEGORICAL_COLS, drop_first=True)
    encoded = align_encoded_columns(encoded, feature_names)
    numeric_in_frame = [col for col in NUMERIC_COLS if col in encoded.columns]
    if scaler is not None and numeric_in_frame:
        encoded.loc[:, numeric_in_frame] = scaler.transform(encoded[numeric_in_frame])
    return encoded


def get_template_prediction_df() -> pd.DataFrame:
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
        "Position": "Forward",
    }])


def build_input_form(form_key: str):
    with st.form(form_key):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=10, max_value=60, value=20, key=f"{form_key}_age")
            height = st.number_input("Height (cm)", min_value=120, max_value=230, value=180, key=f"{form_key}_height")
            weight = st.number_input("Weight (kg)", min_value=35, max_value=180, value=75, key=f"{form_key}_weight")
            training_intensity = st.slider("Training Intensity", 1, 10, 5, key=f"{form_key}_intensity")
            training_hours = st.number_input("Training Hours / Week", min_value=0, max_value=40, value=8, key=f"{form_key}_hours")
        with c2:
            recovery_days = st.number_input("Recovery Days / Week", min_value=0, max_value=7, value=3, key=f"{form_key}_recovery")
            match_count = st.number_input("Match Count / Week", min_value=0, max_value=14, value=1, key=f"{form_key}_matches")
            rest_days = st.number_input("Rest Between Events (Days)", min_value=0, max_value=14, value=2, key=f"{form_key}_rest")
            fatigue = st.slider("Fatigue Score", 1, 10, 3, key=f"{form_key}_fatigue")
            performance = st.slider("Performance Score", 0, 100, 85, key=f"{form_key}_performance")
        with c3:
            team_contribution = st.slider("Team Contribution Score", 0, 100, 80, key=f"{form_key}_team")
            load_balance = st.slider("Load Balance Score", 0, 100, 80, key=f"{form_key}_load")
            acl_risk = st.slider("ACL Risk Score", 0, 100, 50, key=f"{form_key}_acl")
            gender = st.selectbox("Gender", DEFAULT_GENDERS, index=0, key=f"{form_key}_gender")
            position = st.selectbox("Position", DEFAULT_POSITIONS, index=0, key=f"{form_key}_position")
        submitted = st.form_submit_button("Generate prediction", use_container_width=True)

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


# ----------------------------
# App layout
# ----------------------------
st.title("Athlete Risk Assessment")
st.caption("Interactive risk scoring using either a newly trained model or a previously saved `.pkl` model.")

pkl_model, pkl_path, pkl_error = load_pkl_model()
pkl_feature_names = list(getattr(pkl_model, "feature_names_in_", EXPECTED_MODEL_COLUMNS)) if pkl_model is not None else EXPECTED_MODEL_COLUMNS

with st.sidebar:
    st.subheader("Model options")
    if pkl_model is not None:
        st.success(f"Pretrained model found: `{os.path.basename(pkl_path)}`")
        st.caption("This option pulls predictions from an already trained data model saved as a `.pkl` file.")
    elif pkl_error:
        st.error(f"A `.pkl` file was found but could not be loaded: {pkl_error}")
    else:
        st.info("No local `.pkl` model was detected yet.")

    st.divider()
    st.subheader("Optional training CSV for `.pkl` mode")
    training_csv_for_pkl = st.file_uploader("Upload original training CSV", type=["csv"], key="pkl_training_csv")
    allow_unscaled_pkl = st.checkbox("Allow `.pkl` predictions without training CSV", value=False)
    st.caption("Uploading the original training dataset allows the app to rebuild the numeric scaler used during training. This generally produces more reliable `.pkl` predictions.")

pkl_scaler = None
pkl_scaler_ready = False
if training_csv_for_pkl is not None:
    try:
        pkl_train_df = pd.read_csv(training_csv_for_pkl)
        pkl_scaler = fit_scaler_from_training_csv(pkl_train_df)
        pkl_scaler_ready = True
    except Exception as e:
        st.error(f"Could not rebuild scaler from training CSV: {e}")


overview_tab, train_tab, compare_tab, pkl_tab = st.tabs([
    "Overview",
    "Train New Model",
    "Compare Modes",
    "Use Saved .pkl Model",
])

with overview_tab:
    st.subheader("How the app works")
    st.write(
        "This app supports two different workflows. You can either train a new model from a dataset inside the app, or load predictions from an already saved `.pkl` model."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Train New Model**
            - Upload a dataset
            - Apply preprocessing to numeric and categorical inputs
            - Train a logistic regression classifier
            - Review performance metrics
            - Use the trained model for live prediction
            """
        )
    with c2:
        st.markdown(
            """
            **Use Saved `.pkl` Model**
            - Load an already trained model file
            - Optionally rebuild the scaler from the original training CSV
            - Generate single-athlete or batch predictions
            - Compare the pretrained workflow with an in-app trained workflow
            """
        )

    st.subheader("Risk category thresholds")
    threshold_df = pd.DataFrame(
        {
            "Risk Category": ["Low", "Medium", "High"],
            "Probability Range": ["Below 33%", "33% to 65.9%", "66% and above"],
            "Interpretation": [
                "Lower relative likelihood based on the provided inputs.",
                "Moderate estimated likelihood and may warrant closer attention.",
                "Higher estimated likelihood based on the available inputs.",
            ],
        }
    )
    st.dataframe(threshold_df, use_container_width=True, hide_index=True)
    st.caption("These categories are presentation thresholds used by the app. They are not clinical cutoffs.")

with train_tab:
    st.subheader("Train a model inside the app")
    uploaded_file = st.file_uploader("Upload training dataset", type=["csv"], key="train_dataset")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)
        missing_cols = validate_dataset(df)
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            if st.button("Train model", type="primary", use_container_width=True):
                with st.spinner("Training model..."):
                    artifacts = train_model(df)
                    st.session_state["artifacts"] = artifacts
                st.success("Training complete.")

    artifacts = st.session_state.get("artifacts")
    if artifacts:
        st.subheader("Model evaluation")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{artifacts.metrics['accuracy']:.3f}")
        m2.metric("Precision", f"{artifacts.metrics['precision']:.3f}")
        m3.metric("Recall", f"{artifacts.metrics['recall']:.3f}")
        m4.metric("F1 Score", f"{artifacts.metrics['f1']:.3f}")

        a1, a2, a3 = st.columns(3)
        a1.metric("Best C", f"{artifacts.metrics['best_c']}")
        a2.metric("Train Accuracy", f"{artifacts.metrics['train_accuracy']:.3f}")
        a3.metric("Test Accuracy", f"{artifacts.metrics['test_accuracy']:.3f}")

        cm = artifacts.metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        st.dataframe(cm_df, use_container_width=False)

        st.subheader("Feature coefficients")
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

with compare_tab:
    st.subheader("Compare prediction sources")
    st.write("Use the same athlete inputs to compare a newly trained model against the saved `.pkl` model.")
    submitted, athlete_data = build_input_form("compare_form")

    if submitted:
        athlete_df = pd.DataFrame([athlete_data])
        left, right = st.columns(2)

        with left:
            st.markdown("**In-app trained model**")
            artifacts = st.session_state.get("artifacts")
            if artifacts is None:
                st.info("No in-app trained model is available yet. Train one in the previous tab to compare results.")
            else:
                try:
                    encoded = encode_for_trained_model(athlete_df, artifacts.training_columns, artifacts.scaler)
                    prob = float(artifacts.model.predict_proba(encoded)[:, 1][0])
                    st.metric("Probability", f"{prob:.1%}")
                    st.metric("Risk Category", risk_bucket(prob))
                    st.caption(risk_explanation(prob))
                except Exception as e:
                    st.error(f"Could not score with the in-app trained model: {e}")

        with right:
            st.markdown("**Saved `.pkl` model**")
            if pkl_model is None:
                st.info("No saved `.pkl` model is currently available.")
            elif not pkl_scaler_ready and not allow_unscaled_pkl:
                st.warning("Upload the original training CSV in the sidebar or enable unscaled `.pkl` predictions to use this option.")
            else:
                try:
                    encoded = encode_for_pkl_model(athlete_df, pkl_scaler if pkl_scaler_ready else None, pkl_feature_names)
                    prob = float(pkl_model.predict_proba(encoded)[:, 1][0])
                    st.metric("Probability", f"{prob:.1%}")
                    st.metric("Risk Category", risk_bucket(prob))
                    st.caption("This result is being generated from an already trained data model loaded from the `.pkl` file.")
                    st.caption(risk_explanation(prob))
                    if not pkl_scaler_ready:
                        st.caption("Fallback mode is active, so numeric inputs are not being transformed with the original scaler.")
                except Exception as e:
                    st.error(f"Could not score with the saved `.pkl` model: {e}")

with pkl_tab:
    st.subheader("Use the saved `.pkl` model")
    if pkl_model is None:
        st.warning("No saved `.pkl` model is available. Add `injury_model.pkl` to the project root to enable this section.")
    else:
        st.success("A saved `.pkl` model is active.")
        st.write(
            "This section pulls predictions from an already trained data model. When the original training CSV is also provided, the app can rebuild the scaler and more closely match the original training workflow."
        )
        st.write(f"Loaded file: `{os.path.basename(pkl_path)}`")
        st.write(f"Model type: `{type(pkl_model).__name__}`")

        status_cols = st.columns(3)
        status_cols[0].metric("Feature Count", len(pkl_feature_names))
        status_cols[1].metric("Scaler Rebuilt", "Yes" if pkl_scaler_ready else "No")
        status_cols[2].metric("Fallback Allowed", "Yes" if allow_unscaled_pkl else "No")

        st.subheader("Single-athlete prediction")
        submitted, athlete_data = build_input_form("pkl_form")
        can_predict = pkl_scaler_ready or allow_unscaled_pkl
        if submitted:
            if not can_predict:
                st.warning("Upload the original training CSV in the sidebar or enable fallback mode to continue.")
            else:
                try:
                    athlete_df = pd.DataFrame([athlete_data])
                    features = encode_for_pkl_model(athlete_df, pkl_scaler if pkl_scaler_ready else None, pkl_feature_names)
                    prob = float(pkl_model.predict_proba(features)[:, 1][0])
                    pred = int(pkl_model.predict(features)[0])

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Class", "Injured" if pred == 1 else "Not Injured")
                    m2.metric("Injury Probability", f"{prob:.1%}")
                    m3.metric("Risk Category", risk_bucket(prob))
                    st.info("This prediction is being generated from an already trained data model loaded from the `.pkl` file.")
                    st.caption(risk_explanation(prob))
                    st.dataframe(features, use_container_width=True)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        st.subheader("Batch prediction from CSV")
        st.code(", ".join(RAW_REQUIRED_COLUMNS), language="text")
        template_df = get_template_prediction_df()
        st.download_button(
            "Download example CSV template",
            template_df.to_csv(index=False).encode("utf-8"),
            file_name="athlete_prediction_template.csv",
            mime="text/csv",
        )

        batch_file = st.file_uploader("Upload batch prediction CSV", type=["csv"], key="pkl_batch_csv")
        if batch_file is not None:
            if not can_predict:
                st.warning("Upload the original training CSV in the sidebar or enable fallback mode to run batch predictions.")
            else:
                try:
                    batch_df = pd.read_csv(batch_file)
                    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in batch_df.columns]
                    if missing:
                        st.error(f"Input CSV is missing columns: {', '.join(missing)}")
                    else:
                        features = encode_for_pkl_model(batch_df, pkl_scaler if pkl_scaler_ready else None, pkl_feature_names)
                        probs = pkl_model.predict_proba(features)[:, 1]
                        preds = pkl_model.predict(features)

                        results = batch_df.copy()
                        results["Predicted_Class"] = np.where(preds == 1, "Injured", "Not Injured")
                        results["Injury_Probability"] = probs
                        results["Risk_Category"] = [risk_bucket(p) for p in probs]

                        st.dataframe(results, use_container_width=True)
                        st.info("These batch predictions are being generated from an already trained data model loaded from the `.pkl` file.")
                        st.download_button(
                            "Download predictions CSV",
                            results.to_csv(index=False).encode("utf-8"),
                            file_name="athlete_injury_predictions.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

        with st.expander("Saved model details"):
            st.write("Expected input feature names for the saved model:")
            st.json(pkl_feature_names)
            st.write(
                "Generic note: the `.pkl` file stores a model that has already been trained on prior data. The app uses that existing model directly instead of retraining from scratch in this section."
            )
