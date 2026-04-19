import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Athlete Injury Risk Dashboard", page_icon="🏥", layout="wide")

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


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #111827 24%, #f8fafc 24%, #f8fafc 100%);
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(37,99,235,.95), rgba(14,165,233,.9));
            padding: 1.6rem 1.8rem;
            border-radius: 22px;
            color: white;
            box-shadow: 0 20px 45px rgba(2,6,23,.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.1rem;
            line-height: 1.1;
        }
        .hero p {
            margin: .55rem 0 0 0;
            font-size: 1rem;
            opacity: .96;
        }
        .section-card {
            background: white;
            border: 1px solid rgba(148,163,184,.25);
            border-radius: 20px;
            padding: 1.1rem 1.1rem .8rem 1.1rem;
            box-shadow: 0 12px 30px rgba(15,23,42,.06);
            margin-bottom: 1rem;
        }
        .mini-card {
            background: white;
            border-radius: 18px;
            padding: .95rem 1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,.06);
            border: 1px solid rgba(148,163,184,.20);
        }
        .mini-label {
            font-size: .8rem;
            color: #64748b;
            margin-bottom: .25rem;
        }
        .mini-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
        }
        .risk-pill {
            display: inline-block;
            padding: .35rem .8rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: .9rem;
        }
        .low { background: #dcfce7; color: #166534; }
        .medium { background: #fef3c7; color: #92400e; }
        .high { background: #fee2e2; color: #991b1b; }
        .small-note {
            color: #64748b;
            font-size: .92rem;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(148,163,184,.18);
            padding: .8rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(15,23,42,.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def band_class(label: str) -> str:
    return {"Low Risk": "low", "Medium Risk": "medium", "High Risk": "high"}[label]


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


def score_breakdown(row: pd.Series):
    vals = {
        "Workload": min(100, ((row["Training_Hours_Per_Week"] / 20) * 55) + ((row["Training_Intensity"] / 10) * 45)),
        "Recovery": min(100, ((7 - min(row["Recovery_Days_Per_Week"], 7)) / 7) * 60 + ((10 - min(row["Fatigue_Score"], 10)) / 10) * 40),
        "Stress": min(100, ((row["Match_Count_Per_Week"] / 7) * 40) + ((10 - min(row["Rest_Between_Events_Days"], 10)) / 10) * 60),
        "ACL": row["ACL_Risk_Score"],
    }
    return pd.DataFrame({"Category": list(vals.keys()), "Score": [round(v, 1) for v in vals.values()]})


inject_css()
model, model_path = load_model()
feature_names = list(getattr(model, "feature_names_in_", EXPECTED_MODEL_COLUMNS))
model_cols_ok = feature_names == EXPECTED_MODEL_COLUMNS

st.markdown(
    """
    <div class="hero">
        <h1>Athlete Injury Risk Dashboard</h1>
        <p>Interactive Streamlit app for single-athlete screening, batch prediction, and model-backed injury risk assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.6, 1])
with left:
    st.markdown('<div class="mini-card"><div class="mini-label">Model Source</div><div class="mini-value">injury_model.pkl</div></div>', unsafe_allow_html=True)
with right:
    status = "Ready" if model_cols_ok else "Check feature schema"
    st.markdown(f'<div class="mini-card"><div class="mini-label">Status</div><div class="mini-value">{status}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("App Controls")
    st.caption("Load the model, optionally restore preprocessing, and choose how predictions should run.")
    st.success(f"Loaded model from: {model_path}")
    st.write(f"**Model type:** {type(model).__name__}")
    st.write(f"**Feature count:** {len(feature_names)}")
    if not model_cols_ok:
        st.warning("Saved feature names differ from the expected notebook schema.")
    st.divider()
    st.subheader("Restore preprocessing")
    st.write("Upload the original training CSV to rebuild the scaler used during notebook training.")
    train_file = st.file_uploader("Training CSV", type=["csv"], key="training_csv")
    use_identity = st.checkbox("Use fallback mode without scaler", value=False)
    st.caption("Fallback mode runs, but may not match notebook predictions exactly.")

scaler = None
scaler_ready = False

if train_file is not None:
    try:
        train_df = pd.read_csv(train_file)
        scaler = fit_scaler_from_training_csv(train_df)
        scaler_ready = True
        st.toast("Training CSV loaded successfully.")
    except Exception as e:
        st.error(f"Could not use training CSV: {e}")
elif not use_identity:
    st.info("Upload the training CSV for the best prediction quality, or enable fallback mode in the sidebar.")

single_tab, batch_tab, explain_tab = st.tabs(["Single Athlete", "Batch Prediction", "Project Notes"])

with single_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Single-athlete prediction")
    st.caption("Enter athlete details below, then generate an injury risk prediction.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=10, max_value=60, value=20)
        height = st.number_input("Height (cm)", min_value=120, max_value=230, value=180)
        weight = st.number_input("Weight (kg)", min_value=35, max_value=180, value=75)
        gender = st.selectbox("Gender", ["Male", "Female"])
        position = st.selectbox("Position", ["Forward", "Guard", "Center"])
    with c2:
        training_intensity = st.slider("Training Intensity", 1, 10, 5)
        training_hours = st.number_input("Training Hours / Week", min_value=0, max_value=40, value=8)
        recovery_days = st.number_input("Recovery Days / Week", min_value=0, max_value=7, value=3)
        match_count = st.number_input("Match Count / Week", min_value=0, max_value=14, value=1)
        rest_days = st.number_input("Rest Between Events (Days)", min_value=0, max_value=14, value=2)
    with c3:
        fatigue = st.slider("Fatigue Score", 1, 10, 3)
        performance = st.slider("Performance Score", 0, 100, 85)
        team_contribution = st.slider("Team Contribution Score", 0, 100, 80)
        load_balance = st.slider("Load Balance Score", 0, 100, 80)
        acl_risk = st.slider("ACL Risk Score", 0, 100, 50)

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
        "Position": position,
    }])

    can_predict = scaler_ready or use_identity
    if st.button("Run risk assessment", type="primary", use_container_width=True, disabled=not can_predict):
        try:
            features = build_feature_frame(raw_single, scaler if scaler_ready else None)
            prob = float(model.predict_proba(features)[:, 1][0])
            pred = int(model.predict(features)[0])
            band = risk_band(prob)

            st.markdown("### Prediction Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Class", "Injured" if pred == 1 else "Not Injured")
            m2.metric("Injury Probability", f"{prob:.1%}")
            m3.markdown(
                f'<div class="mini-card"><div class="mini-label">Risk Band</div><div class="risk-pill {band_class(band)}">{band}</div></div>',
                unsafe_allow_html=True,
            )

            st.progress(min(max(prob, 0.0), 1.0), text=f"Risk level: {prob:.1%}")

            b1, b2 = st.columns([1.1, 1])
            with b1:
                st.markdown("#### Athlete profile")
                display_df = raw_single.T.reset_index()
                display_df.columns = ["Variable", "Value"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            with b2:
                st.markdown("#### Risk factor snapshot")
                breakdown = score_breakdown(raw_single.iloc[0]).set_index("Category")
                st.bar_chart(breakdown)

            with st.expander("Show processed model features"):
                st.dataframe(features, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

with batch_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Batch prediction")
    st.caption("Upload a CSV with the raw columns from your notebook dataset.")
    st.code(", ".join(RAW_REQUIRED_COLUMNS), language="text")

    template_df = get_template_df()
    st.download_button(
        "Download example CSV template",
        template_df.to_csv(index=False).encode("utf-8"),
        file_name="athlete_prediction_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    batch_file = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_csv")
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

            top1, top2, top3 = st.columns(3)
            top1.metric("Athletes scored", len(results))
            top2.metric("Average risk", f"{results['Injury_Probability'].mean():.1%}")
            top3.metric("High-risk cases", int((results["Risk_Band"] == "High Risk").sum()))

            st.dataframe(results, use_container_width=True)

            summary = results["Risk_Band"].value_counts().rename_axis("Risk Band").to_frame("Count")
            st.markdown("#### Risk distribution")
            st.bar_chart(summary)

            st.download_button(
                "Download predictions CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="athlete_injury_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

with explain_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("How this app works")
    st.markdown(
        """
        This version is built around your saved `injury_model.pkl` and is designed to look cleaner for a portfolio or Streamlit Cloud deployment.

        **Important note:** the `.pkl` stores the trained logistic regression model, but not the fitted scaler used during notebook preprocessing.

        **Best setup**
        1. Keep `injury_model.pkl` in the repo root.
        2. Keep `streamlit_app.py` in the repo root.
        3. Upload the original training CSV in the sidebar when using the app.

        **Fallback mode**
        - The app can still run without the training CSV.
        - Predictions may be less faithful to the original notebook workflow.
        """
    )
    st.markdown('<p class="small-note">Expected model columns</p>', unsafe_allow_html=True)
    st.json(feature_names)
    st.markdown('</div>', unsafe_allow_html=True)
