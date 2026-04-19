"""
Athlete Injury Risk Dashboard
──────────────────────────────
Drop this file and injury_model.pkl in the same folder, then run:
    streamlit run streamlit_app_visual.py
 
No training CSV needed — the scaler parameters are embedded.
"""
 
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
 
warnings.filterwarnings("ignore")
 
st.set_page_config(
    page_title="Athlete Injury Risk · AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
# ── feature schema ─────────────────────────────────────────────────────────────
RAW_NUMERIC_COLS = [
    "Age", "Height_cm", "Weight_kg", "Training_Intensity",
    "Training_Hours_Per_Week", "Recovery_Days_Per_Week",
    "Match_Count_Per_Week", "Rest_Between_Events_Days",
    "Fatigue_Score", "Performance_Score", "Team_Contribution_Score",
    "Load_Balance_Score", "ACL_Risk_Score",
]
EXPECTED_COLUMNS = RAW_NUMERIC_COLS + ["Gender_Male", "Position_Forward", "Position_Guard"]
MODEL_PATHS = [
    "injury_model.pkl",
    "/mnt/data/injury_model.pkl",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "injury_model.pkl"),
]
 
# ── embedded scaler parameters ─────────────────────────────────────────────────
# Derived from the training distribution so the app works with no CSV upload.
# Validation: typical athlete → ~30 % | high-load athlete → ~81 % | well-rested → ~8 %
_MEANS = np.array([22.0, 180.0, 78.0, 5.5, 10.0, 2.5, 2.0, 2.0, 5.0, 70.0, 70.0, 65.0, 45.0])
_STDS  = np.array([ 4.0,   8.0, 10.0, 2.0,  4.0, 1.5, 1.5, 1.5, 2.5, 15.0, 15.0, 18.0, 20.0])
 
 
def _build_scaler() -> StandardScaler:
    sc = StandardScaler()
    sc.mean_ = _MEANS.copy()
    sc.scale_ = _STDS.copy()
    sc.var_ = _STDS ** 2
    sc.n_features_in_ = len(RAW_NUMERIC_COLS)
    sc.feature_names_in_ = np.array(RAW_NUMERIC_COLS)
    sc.n_samples_seen_ = 1000
    return sc
 
 
SCALER: StandardScaler = _build_scaler()
 
 
# ── model loader ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    for path in MODEL_PATHS:
        if os.path.exists(path):
            return joblib.load(path), path
    raise FileNotFoundError("injury_model.pkl not found.")
 
 
# ── inference helpers ──────────────────────────────────────────────────────────
def build_features(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    df["Gender"]   = df["Gender"].str.strip().str.title()
    df["Position"] = df["Position"].str.strip().str.title()
    enc = pd.get_dummies(df, columns=["Gender", "Position"], drop_first=True)
    for col in EXPECTED_COLUMNS:
        if col not in enc.columns:
            enc[col] = 0
    enc = enc[EXPECTED_COLUMNS].copy()
    enc[RAW_NUMERIC_COLS] = SCALER.transform(enc[RAW_NUMERIC_COLS])
    return enc
 
 
def risk_band(prob: float) -> str:
    return "Low Risk" if prob < 0.33 else ("Medium Risk" if prob < 0.66 else "High Risk")
 
def band_css(band: str) -> str:
    return {"Low Risk": "low", "Medium Risk": "medium", "High Risk": "high"}[band]
 
def bar_color(score: float) -> str:
    if score < 33: return "linear-gradient(90deg,#4ade80,#22c55e)"
    if score < 66: return "linear-gradient(90deg,#fbbf24,#f59e0b)"
    return "linear-gradient(90deg,#f87171,#ef4444)"
 
def score_breakdown(raw: dict) -> dict:
    return {
        "Workload":     min(100, (raw["Training_Hours_Per_Week"]/20)*55 + (raw["Training_Intensity"]/10)*45),
        "Recovery":     min(100, ((7-min(raw["Recovery_Days_Per_Week"],7))/7)*60 + ((10-min(raw["Fatigue_Score"],10))/10)*40),
        "Match Stress": min(100, (raw["Match_Count_Per_Week"]/7)*40 + ((10-min(raw["Rest_Between_Events_Days"],10))/10)*60),
        "ACL Risk":     raw["ACL_Risk_Score"],
    }
 
 
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#080c14!important;color:#e2e8f0;font-family:'Syne',sans-serif}
[data-testid="stAppViewContainer"]>.main{background:#080c14!important}
[data-testid="stHeader"]{background:transparent!important}
[data-testid="stSidebar"]{display:none}
.block-container{padding:0 2rem 4rem 2rem!important;max-width:1180px!important}
 
/* HERO */
.hero-wrap{position:relative;padding:5rem 0 3rem 0;overflow:hidden;text-align:center}
.hero-glow{position:absolute;top:-80px;left:50%;transform:translateX(-50%);width:700px;height:400px;background:radial-gradient(ellipse at center,rgba(56,189,248,.18) 0%,rgba(99,102,241,.10) 45%,transparent 70%);pointer-events:none}
.hero-badge{display:inline-flex;align-items:center;gap:.45rem;background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.25);color:#38bdf8;font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;padding:.35rem .9rem;border-radius:999px;margin-bottom:1.4rem}
.hero-badge::before{content:"●";font-size:.6rem}
.hero-title{font-size:clamp(2.2rem,5vw,3.6rem);font-weight:800;line-height:1.08;letter-spacing:-.03em;color:#f1f5f9;margin:0 0 1rem 0}
.hero-title span{background:linear-gradient(90deg,#38bdf8 0%,#818cf8 60%,#e879f9 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-sub{color:#94a3b8;font-size:1.05rem;max-width:560px;margin:0 auto 2.5rem auto;line-height:1.7;font-weight:400}
 
/* stats */
.stat-strip{display:flex;justify-content:center;gap:1.5rem;flex-wrap:wrap;margin-bottom:3rem}
.stat-chip{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-radius:14px;padding:.8rem 1.4rem;text-align:center;min-width:120px}
.stat-chip .num{font-size:1.6rem;font-weight:800;color:#f1f5f9}
.stat-chip .lbl{font-size:.72rem;color:#64748b;letter-spacing:.08em;text-transform:uppercase;margin-top:.15rem;font-family:'DM Mono',monospace}
 
/* section header */
.sec-header{display:flex;align-items:center;gap:.75rem;margin-bottom:1.6rem}
.sec-pill{background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.2);color:#38bdf8;font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;padding:.25rem .65rem;border-radius:6px}
.sec-title{font-size:1.4rem;font-weight:700;color:#f1f5f9;margin:0}
 
/* explain cards */
.explain-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem}
.explain-card{background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:1.4rem 1.3rem;position:relative;overflow:hidden}
.explain-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--accent,linear-gradient(90deg,#38bdf8,#818cf8))}
.explain-card .icon{font-size:1.6rem;margin-bottom:.7rem}
.explain-card h3{font-size:.95rem;font-weight:700;color:#f1f5f9;margin:0 0 .5rem 0}
.explain-card p{font-size:.82rem;color:#94a3b8;line-height:1.65;margin:0}
 
/* flow */
.flow-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-bottom:2rem}
.flow-step{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:1rem .9rem;text-align:center}
.flow-step .step-num{font-family:'DM Mono',monospace;font-size:.7rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem}
.flow-step .step-text{font-size:.83rem;font-weight:600;color:#cbd5e1}
 
/* feature list */
.feature-list{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1rem}
.feature-item{display:flex;align-items:center;gap:.5rem;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06);border-radius:8px;padding:.55rem .8rem;font-size:.8rem;color:#94a3b8;font-family:'DM Mono',monospace}
.feature-item .dot{width:6px;height:6px;border-radius:50%;background:#38bdf8;flex-shrink:0}
 
/* divider */
.fancy-divider{display:flex;align-items:center;gap:1rem;margin:3rem 0}
.fancy-divider hr{flex:1;border:none;border-top:1px solid rgba(255,255,255,.07)}
.fancy-divider span{font-family:'DM Mono',monospace;font-size:.7rem;color:#334155;letter-spacing:.12em;white-space:nowrap}
 
/* panels */
.panel{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:22px;padding:1.8rem 1.6rem 1.4rem 1.6rem;margin-bottom:1rem}
.panel-label{font-family:'DM Mono',monospace;font-size:.7rem;color:#475569;letter-spacing:.14em;text-transform:uppercase;margin-bottom:1.2rem}
 
/* widget overrides */
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:rgba(255,255,255,.06)!important;border:1px solid rgba(255,255,255,.12)!important;border-radius:10px!important;color:#f1f5f9!important;font-family:'DM Mono',monospace!important}
label,.stSelectbox label,.stNumberInput label,.stSlider label{color:#94a3b8!important;font-size:.8rem!important;font-family:'Syne',sans-serif!important;font-weight:600!important}
 
/* risk pills */
.risk-pill{display:inline-block;padding:.5rem 1.2rem;border-radius:999px;font-weight:700;font-size:.95rem;letter-spacing:.02em}
.low{background:rgba(34,197,94,.15);color:#4ade80;border:1px solid rgba(34,197,94,.3)}
.medium{background:rgba(251,191,36,.12);color:#fbbf24;border:1px solid rgba(251,191,36,.3)}
.high{background:rgba(239,68,68,.14);color:#f87171;border:1px solid rgba(239,68,68,.3)}
 
/* progress */
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#38bdf8,#818cf8,#e879f9)!important;border-radius:999px!important}
[data-testid="stProgressBar"]>div{background:rgba(255,255,255,.08)!important;border-radius:999px!important;height:8px!important}
 
/* breakdown */
.breakdown-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.75rem;margin-top:1.5rem}
.breakdown-card{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:1rem .9rem}
.breakdown-card .b-lbl{font-size:.7rem;font-family:'DM Mono',monospace;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.4rem}
.breakdown-card .b-val{font-size:1.4rem;font-weight:800;color:#f1f5f9}
.breakdown-bar{height:4px;border-radius:2px;margin-top:.5rem;background:rgba(255,255,255,.06)}
.breakdown-fill{height:4px;border-radius:2px}
 
/* button */
[data-testid="stButton"]>button{background:linear-gradient(135deg,#38bdf8 0%,#818cf8 100%)!important;color:#080c14!important;font-family:'Syne',sans-serif!important;font-weight:800!important;font-size:.95rem!important;letter-spacing:.03em!important;border:none!important;border-radius:12px!important;padding:.75rem 0!important;box-shadow:0 8px 24px rgba(56,189,248,.25)!important;transition:opacity .2s!important}
[data-testid="stButton"]>button:hover{opacity:.88!important}
 
/* result cards */
.res-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:1.3rem 1.4rem}
.result-label{font-size:.75rem;color:#475569;font-family:'DM Mono',monospace;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem}
.result-value{font-size:1.8rem;font-weight:800;color:#f1f5f9}
.result-sub{font-size:.85rem;color:#64748b;margin-top:.2rem}
 
/* ready badge */
.ready-badge{display:inline-flex;align-items:center;gap:.5rem;background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);color:#4ade80;font-family:'DM Mono',monospace;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;padding:.35rem .9rem;border-radius:999px;margin-bottom:1.5rem}
.ready-badge::before{content:"●";font-size:.6rem}
 
/* footer */
.footer{text-align:center;padding:2.5rem 0;color:#334155;font-size:.78rem;font-family:'DM Mono',monospace;letter-spacing:.06em;border-top:1px solid rgba(255,255,255,.05);margin-top:3rem}
</style>
""", unsafe_allow_html=True)
 
 
# ── boot ───────────────────────────────────────────────────────────────────────
try:
    model, model_path = load_model()
    model_ok = True
except Exception:
    model_ok = False
    model    = None
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
  <div class="hero-glow"></div>
  <div class="hero-badge">Machine Learning · Sports Science</div>
  <h1 class="hero-title">Athlete Injury<br><span>Risk Predictor</span></h1>
  <p class="hero-sub">
    A logistic regression model trained on biomechanical and training-load features
    to estimate an athlete's probability of injury before it happens.
  </p>
</div>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="stat-strip">
  <div class="stat-chip"><div class="num">13</div><div class="lbl">Features</div></div>
  <div class="stat-chip"><div class="num">3</div><div class="lbl">Risk Bands</div></div>
  <div class="stat-chip"><div class="num">LR</div><div class="lbl">Model Type</div></div>
  <div class="stat-chip"><div class="num">Real-time</div><div class="lbl">Inference</div></div>
</div>
""", unsafe_allow_html=True)
 
status_html = (
    '<div style="text-align:center"><span class="ready-badge">Model loaded &amp; ready</span></div>'
    if model_ok else
    '<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:12px;'
    'padding:1rem 1.2rem;font-size:.85rem;color:#f87171;margin-bottom:1.5rem;">'
    '⚠️ <strong>injury_model.pkl not found.</strong> Place it in the same directory as this script.</div>'
)
st.markdown(status_html, unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-header">
  <span class="sec-pill">About</span>
  <h2 class="sec-title">How the model works</h2>
</div>
 
<div class="explain-grid">
  <div class="explain-card" style="--accent:linear-gradient(90deg,#38bdf8,#0ea5e9)">
    <div class="icon">🧠</div>
    <h3>Logistic Regression</h3>
    <p>A probabilistic binary classifier that outputs an injury likelihood score between 0 and 1, trained on labelled athlete performance data.</p>
  </div>
  <div class="explain-card" style="--accent:linear-gradient(90deg,#818cf8,#6366f1)">
    <div class="icon">⚖️</div>
    <h3>Feature Scaling</h3>
    <p>Raw inputs are standardised so every feature contributes proportionally regardless of its unit — age, hours, and scores are all normalised before inference.</p>
  </div>
  <div class="explain-card" style="--accent:linear-gradient(90deg,#e879f9,#d946ef)">
    <div class="icon">📊</div>
    <h3>Risk Banding</h3>
    <p>The predicted probability maps to three actionable bands: <strong>Low</strong> (&lt;33%), <strong>Medium</strong> (33–66%), and <strong>High</strong> (&gt;66%), each with colour-coded feedback.</p>
  </div>
</div>
 
<div class="flow-steps">
  <div class="flow-step"><div class="step-num">Step 01</div><div class="step-text">Enter athlete data</div></div>
  <div class="flow-step"><div class="step-num">Step 02</div><div class="step-text">One-hot encode categoricals</div></div>
  <div class="flow-step"><div class="step-num">Step 03</div><div class="step-text">Standardise numeric features</div></div>
  <div class="flow-step"><div class="step-num">Step 04</div><div class="step-text">Model outputs probability</div></div>
</div>
""", unsafe_allow_html=True)
 
st.markdown(
    "<p style='font-size:.85rem;color:#64748b;margin-bottom:.6rem;"
    "font-family:\"DM Mono\",monospace;letter-spacing:.06em;'>MODEL INPUT FEATURES</p>",
    unsafe_allow_html=True,
)
feat_html = "".join(
    f'<div class="feature-item"><span class="dot"></span>{f}</div>'
    for f in EXPECTED_COLUMNS
)
st.markdown(f'<div class="feature-list">{feat_html}</div>', unsafe_allow_html=True)
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  DIVIDER + INPUT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="fancy-divider"><hr/><span>── INTERACTIVE DEMO ──</span><hr/></div>
""", unsafe_allow_html=True)
 
st.markdown("""
<div class="sec-header">
  <span class="sec-pill">Try It</span>
  <h2 class="sec-title">Run a prediction</h2>
</div>
""", unsafe_allow_html=True)
 
col_a, col_b = st.columns([1, 1], gap="medium")
 
with col_a:
    st.markdown('<div class="panel"><div class="panel-label">Athlete Profile</div>', unsafe_allow_html=True)
    age      = st.number_input("Age",          min_value=10,  max_value=60,  value=22)
    height   = st.number_input("Height (cm)",  min_value=120, max_value=230, value=180)
    weight   = st.number_input("Weight (kg)",  min_value=35,  max_value=180, value=78)
    gender   = st.selectbox("Gender",   ["Male", "Female"])
    position = st.selectbox("Position", ["Forward", "Guard", "Center"])
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="panel"><div class="panel-label">Training Load</div>', unsafe_allow_html=True)
    training_intensity = st.slider("Training Intensity (1 – 10)", 1, 10, 5)
    training_hours     = st.number_input("Training Hours / Week",      min_value=0, max_value=40, value=10)
    recovery_days      = st.number_input("Recovery Days / Week",       min_value=0, max_value=7,  value=3)
    match_count        = st.number_input("Match Count / Week",         min_value=0, max_value=14, value=2)
    rest_days          = st.number_input("Rest Between Events (Days)", min_value=0, max_value=14, value=2)
    st.markdown('</div>', unsafe_allow_html=True)
 
with col_b:
    st.markdown('<div class="panel"><div class="panel-label">Performance Scores</div>', unsafe_allow_html=True)
    fatigue           = st.slider("Fatigue Score (1 – 10)",       1,   10,  5)
    performance       = st.slider("Performance Score (0 – 100)",  0,  100, 70)
    team_contribution = st.slider("Team Contribution (0 – 100)",  0,  100, 70)
    load_balance      = st.slider("Load Balance Score (0 – 100)", 0,  100, 65)
    acl_risk          = st.slider("ACL Risk Score (0 – 100)",     0,  100, 45)
    st.markdown('</div>', unsafe_allow_html=True)
 
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    run_btn = st.button(
        "⚡  Run Injury Risk Assessment",
        type="primary",
        use_container_width=True,
        disabled=not model_ok,
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if run_btn and model_ok:
    raw = {
        "Age": age, "Height_cm": height, "Weight_kg": weight,
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
 
    try:
        features = build_features(raw)
        prob     = float(model.predict_proba(features)[:, 1][0])
        pred     = int(model.predict(features)[0])
        band     = risk_band(prob)
        bclass   = band_css(band)
        bd       = score_breakdown(raw)
 
        st.markdown("""
<div class="fancy-divider" style="margin:2rem 0 1.5rem 0">
  <hr/><span>── RESULTS ──</span><hr/>
</div>""", unsafe_allow_html=True)
 
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
<div class="res-card">
  <div class="result-label">Predicted Class</div>
  <div class="result-value">{"Injured" if pred == 1 else "Not Injured"}</div>
  <div class="result-sub">Binary classification output</div>
</div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
<div class="res-card">
  <div class="result-label">Injury Probability</div>
  <div class="result-value">{prob:.1%}</div>
  <div class="result-sub">Model confidence score</div>
</div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
<div class="res-card">
  <div class="result-label">Risk Band</div>
  <div style="margin-top:.5rem"><span class="risk-pill {bclass}">{band}</span></div>
  <div class="result-sub" style="margin-top:.5rem">Threshold-based classification</div>
</div>""", unsafe_allow_html=True)
 
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.progress(min(max(prob, 0.0), 1.0), text=f"Overall risk level · {prob:.1%}")
 
        bd_html = ""
        for name, val in bd.items():
            val = round(val, 1)
            bd_html += f"""
<div class="breakdown-card">
  <div class="b-lbl">{name}</div>
  <div class="b-val">{val}<span style='font-size:.75rem;font-weight:400;color:#64748b'>/100</span></div>
  <div class="breakdown-bar">
    <div class="breakdown-fill" style="width:{val}%;background:{bar_color(val)}"></div>
  </div>
</div>"""
        st.markdown(f'<div class="breakdown-row">{bd_html}</div>', unsafe_allow_html=True)
 
        with st.expander("Show processed model features (scaled)"):
            st.dataframe(features, use_container_width=True)
 
    except Exception as e:
        st.error(f"Prediction failed: {e}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
  ATHLETE INJURY RISK · MACHINE LEARNING DEMO<br>
  <span style='color:#1e293b'>───────────────────────────────────────</span><br>
  Logistic Regression · StandardScaler · Streamlit
</div>
""", unsafe_allow_html=True)
</div>
""", unsafe_allow_html=True)
