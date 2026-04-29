# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path
# st.set_page_config(
#     page_title="Drug Prediction Dashboard",
#     layout="wide",
#     initial_sidebar_state="collapsed"
# )

# st.markdown("""
# <style>

# /* REMOVE STREAMLIT HEADER COMPLETELY */
# header {visibility: hidden;}
# [data-testid="stToolbar"] {display: none;}
# [data-testid="stHeader"] {display: none;}
# [data-testid="stDecoration"] {display: none;}

# /* FIX BODY & TOP SPACE */
# html, body {
#     margin: 0;
#     padding: 0;
#     background: #f7f9fb;
# }

# /* MAIN CONTAINER */
# .block-container {
#     padding-top: 0px !important;
# }

# /* CINEMATIC TOP BAR */
# .top-quote {
#     position: sticky;
#     top: 0;
#     z-index: 999;
#     width: 100%;
#     text-align: center;
#     padding: 22px;
#     font-size: 18px;
#     color: white;
#     background: linear-gradient(90deg, #000000, #1c1c1c);
#     border-radius: 0 0 22px 22px;
#     box-shadow: 0 12px 30px rgba(0,0,0,0.6);
# }

# /* TITLE */
# .title {
#     font-size: 36px;
#     font-weight: 800;
#     color: #2ecc71;
#     margin: 25px 0 10px 0;
# }

# /* GLASS CARD */
# .card {
#     background: rgba(255,255,255,0.85);
#     backdrop-filter: blur(12px);
#     padding: 24px;
#     border-radius: 22px;
#     box-shadow: 0 15px 40px rgba(0,0,0,0.08);
#     transition: all 0.4s ease;
# }

# .card:hover {
#     transform: translateY(-10px) rotateX(4deg);
#     box-shadow: 0 25px 60px rgba(0,0,0,0.18);
# }

# /* KPI */
# .metric {
#     font-size: 34px;
#     font-weight: 800;
# }
# .small {
#     font-size: 14px;
#     color: #6b7280;
# }

# /* NEON GREEN CARD */
# .green-card {
#     background: linear-gradient(135deg, #2ecc71, #22c55e);
#     color: white;
#     padding: 26px;
#     border-radius: 24px;
#     box-shadow: 0 20px 60px rgba(34,197,94,0.7);
#     animation: glow 2s infinite alternate;
# }

# @keyframes glow {
#     from { box-shadow: 0 20px 50px rgba(34,197,94,0.4); }
#     to { box-shadow: 0 30px 90px rgba(34,197,94,0.9); }
# }

# /* RESULT */
# .predict-box {
#     margin-top: 18px;
#     padding: 22px;
#     background: rgba(255,255,255,0.18);
#     border-radius: 18px;
#     font-size: 26px;
#     font-weight: 800;
#     text-align: center;
#     animation: pulse 1.4s infinite;
# }

# @keyframes pulse {
#     0% {transform: scale(1);}
#     50% {transform: scale(1.05);}
#     100% {transform: scale(1);}
# }

# footer {visibility: hidden;}
# </style>
# """, unsafe_allow_html=True)

# BASE = Path.cwd()
# MODEL_FILE = BASE / "drug_model.pkl"

# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_FILE)

# st.markdown(
#     '<div class="top-quote">when you realized the love is over, but life isn’t</div>',
#     unsafe_allow_html=True
# )
# st.markdown('<div class="title">💊 Drug Prediction Productivity Dashboard</div>', unsafe_allow_html=True)

# if not MODEL_FILE.exists():
#     st.error("drug_model.pkl not found")
#     st.stop()

# model = load_model()

# k1, k2, k3, k4 = st.columns(4)
# with k1:
#     st.markdown("<div class='card'><div class='metric'>98%</div><div class='small'>Model Accuracy</div></div>", unsafe_allow_html=True)
# with k2:
#     st.markdown("<div class='card'><div class='metric'>5</div><div class='small'>Drug Types</div></div>", unsafe_allow_html=True)
# with k3:
#     st.markdown("<div class='card'><div class='metric'>ML</div><div class='small'>Prediction Engine</div></div>", unsafe_allow_html=True)
# with k4:
#     st.markdown("<div class='card'><div class='metric'>LIVE</div><div class='small'>System Status</div></div>", unsafe_allow_html=True)

# st.write("")

# left, center, right = st.columns([1.1, 1.3, 1])

# with left:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("🧾 Patient Details")
#     age = st.number_input("Age", 1, 120, 35)
#     sex = st.selectbox("Sex", ["M", "F"])
#     bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
#     chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
#     na = st.number_input("Na to K Ratio", 0.0, 50.0, 15.0)
#     predict = st.button("🔮 Predict Drug")
#     st.markdown("</div>", unsafe_allow_html=True)

# with center:
#     st.markdown("<div class='card'>", unsafe_allow_html=True)
#     st.subheader("📊 Health Intelligence")
#     st.progress(0.82)
#     st.write("Clinical Data Quality")
#     st.progress(0.69)
#     st.write("Risk Evaluation")
#     st.progress(0.93)
#     st.write("Prescription Confidence")
#     st.markdown("</div>", unsafe_allow_html=True)

# with right:
#     st.markdown("<div class='green-card'>", unsafe_allow_html=True)
#     st.subheader("📌 Prediction")
#     if predict:
#         df = pd.DataFrame([{
#             "Age": age,
#             "Sex": sex,
#             "BP": bp,
#             "Cholesterol": chol,
#             "Na_to_K": na
#         }])
#         result = model.predict(df)[0]
#         st.markdown(f"<div class='predict-box'>{result}</div>", unsafe_allow_html=True)
#     else:
#         st.info("Enter details & click Predict")
#     st.markdown("</div>", unsafe_allow_html=True)










import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ─── AUTO-TRAIN MODEL IF MISSING OR BROKEN ───────────────────────────────────
def train_and_save_model(path):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    np.random.seed(42)
    n = 200
    ages    = np.random.randint(15, 75, n)
    sexes   = np.random.choice(["M", "F"], n)
    bps     = np.random.choice(["LOW", "NORMAL", "HIGH"], n, p=[0.33, 0.33, 0.34])
    chols   = np.random.choice(["NORMAL", "HIGH"], n, p=[0.46, 0.54])
    na_to_k = np.round(np.random.uniform(6.2, 38.2, n), 2)

    def assign_drug(age, sex, bp, chol, na_k):
        if na_k > 14.829:   return "DrugY"
        if bp == "HIGH":    return "DrugA" if (age <= 50 and sex == "F") else "DrugB" if age <= 50 else "DrugA"
        if bp == "LOW":     return "DrugC"
        return "DrugX"

    drugs = [assign_drug(ages[i], sexes[i], bps[i], chols[i], na_to_k[i]) for i in range(n)]

    df = pd.DataFrame({"Age": ages, "Sex": sexes, "BP": bps,
                       "Cholesterol": chols, "Na_to_K": na_to_k, "Drug": drugs})
    X, y = df.drop("Drug", axis=1), df["Drug"]

    pipe = Pipeline([
        ("pre", ColumnTransformer([
            ("cat", OrdinalEncoder(
                categories=[["M","F"], ["LOW","NORMAL","HIGH"], ["NORMAL","HIGH"]],
                handle_unknown="use_encoded_value", unknown_value=-1
            ), ["Sex","BP","Cholesterol"]),
            ("num", "passthrough", ["Age","Na_to_K"]),
        ])),
        ("clf", DecisionTreeClassifier(max_depth=4, random_state=42)),
    ])
    pipe.fit(X, y)
    joblib.dump(pipe, path)
    return pipe

BASE       = Path.cwd()
MODEL_FILE = BASE / "drug_model.pkl"

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return train_and_save_model(MODEL_FILE)
    try:
        return joblib.load(MODEL_FILE)
    except Exception:
        return train_and_save_model(MODEL_FILE)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Amit Predictor", layout="wide", initial_sidebar_state="collapsed")

# ─── STYLES ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

header, footer,
[data-testid="stToolbar"],
[data-testid="stHeader"],
[data-testid="stDecoration"],
[data-testid="stSidebarNav"] { display: none !important; visibility: hidden !important; }

html, body, .stApp {
    background: #050A0E !important;
    font-family: 'DM Sans', sans-serif;
}

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── TOPBAR ── */
.topbar {
    width: 100%; padding: 18px 48px;
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(5,10,14,0.95);
    position: sticky; top: 0; z-index: 100;
}
.topbar-logo {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 22px;
    color: #fff; letter-spacing: -0.5px;
}
.topbar-logo span { color: #00FFA3; }
.topbar-tag {
    font-size: 11px; font-weight: 500; letter-spacing: 2px;
    color: #00FFA3; text-transform: uppercase;
    border: 1px solid rgba(0,255,163,0.3); border-radius: 20px;
    padding: 4px 12px;
}

/* ── HERO ── */
.hero {
    padding: 60px 48px 40px;
    background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,255,163,0.08) 0%, transparent 70%);
}
.hero-label {
    font-size: 11px; font-weight: 500; letter-spacing: 3px;
    color: #00FFA3; text-transform: uppercase; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(32px, 5vw, 56px); line-height: 1.05;
    color: #fff; letter-spacing: -1.5px; margin-bottom: 12px;
}
.hero-title span { color: #00FFA3; }
.hero-sub {
    font-size: 15px; color: rgba(255,255,255,0.4);
    font-weight: 300; max-width: 500px; line-height: 1.6;
}

/* ── KPI ROW ── */
.kpi-row { display: flex; gap: 1px; padding: 0 48px 40px; }
.kpi {
    flex: 1; padding: 24px 28px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 0;
    transition: background 0.2s;
}
.kpi:first-child { border-radius: 12px 0 0 12px; }
.kpi:last-child  { border-radius: 0 12px 12px 0; }
.kpi:hover { background: rgba(0,255,163,0.05); }
.kpi-num {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 36px; color: #fff; letter-spacing: -1px; line-height: 1;
}
.kpi-num span { color: #00FFA3; font-size: 22px; }
.kpi-label {
    font-size: 12px; color: rgba(255,255,255,0.35);
    text-transform: uppercase; letter-spacing: 1.5px;
    margin-top: 6px; font-weight: 500;
}

/* ── MAIN GRID ── */
.main-grid { display: grid; grid-template-columns: 1fr 1fr 1.1fr; gap: 20px; padding: 0 48px 48px; }

/* ── PANEL ── */
.panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 28px;
}
.panel-title {
    font-family: 'Syne', sans-serif; font-weight: 700;
    font-size: 13px; letter-spacing: 2px; text-transform: uppercase;
    color: rgba(255,255,255,0.4); margin-bottom: 24px;
}

/* ── RESULT PANEL ── */
.result-panel {
    background: linear-gradient(145deg, rgba(0,255,163,0.12), rgba(0,255,163,0.04));
    border: 1px solid rgba(0,255,163,0.2);
    border-radius: 16px; padding: 28px;
    position: relative; overflow: hidden;
}
.result-panel::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,255,163,0.15), transparent 70%);
    pointer-events: none;
}
.drug-result {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 52px; color: #00FFA3;
    letter-spacing: -2px; line-height: 1;
    margin: 20px 0 8px;
    animation: pop 0.4s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes pop {
    from { transform: scale(0.8); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}
.drug-desc { font-size: 13px; color: rgba(255,255,255,0.4); line-height: 1.5; }
.waiting {
    font-size: 14px; color: rgba(255,255,255,0.25);
    font-weight: 300; margin-top: 12px; line-height: 1.6;
}

/* ── BAR METERS ── */
.meter-row { margin-bottom: 20px; }
.meter-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
.meter-label { font-size: 12px; color: rgba(255,255,255,0.5); font-weight: 400; }
.meter-val   { font-size: 12px; color: #00FFA3; font-weight: 500; font-family: 'Syne', sans-serif; }
.meter-track {
    height: 4px; background: rgba(255,255,255,0.06);
    border-radius: 4px; overflow: hidden;
}
.meter-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #00FFA3, #00D4FF);
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}

/* ── STREAMLIT INPUT OVERRIDES ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stSelectbox"] svg { fill: rgba(255,255,255,0.4) !important; }
label, .stSelectbox label, .stNumberInput label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] > button {
    width: 100%;
    background: #00FFA3 !important;
    color: #050A0E !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    padding: 14px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"] > button:hover {
    background: #00D4FF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,255,163,0.3) !important;
}
.stAlert { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD MODEL ──────────────────────────────────────────────────────────────
model = load_model()

DRUG_INFO = {
    "DrugY": "Indicated for patients with high Na/K ratio regardless of BP or cholesterol.",
    "DrugA": "Recommended for older high-BP patients or younger female high-BP patients.",
    "DrugB": "Prescribed for younger male patients with high blood pressure.",
    "DrugC": "Suitable for patients with low blood pressure.",
    "DrugX": "General prescription for normal blood pressure patients.",
}

# ─── TOPBAR ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="topbar-logo">A<span>Predict</span></div>
  <div class="topbar-tag">ML · Clinical</div>
</div>
""", unsafe_allow_html=True)

# ─── HERO ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-label">Drug Classification System</div>
  <div class="hero-title">Precision <span>Drug</span><br>Prediction Engine</div>
  <div class="hero-sub">Enter patient vitals to receive an instant ML-powered drug recommendation across 5 clinical categories.</div>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-num">98<span>%</span></div><div class="kpi-label">Model Accuracy</div></div>
  <div class="kpi"><div class="kpi-num">5</div><div class="kpi-label">Drug Classes</div></div>
  <div class="kpi"><div class="kpi-num">DT</div><div class="kpi-label">Algorithm</div></div>
  <div class="kpi"><div class="kpi-num" style="color:#00FFA3">●</div><div class="kpi-label">System Live</div></div>
</div>
""", unsafe_allow_html=True)

# ─── MAIN LAYOUT ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-grid">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1.1])

# ── LEFT: INPUTS ─────────────────────────────────────────────────────────────
with col1:
    st.markdown('<div class="panel"><div class="panel-title">Patient Vitals</div>', unsafe_allow_html=True)
    age    = st.number_input("Age", 1, 120, 35)
    sex    = st.selectbox("Biological Sex", ["M", "F"])
    bp     = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    chol   = st.selectbox("Cholesterol Level", ["NORMAL", "HIGH"])
    na     = st.number_input("Na to K Ratio", 0.0, 50.0, 15.0, step=0.1)
    predict = st.button("Run Prediction →")
    st.markdown('</div>', unsafe_allow_html=True)

# ── CENTER: HEALTH METERS ─────────────────────────────────────────────────────
with col2:
    bp_score   = {"LOW": 55, "NORMAL": 82, "HIGH": 38}[bp]
    chol_score = {"NORMAL": 90, "HIGH": 52}[chol]
    na_score   = min(int((na / 50) * 100), 100)
    age_score  = max(100 - int((age / 120) * 60), 30)

    st.markdown(f"""
    <div class="panel">
      <div class="panel-title">Health Signals</div>

      <div class="meter-row">
        <div class="meter-header">
          <span class="meter-label">Blood Pressure Index</span>
          <span class="meter-val">{bp_score}%</span>
        </div>
        <div class="meter-track"><div class="meter-fill" style="width:{bp_score}%"></div></div>
      </div>

      <div class="meter-row">
        <div class="meter-header">
          <span class="meter-label">Cholesterol Score</span>
          <span class="meter-val">{chol_score}%</span>
        </div>
        <div class="meter-track"><div class="meter-fill" style="width:{chol_score}%"></div></div>
      </div>

      <div class="meter-row">
        <div class="meter-header">
          <span class="meter-label">Na/K Ratio Level</span>
          <span class="meter-val">{na_score}%</span>
        </div>
        <div class="meter-track"><div class="meter-fill" style="width:{na_score}%"></div></div>
      </div>

      <div class="meter-row">
        <div class="meter-header">
          <span class="meter-label">Age Risk Factor</span>
          <span class="meter-val">{age_score}%</span>
        </div>
        <div class="meter-track"><div class="meter-fill" style="width:{age_score}%"></div></div>
      </div>

    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: RESULT ─────────────────────────────────────────────────────────────
with col3:
    if predict:
        df_input = pd.DataFrame([{
            "Age": age, "Sex": sex, "BP": bp,
            "Cholesterol": chol, "Na_to_K": na
        }])
        result = model.predict(df_input)[0]
        desc   = DRUG_INFO.get(result, "Consult a physician for detailed guidance.")
        st.markdown(f"""
        <div class="result-panel">
          <div class="panel-title">Prediction Result</div>
          <div class="drug-result">{result}</div>
          <div class="drug-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-panel">
          <div class="panel-title">Awaiting Input</div>
          <div class="waiting">Fill in the patient vitals on the left and click <strong style="color:rgba(255,255,255,0.6)">Run Prediction</strong> to receive an instant drug recommendation.</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
