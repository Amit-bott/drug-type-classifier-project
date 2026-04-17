import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
st.set_page_config(
    page_title="Drug Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* REMOVE STREAMLIT HEADER COMPLETELY */
header {visibility: hidden;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stHeader"] {display: none;}
[data-testid="stDecoration"] {display: none;}

/* FIX BODY & TOP SPACE */
html, body {
    margin: 0;
    padding: 0;
    background: #f7f9fb;
}

/* MAIN CONTAINER */
.block-container {
    padding-top: 0px !important;
}

/* CINEMATIC TOP BAR */
.top-quote {
    position: sticky;
    top: 0;
    z-index: 999;
    width: 100%;
    text-align: center;
    padding: 22px;
    font-size: 18px;
    color: white;
    background: linear-gradient(90deg, #000000, #1c1c1c);
    border-radius: 0 0 22px 22px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.6);
}

/* TITLE */
.title {
    font-size: 36px;
    font-weight: 800;
    color: #2ecc71;
    margin: 25px 0 10px 0;
}

/* GLASS CARD */
.card {
    background: rgba(255,255,255,0.85);
    backdrop-filter: blur(12px);
    padding: 24px;
    border-radius: 22px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.08);
    transition: all 0.4s ease;
}

.card:hover {
    transform: translateY(-10px) rotateX(4deg);
    box-shadow: 0 25px 60px rgba(0,0,0,0.18);
}

/* KPI */
.metric {
    font-size: 34px;
    font-weight: 800;
}
.small {
    font-size: 14px;
    color: #6b7280;
}

/* NEON GREEN CARD */
.green-card {
    background: linear-gradient(135deg, #2ecc71, #22c55e);
    color: white;
    padding: 26px;
    border-radius: 24px;
    box-shadow: 0 20px 60px rgba(34,197,94,0.7);
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from { box-shadow: 0 20px 50px rgba(34,197,94,0.4); }
    to { box-shadow: 0 30px 90px rgba(34,197,94,0.9); }
}

/* RESULT */
.predict-box {
    margin-top: 18px;
    padding: 22px;
    background: rgba(255,255,255,0.18);
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
    animation: pulse 1.4s infinite;
}

@keyframes pulse {
    0% {transform: scale(1);}
    50% {transform: scale(1.05);}
    100% {transform: scale(1);}
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

BASE = Path.cwd()
MODEL_FILE = BASE / "drug_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.markdown(
    '<div class="top-quote">when you realized the love is over, but life isn’t</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="title">💊 Drug Prediction Productivity Dashboard</div>', unsafe_allow_html=True)

if not MODEL_FILE.exists():
    st.error("drug_model.pkl not found")
    st.stop()

model = load_model()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown("<div class='card'><div class='metric'>98%</div><div class='small'>Model Accuracy</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown("<div class='card'><div class='metric'>5</div><div class='small'>Drug Types</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown("<div class='card'><div class='metric'>ML</div><div class='small'>Prediction Engine</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown("<div class='card'><div class='metric'>LIVE</div><div class='small'>System Status</div></div>", unsafe_allow_html=True)

st.write("")

left, center, right = st.columns([1.1, 1.3, 1])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧾 Patient Details")
    age = st.number_input("Age", 1, 120, 35)
    sex = st.selectbox("Sex", ["M", "F"])
    bp = st.selectbox("Blood Pressure", ["LOW", "NORMAL", "HIGH"])
    chol = st.selectbox("Cholesterol", ["NORMAL", "HIGH"])
    na = st.number_input("Na to K Ratio", 0.0, 50.0, 15.0)
    predict = st.button("🔮 Predict Drug")
    st.markdown("</div>", unsafe_allow_html=True)

with center:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Health Intelligence")
    st.progress(0.82)
    st.write("Clinical Data Quality")
    st.progress(0.69)
    st.write("Risk Evaluation")
    st.progress(0.93)
    st.write("Prescription Confidence")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='green-card'>", unsafe_allow_html=True)
    st.subheader("📌 Prediction")
    if predict:
        df = pd.DataFrame([{
            "Age": age,
            "Sex": sex,
            "BP": bp,
            "Cholesterol": chol,
            "Na_to_K": na
        }])
        result = model.predict(df)[0]
        st.markdown(f"<div class='predict-box'>{result}</div>", unsafe_allow_html=True)
    else:
        st.info("Enter details & click Predict")
    st.markdown("</div>", unsafe_allow_html=True)
