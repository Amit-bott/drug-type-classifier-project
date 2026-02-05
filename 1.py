import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

st.set_page_config(page_title="Combined ML Dashboard", layout="wide")
BASE = Path.cwd()

DRUG_MODEL_FILE = BASE / "drug_model.pkl"
INS_MODEL_FILE = BASE / "insurance_model.pkl"
DRUG_CSV = BASE / "drug200.csv"
INS_CSV = BASE / "insurance.csv"

st.title("Combined ML Dashboard — Drug Classifier & Insurance Cost Predictor")
st.markdown(
    """
This dashboard bundles two ML modules:
- **Drug classifier** — predicts drug type from patient features.
- **Insurance cost predictor** — estimates annual insurance charges.

Drop your `drug200.csv` and `insurance.csv` in the app folder or upload via the UI.
"""
)
def make_demo_drug():
    # a demo version of the common drug200 dataset shape
    df = pd.DataFrame({
        "Age": np.random.randint(18, 80, size=200),
        "Sex": np.random.choice(["F", "M"], size=200),
        "BP": np.random.choice(["HIGH", "LOW", "NORMAL"], size=200),
        "Cholesterol": np.random.choice(["NORMAL", "HIGH"], size=200),
        "Na_to_K": np.round(np.random.normal(15, 2.5, size=200), 2),
        "Drug": np.random.choice(["drugA","drugB","drugC","drugX","drugY"], size=200)
    })
    return df

def make_demo_insurance(n=1000, seed=42):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({
        "age": rng.randint(18, 65, size=n),
        "sex": rng.choice(["male","female"], size=n),
        "bmi": np.round(rng.normal(30,6,size=n),1),
        "children": rng.randint(0,5,size=n),
        "smoker": rng.choice(["yes","no"], size=n, p=[0.2,0.8]),
        "region": rng.choice(["southwest","southeast","northwest","northeast"], size=n),
    })
    df["charges"] = (
        250 + df["age"]*12 + df["bmi"]*50 + df["children"]*200 +
        np.where(df["smoker"]=="yes", 10000, 0) + rng.normal(0,2000,size=n)
    ).round(2)
    return df

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

# -------------------------
# Load or upload datasets
# -------------------------
st.sidebar.header("Data / Models")

st.sidebar.markdown("You can upload CSV files here (or place `drug200.csv` and `insurance.csv` in the app folder).")

uploaded_drug = st.sidebar.file_uploader("Upload drug200.csv", type=["csv"], key="drug_upload")
uploaded_ins = st.sidebar.file_uploader("Upload insurance.csv", type=["csv"], key="ins_upload")

if uploaded_drug:
    try:
        df_drug = pd.read_csv(uploaded_drug)
        st.sidebar.success("Loaded uploaded drug200.csv")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded drug CSV: {e}")
        df_drug = None
elif DRUG_CSV.exists():
    df_drug = pd.read_csv(DRUG_CSV)
    st.sidebar.info(f"Loaded drug200.csv from app folder ({len(df_drug)} rows)")
else:
    df_drug = make_demo_drug()
    st.sidebar.info("Using demo drug dataset (replace with your drug200.csv)")

if uploaded_ins:
    try:
        df_ins = pd.read_csv(uploaded_ins)
        st.sidebar.success("Loaded uploaded insurance.csv")
    except Exception as e:
        st.sidebar.error(f"Error reading uploaded insurance CSV: {e}")
        df_ins = None
elif INS_CSV.exists():
    df_ins = pd.read_csv(INS_CSV)
    st.sidebar.info(f"Loaded insurance.csv from app folder ({len(df_ins)} rows)")
else:
    df_ins = make_demo_insurance()
    st.sidebar.info("Using demo insurance dataset (replace with your insurance.csv)")

# -------------------------
# Model training functions
# -------------------------
def build_drug_pipeline(df):
    # Expect df with columns Age, Sex, BP, Cholesterol, Na_to_K, Drug (target)
    X = df.drop(columns=["Drug"])
    y = df["Drug"]
    # feature lists
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # small dataset: scaling optional for tree-based models; keep for safety
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    clf = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
    ])

    return clf, X, y

def build_insurance_pipeline(df):
    # Expect df with columns age, sex, bmi, children, smoker, region, charges
    X = df.drop(columns=["charges"])
    y = df["charges"]
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object","category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_features),
        ("cat", cat_pipeline, categorical_features)
    ])

    reg = Pipeline([
        ("pre", preprocessor),
        ("reg", RandomForestRegressor(n_estimators=150, random_state=42))
    ])

    return reg, X, y

def train_and_save_drug(df, force_retrain=False):
    if DRUG_MODEL_FILE.exists() and not force_retrain:
        model = load_model(DRUG_MODEL_FILE)
        return model, None
    model, X, y = build_drug_pipeline(df)
    # if very small dataset, don't stratify
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    save_model(model, DRUG_MODEL_FILE)
    # evaluate
    preds = model.predict(Xte)
    acc = accuracy_score(yte, preds)
    return model, {"accuracy": acc, "report": classification_report(yte, preds, zero_division=0)}

def train_and_save_ins(df, force_retrain=False):
    if INS_MODEL_FILE.exists() and not force_retrain:
        model = load_model(INS_MODEL_FILE)
        return model, None
    model, X, y = build_insurance_pipeline(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(Xtr, ytr)
    save_model(model, INS_MODEL_FILE)
    preds = model.predict(Xte)
    rmse = mean_squared_error(yte, preds, squared=False)
    r2 = r2_score(yte, preds)
    return model, {"rmse": rmse, "r2": r2}

# -------------------------
# Load or train models (initial)
# -------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.header("Drug Model")
    if DRUG_MODEL_FILE.exists():
        st.write(f"Found saved model: `{DRUG_MODEL_FILE.name}`")
        try:
            drug_model = load_model(DRUG_MODEL_FILE)
        except Exception as e:
            st.error(f"Failed to load saved drug model: {e}")
            drug_model = None
    else:
        st.write("No saved drug model found. Will train from dataset below (or use demo).")
        drug_model = None

    if st.button("(Re)train Drug Model"):
        with st.spinner("Training drug classifier..."):
            drug_model, stats = train_and_save_drug(df_drug, force_retrain=True)
        if stats:
            st.success(f"Drug model trained. Test accuracy: {stats['accuracy']:.4f}")
            st.text("Classification report:")
            st.text(stats["report"])
        else:
            st.success("Drug model loaded from disk.")

with col2:
    st.header("Insurance Model")
    if INS_MODEL_FILE.exists():
        st.write(f"Found saved model: `{INS_MODEL_FILE.name}`")
        try:
            ins_model = load_model(INS_MODEL_FILE)
        except Exception as e:
            st.error(f"Failed to load saved insurance model: {e}")
            ins_model = None
    else:
        st.write("No saved insurance model found. Will train from dataset below (or use demo).")
        ins_model = None

    if st.button("(Re)train Insurance Model"):
        with st.spinner("Training insurance regressor..."):
            ins_model, stats = train_and_save_ins(df_ins, force_retrain=True)
        if stats:
            st.success(f"Insurance model trained. RMSE: {stats['rmse']:.2f}, R2: {stats['r2']:.3f}")
        else:
            st.success("Insurance model loaded from disk.")

# Ensure models exist (if not, attempt auto-train)
if 'drug_model' not in locals() or (drug_model is None and DRUG_MODEL_FILE.exists()):
    try:
        drug_model = load_model(DRUG_MODEL_FILE)
    except Exception:
        drug_model = None

if 'ins_model' not in locals() or (ins_model is None and INS_MODEL_FILE.exists()):
    try:
        ins_model = load_model(INS_MODEL_FILE)
    except Exception:
        ins_model = None

if drug_model is None:
    # train automatically (no button press) to make the app usable
    with st.spinner("Auto-training drug model (if needed)..."):
        try:
            drug_model, stats = train_and_save_drug(df_drug, force_retrain=False)
        except Exception as e:
            st.error("Couldn't train or load drug model: " + str(e))
            drug_model = None

if ins_model is None:
    with st.spinner("Auto-training insurance model (if needed)..."):
        try:
            ins_model, stats = train_and_save_ins(df_ins, force_retrain=False)
        except Exception as e:
            st.error("Couldn't train or load insurance model: " + str(e))
            ins_model = None

st.markdown("---")

# -------------------------
# Tabs — Drug & Insurance
# -------------------------
tab1, tab2 = st.tabs(["Drug Classifier", "Insurance Predictor"])

with tab1:
    st.header("Drug Classification")
    st.markdown("Make a prediction or inspect the dataset and model performance.")

    st.subheader("Dataset preview")
    st.dataframe(df_drug.head())

    st.subheader("Prediction form")
    with st.form("drug_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=120, value=int(df_drug['Age'].median()))
            sex = st.selectbox("Sex", options=sorted(df_drug['Sex'].dropna().unique()))
        with c2:
            bp = st.selectbox("BP", options=sorted(df_drug['BP'].dropna().unique()))
            chol = st.selectbox("Cholesterol", options=sorted(df_drug['Cholesterol'].dropna().unique()))
        with c3:
            na_to_k = st.number_input("Na_to_K", value=float(df_drug['Na_to_K'].median()), format="%.2f")
        submit_drug = st.form_submit_button("Predict Drug")

    if submit_drug:
        if drug_model is None:
            st.error("Drug model not available.")
        else:
            Xnew = pd.DataFrame([{"Age": age, "Sex": sex, "BP": bp, "Cholesterol": chol, "Na_to_K": na_to_k }])
            try:
                pred = drug_model.predict(Xnew)[0]
                st.success(f"Predicted Drug: **{pred}**")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

    st.markdown("### Model evaluation on available dataset (quick check)")
    try:
        X_all = df_drug.drop(columns=["Drug"])
        y_all = df_drug["Drug"]
        preds_all = drug_model.predict(X_all)
        acc_all = accuracy_score(y_all, preds_all)
        st.write(f"Accuracy on whole dataset (not a true test): **{acc_all:.4f}**")
    except Exception as e:
        st.info("Couldn't compute full-dataset metrics: " + str(e))

with tab2:
    st.header("Insurance Cost Prediction")
    st.markdown("Predict insurance charges and inspect model performance.")

    st.subheader("Dataset preview")
    st.dataframe(df_ins.head())

    st.subheader("Prediction form")
    with st.form("ins_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age_i = st.number_input("Age", min_value=0, max_value=120, value=int(df_ins['age'].median()))
            sex_i = st.selectbox("Sex", options=sorted(df_ins['sex'].dropna().unique()))
        with c2:
            bmi_i = st.number_input("BMI", min_value=10.0, max_value=60.0, value=float(df_ins['bmi'].median()))
            children_i = st.number_input("Children", min_value=0, max_value=10, value=int(df_ins['children'].median()))
        with c3:
            smoker_i = st.selectbox("Smoker", options=sorted(df_ins['smoker'].dropna().unique()))
            region_i = st.selectbox("Region", options=sorted(df_ins['region'].dropna().unique()))
        submit_ins = st.form_submit_button("Predict Charges")

    if submit_ins:
        if ins_model is None:
            st.error("Insurance model not available.")
        else:
            Xnew = pd.DataFrame([{
                "age": age_i, "sex": sex_i, "bmi": bmi_i,
                "children": children_i, "smoker": smoker_i, "region": region_i
            }])
            try:
                charges_pred = ins_model.predict(Xnew)[0]
                st.success(f"Predicted annual charges: **${charges_pred:,.2f}**")
            except Exception as e:
                st.error("Prediction failed: " + str(e))

    st.markdown("### Model evaluation (quick test)")
    try:
        X_all = df_ins.drop(columns=["charges"])
        y_all = df_ins["charges"]
        preds_all = ins_model.predict(X_all)
        rmse_all = mean_squared_error(y_all, preds_all, squared=False)
        r2_all = r2_score(y_all, preds_all)
        st.write(f"RMSE on whole dataset (not a true test): **{rmse_all:.2f}**, R²: **{r2_all:.3f}**")
    except Exception as e:
        st.info("Couldn't compute full-dataset metrics: " + str(e))

    # Feature importance from RandomForestRegressor (after preprocessing not directly mappable to original features easily)
    st.markdown("### Feature importance (approximate)")
    try:
        # Attempt to extract regressor and its preprocessing
        pre = ins_model.named_steps["pre"]
        reg = ins_model.named_steps["reg"]
        # Build transformed column names
        # numeric features
        numeric_cols = df_ins.drop(columns=["charges"]).select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_ins.drop(columns=["charges"]).select_dtypes(include=["object","category"]).columns.tolist()

        # get OHE categories length to approximate column names
        ohe = None
        for name, trans, cols in pre.transformers_:
            if name == "cat":
                # trans is a Pipeline; get the OHE
                ohe = trans.named_steps["ohe"]

        feature_names = []
        # numeric names
        feature_names.extend(numeric_cols)
        # for categorical, get names from ohe if available
        if ohe is not None:
            try:
                cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
                feature_names.extend(cat_feature_names)
            except Exception:
                # fallback simple names
                for c in categorical_cols:
                    # use one column per category as a placeholder
                    feature_names.append(c)
        # get importances
        importances = reg.feature_importances_
        # align lengths
        if len(importances) == len(feature_names):
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8,4))
            fi.plot.bar(ax=ax)
            ax.set_title("Top feature importances")
            ax.set_ylabel("Importance")
            st.pyplot(fig)
        else:
            # just show top importances without names
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(range(len(importances)), importances)
            ax.set_title("Feature importances (unnamed)")
            st.pyplot(fig)
    except Exception as e:
        st.info("Couldn't compute feature importance: " + str(e))

st.markdown("---")
st.caption("Tip: To use your own datasets, place `drug200.csv` and `insurance.csv` in the same folder as this app or upload them in the sidebar. Models are saved as `drug_model.pkl` and `insurance_model.pkl`.")

