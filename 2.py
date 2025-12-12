# # drug_dashboard_app.py
# """
# Drug Prediction Dashboard (Streamlit)
# Run:
#     streamlit run drug_dashboard_app.py
# Requirements:
#     pip install streamlit pandas numpy scikit-learn joblib matplotlib
# """

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# from pathlib import Path
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# st.set_page_config(page_title="Drug Prediction Dashboard", layout="wide")
# BASE = Path.cwd()
# MODEL_FILE = BASE / "drug_model.pkl"
# CSV_FILE = BASE / "drug200.csv"

# st.title("Drug Prediction Dashboard")
# st.markdown("Upload your dataset, train the model, and make predictions.")

# # ---------------------- Utility ----------------------
# def make_demo_drug():
#     df = pd.DataFrame({
#         "Age": np.random.randint(18, 80, 200),
#         "Sex": np.random.choice(["F", "M"], 200),
#         "BP": np.random.choice(["HIGH", "LOW", "NORMAL"], 200),
#         "Cholesterol": np.random.choice(["NORMAL", "HIGH"], 200),
#         "Na_to_K": np.round(np.random.normal(15, 2.5, 200), 2),
#         "Drug": np.random.choice(["drugA", "drugB", "drugC", "drugX", "drugY"], 200)
#     })
#     return df

# def save_model(model): joblib.dump(model, MODEL_FILE)

# def load_model(): return joblib.load(MODEL_FILE)

# # ---------------------- Load Data ----------------------
# st.sidebar.header("Dataset Upload")
# file = st.sidebar.file_uploader("Upload drug200.csv", type=["csv"])

# if file:
#     df = pd.read_csv(file)
#     st.sidebar.success("Uploaded dataset loaded.")
# elif CSV_FILE.exists():
#     df = pd.read_csv(CSV_FILE)
#     st.sidebar.info("Loaded drug200.csv from app folder.")
# else:
#     df = make_demo_drug()
#     st.sidebar.info("Using demo dataset.")

# # ---------------------- Training Pipeline ----------------------
# def build_pipeline(df):
#     X = df.drop(columns=["Drug"])
#     y = df["Drug"]

#     numeric = X.select_dtypes(include=[np.number]).columns.tolist()
#     categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

#     num_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])

#     cat_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
#     ])

#     pre = ColumnTransformer([
#         ("num", num_pipe, numeric),
#         ("cat", cat_pipe, categorical)
#     ])

#     model = Pipeline([
#         ("pre", pre),
#         ("clf", RandomForestClassifier(n_estimators=120, random_state=42))
#     ])

#     return model, X, y

# # ---------------------- Train Model ----------------------
# st.header("Train Drug Model")

# if st.button("Train Model"):
#     model, X, y = build_pipeline(df)

#     try:
#         Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     except:
#         Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

#     with st.spinner("Training model..."):
#         model.fit(Xtr, ytr)
#         save_model(model)

#         preds = model.predict(Xte)
#         acc = accuracy_score(yte, preds)
#         rep = classification_report(yte, preds, zero_division=0)

#     st.success(f"Model trained. Accuracy: {acc:.4f}")
#     st.text(rep)

# # ---------------------- Prediction Form ----------------------
# st.header("Predict Drug Type")
# if MODEL_FILE.exists():
#     model = load_model()
# else:
#     st.warning("Model not trained yet.")
#     model = None

# if model:
#     with st.form("predict_form"):
#         c1, c2, c3 = st.columns(3)
#         with c1:
#             age = st.number_input("Age", min_value=1, max_value=120, value=40)
#             sex = st.selectbox("Sex", df["Sex"].unique())
#         with c2:
#             bp = st.selectbox("BP", df["BP"].unique())
#             chol = st.selectbox("Cholesterol", df["Cholesterol"].unique())
#         with c3:
#             na_k = st.number_input("Na_to_K", value=float(df["Na_to_K"].median()))

#         submit = st.form_submit_button("Predict")

#     if submit:
#         Xnew = pd.DataFrame([{ "Age": age, "Sex": sex, "BP": bp,
#                                "Cholesterol": chol, "Na_to_K": na_k }])
#         pred = model.predict(Xnew)[0]
#         st.success(f"Predicted Drug: {pred}")

# st.caption("Place drug200.csv in the same folder to auto-load. Model saved as drug_model.pkl.")








# drug_dashboard_app.py
"""
Drug Prediction Dashboard (Streamlit) — FULL VERSION WITH ALL GRAPHS
Run:
    streamlit run drug_dashboard_app.py
Requirements:
    pip install streamlit pandas numpy scikit-learn joblib matplotlib seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Drug Prediction Dashboard", layout="wide")
BASE = Path.cwd()
MODEL_FILE = BASE / "drug_model.pkl"
CSV_FILE = BASE / "drug200.csv"

# ---------------------- TITLE ----------------------
st.title("💊 Drug Prediction Machine Learning Dashboard")
st.markdown("This dashboard includes training, prediction, and ALL visual charts & graphs.")

# ---------------------- UTILITY ----------------------
def make_demo_drug():
    df = pd.DataFrame({
        "Age": np.random.randint(18, 80, 200),
        "Sex": np.random.choice(["F", "M"], 200),
        "BP": np.random.choice(["HIGH", "LOW", "NORMAL"], 200),
        "Cholesterol": np.random.choice(["NORMAL", "HIGH"], 200),
        "Na_to_K": np.round(np.random.normal(15, 2.5, 200), 2),
        "Drug": np.random.choice(["drugA", "drugB", "drugC", "drugX", "drugY"], 200)
    })
    return df

def save_model(model): joblib.dump(model, MODEL_FILE)

def load_model(): return joblib.load(MODEL_FILE)

# ---------------------- LOAD DATA ----------------------
st.sidebar.header("📂 Dataset Upload")
file = st.sidebar.file_uploader("Upload drug200.csv", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.sidebar.success("Uploaded dataset loaded.")
elif CSV_FILE.exists():
    df = pd.read_csv(CSV_FILE)
    st.sidebar.info("Loaded drug200.csv from app folder.")
else:
    df = make_demo_drug()
    st.sidebar.info("Using demo dataset.")

# ---------------------- FULL DATA PREVIEW ----------------------
st.subheader("📘 Dataset Preview")
st.dataframe(df.head())

# ---------------------- ALL GRAPHS SECTION ----------------------
st.header("📊 Data Visualization — All Charts & Graphs")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Bar Chart — Age Distribution")
    st.bar_chart(df['Age'])

with col2:
    st.subheader("Line Chart — Na_to_K Trend")
    st.line_chart(df['Na_to_K'])

# Histogram
st.subheader("Histogram — Age")
fig, ax = plt.subplots()
ax.hist(df['Age'], bins=15)
st.pyplot(fig)

# Countplot
st.subheader("Categorical Count Charts")
fig, ax = plt.subplots(1,3, figsize=(12,4))
sns.countplot(df, x='Sex', ax=ax[0])
sns.countplot(df, x='BP', ax=ax[1])
sns.countplot(df, x='Cholesterol', ax=ax[2])
st.pyplot(fig)

# Heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(df[['Age','Na_to_K']].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pairplot
st.subheader("Pairplot — Numeric Features")
fig = sns.pairplot(df[['Age','Na_to_K']])
st.pyplot(fig)

# ---------------------- BUILD PIPELINE ----------------------
def build_pipeline(df):
    X = df.drop(columns=["Drug"])
    y = df["Drug"]

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])

    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))])

    pre = ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])

    model = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=120, random_state=42))])
    return model, X, y

# ---------------------- TRAINING ----------------------
st.header("🧠 Train Drug Model")

if st.button("Train Model"):
    model, X, y = build_pipeline(df)

    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Training model..."):
        model.fit(Xtr, ytr)
        save_model(model)

        preds = model.predict(Xte)
        acc = accuracy_score(yte, preds)
        rep = classification_report(yte, preds, zero_division=0)

    st.success(f"Model trained successfully. Accuracy: {acc:.4f}")

    # Feature Importance
    st.subheader("Feature Importance Chart")
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(range(len(importances)), importances)
    ax.set_title("Feature Importance")
st.pyplot(fig)

# ---------------------- PREDICTION ----------------------
st.header("🔍 Predict Drug Type")

if MODEL_FILE.exists():
    model = load_model()
else:
    st.warning("Model not trained yet.")
    model = None

if model:
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=1, max_value=120, value=40)
            sex = st.selectbox("Sex", df['Sex'].unique())
        with c2:
            bp = st.selectbox("BP", df['BP'].unique())
            chol = st.selectbox("Cholesterol", df['Cholesterol'].unique())
        with c3:
            na_k = st.number_input("Na_to_K", value=float(df['Na_to_K'].median()))

        submit = st.form_submit_button("Predict Drug")

    if submit:
        Xnew = pd.DataFrame([{ "Age": age, "Sex": sex, "BP": bp,
                               "Cholesterol": chol, "Na_to_K": na_k }])
        pred = model.predict(Xnew)[0]
        st.success(f"Predicted Drug: **{pred}**")

st.caption("Place drug200.csv in the app folder. Model is saved as drug_model.pkl.")