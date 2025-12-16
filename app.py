import io
from typing import Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from transformers import pipeline as hf_pipeline


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="CSV Insight Studio",
    page_icon="💗",
    layout="wide",
)


# ======================================================
# CUSTOM UI (Hero Centered + Clean Theme)
# ======================================================
CUSTOM_CSS = """
<style>

.stApp {
    background: radial-gradient(circle at top left, rgba(236,72,153,0.28), transparent 50%),
                radial-gradient(circle at bottom right, rgba(129,140,248,0.32), transparent 50%),
                #050016;
    color: #f9fafb;
    font-family: 'Inter', sans-serif;
}

/* Remove extra padding */
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.4rem;
}

/* HERO SECTION CENTER */
.hero-wrapper {
    width: 100%;
    text-align: center;
    margin-top: 1.6rem;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #ff4ecb, #c4b5fd, #38bdf8);
    -webkit-background-clip: text;
    color: transparent;
    letter-spacing: 0.02em;
    text-align: center;
}

.hero-subtext {
    font-size: 1.05rem;
    max-width: 820px;
    margin: 0.6rem auto;
    color: #e5e7eb;
    text-align: center;
    opacity: 0.95;
}

/* CARD */
.card {
    background: rgba(15,23,42,0.92);
    padding: 1.3rem;
    border-radius: 18px;
    border: 1px solid rgba(148,163,184,0.35);
    margin-top: 1rem;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #ec4899, #a855f7);
    color: white;
    border-radius: 999px;
    padding: 0.5rem 1.4rem;
    border: none;
    font-weight: 600;
    box-shadow: 0 10px 28px rgba(236,72,153,0.36);
}

.stButton > button:hover {
    box-shadow: 0 16px 34px rgba(129,140,248,0.65);
}

/* Hide accidental blank search bar */
div[role="search"], 
input[placeholder="Search"], 
[data-testid*="filter"] input {
    display: none !important;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ======================================================
# HERO SECTION (Centered)
# ======================================================
st.markdown(
    """
<div class="hero-wrapper">
    <div class="hero-title">CSV Insight Studio</div>
    <div class="hero-subtext">
        Upload any CSV and explore it like a personal data lab — inspect structure, 
        create visuals, build ML models, run clustering, and get expert AI insights.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ======================================================
# UPLOAD SECTION
# ======================================================
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.subheader("📂 Upload CSV File")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=["csv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

    with right:
        st.subheader("⚙️ Read Options")
        c1, c2 = st.columns(2)
        sep = c1.selectbox("Separator", [",", ";", "|", "\\t"], index=0)
        encoding = c2.selectbox("Encoding", ["utf-8", "latin-1"], index=0)

    st.markdown("</div>", unsafe_allow_html=True)


# ======================================================
# LOAD CSV FUNCTION
# ======================================================
@st.cache_data
def load_csv(bytes_data, sep, encoding):
    sep_eff = "\t" if sep == "\\t" else sep
    return pd.read_csv(io.BytesIO(bytes_data), sep=sep_eff, encoding=encoding)


datasets: Dict[str, pd.DataFrame] = {}

if uploaded_files:
    for f in uploaded_files:
        try:
            datasets[f.name] = load_csv(f.read(), sep, encoding)
        except Exception as e:
            st.error(f"❌ Failed to read {f.name}: {e}")

if not datasets:
    st.info("⬆️ Upload at least one CSV to begin analysis.")
    st.stop()


# ======================================================
# ACTIVE DATASET
# ======================================================
dataset_name = (
    list(datasets.keys())[0]
    if len(datasets) == 1
    else st.selectbox("Active Dataset", datasets.keys())
)

df = datasets[dataset_name]

# Counters
col1, col2, col3 = st.columns(3)
col1.success(f"📄 File: {dataset_name}")
col2.success(f"🔢 Shape: {df.shape[0]} × {df.shape[1]}")
col3.success(
    f"Numeric: {df.select_dtypes(include=[np.number]).shape[1]} | "
    f"Categorical: {df.select_dtypes(exclude=[np.number]).shape[1]}"
)


# ======================================================
# HELPER FUNCTIONS
# ======================================================
def profile(df: pd.DataFrame) -> pd.DataFrame:
    p = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non-null": df.notnull().sum(),
            "null": df.isnull().sum(),
            "unique": df.nunique(),
        }
    )
    p["null_%"] = (p["null"] / len(df) * 100).round(2)
    return p


def infer_task(df: pd.DataFrame, col: str) -> str:
    y = df[col]
    if y.dtype == object or (
        pd.api.types.is_integer_dtype(y) and y.nunique() <= 20
    ):
        return "classification"
    return "regression"


def describe_dataset(df: pd.DataFrame, name: str) -> str:
    p = profile(df)
    t = [
        f"Dataset: {name}",
        f"Rows: {df.shape[0]} | Columns: {df.shape[1]}",
        "",
        "Columns:",
    ]
    for c, r in p.iterrows():
        t.append(
            f"- {c}: {r['dtype']}, null={r['null_%']}%, unique={r['unique']}"
        )
    return "\n".join(t)


def get_numeric_columns_without_ids(df: pd.DataFrame):
    """Utility: numeric columns but remove typical index/id columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_names = {"index", "id", "serial", "sr_no", "srno"}
    num_cols = [c for c in num_cols if c.lower() not in drop_names]
    return num_cols


# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Snapshot", "Visual Lab", "ML Studio", "Clustering", "AI Mentor"]
)


# ======================================================
# 1️⃣ SNAPSHOT
# ======================================================
with tab1:
    st.header("🔍 Dataset Snapshot")
    st.dataframe(df.head(20), use_container_width=True)
    st.subheader("📌 Column Summary")
    st.dataframe(profile(df), use_container_width=True)


# ======================================================
# 2️⃣ VISUAL LAB
# ======================================================
with tab2:
    st.header("📊 Visual Lab")

    num_cols = get_numeric_columns_without_ids(df)
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    chart = st.radio(
        "Choose Chart:", ["Histogram", "Scatter", "Correlation"], horizontal=True
    )

    if chart == "Histogram":
        if num_cols:
            col = st.selectbox("Column", num_cols)
            fig = px.histogram(df, x=col, nbins=35)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric columns found.")

    elif chart == "Scatter":
        if len(num_cols) >= 2:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X", num_cols)
            # by default y ko x ke alawa kuch aur set karne ke liye
            remaining = [c for c in num_cols if c != x] or num_cols
            y = c2.selectbox("Y", remaining)
            col = c3.selectbox("Color", ["(none)"] + cat_cols)
            fig = px.scatter(df, x=x, y=y, color=None if col == "(none)" else col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("At least 2 numeric columns needed for scatter plot.")

    elif chart == "Correlation":
        if len(num_cols) >= 2:
            fig = px.imshow(df[num_cols].corr(), text_auto=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for correlation heatmap.")


# ======================================================
# 3️⃣ ML STUDIO
# ======================================================
with tab3:
    st.header("🤖 ML Studio")

    target = st.selectbox("Target Column", df.columns)
    task = infer_task(df, target)
    st.info(f"Detected Task: **{task.upper()}**")

    model_name = st.selectbox(
        "Choose Model",
        ["Logistic Regression", "Random Forest Classifier"]
        if task == "classification"
        else ["Linear Regression", "Random Forest Regressor"],
    )

    test_size = st.slider("Test Split", 0.1, 0.4, 0.2)

    if st.button("Train Model"):
        try:
            clean = df.dropna(subset=[target])
            X = clean.drop(columns=[target])
            y = clean[target]

            num = X.select_dtypes(include=[np.number]).columns
            cat = X.select_dtypes(exclude=[np.number]).columns

            pre = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()),
                            ]
                        ),
                        num,
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="most_frequent")),
                                ("oh", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat,
                    ),
                ]
            )

            # -------------------- MODEL SELECTION --------------------
            if task == "classification":
                if model_name == "Logistic Regression":
                    model = LogisticRegression(
                        max_iter=500,
                        n_jobs=-1,
                        multi_class="auto",
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    )
            else:  # regression
                if model_name == "Linear Regression":
                    model = LinearRegression(n_jobs=-1)
                else:
                    model = RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        n_jobs=-1,
                    )

            pipe = Pipeline([("pre", pre), ("model", model)])

            Xtr, Xte, ytr, yte = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=y if task == "classification" else None,
            )

            pipe.fit(Xtr, ytr)
            ypred = pipe.predict(Xte)

            st.subheader("📈 Results")

            if task == "classification":
                acc = accuracy_score(yte, ypred)
                st.success(f"Accuracy: {acc:.4f}")

                # precision / recall / f1 (macro)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    yte, ypred, average="macro", zero_division=0
                )
                st.write(
                    f"Precision (macro): **{prec:.4f}**, "
                    f"Recall (macro): **{rec:.4f}**, "
                    f"F1 (macro): **{f1:.4f}**"
                )

                st.write("Confusion Matrix:")
                cm = confusion_matrix(yte, ypred)
                cm_df = pd.DataFrame(cm,
                                     index=[f"True {c}" for c in np.unique(yte)],
                                     columns=[f"Pred {c}" for c in np.unique(yte)])
                st.dataframe(cm_df)

            else:  # regression metrics
                mae = mean_absolute_error(yte, ypred)
                rmse = mean_squared_error(yte, ypred) ** 0.5
                r2 = r2_score(yte, ypred)
                st.success(f"MAE: **{mae:.3f}**")
                st.success(f"RMSE: **{rmse:.3f}**")
                st.success(f"R²: **{r2:.4f}**")

        except Exception as e:
            st.error(f"Error while training model: {e}")


# ======================================================
# 4️⃣ CLUSTERING
# ======================================================
with tab4:
    st.header("🧩 K-Means Clustering")

    num_cols = get_numeric_columns_without_ids(df)

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
    else:
        use_cols = st.multiselect("Columns for clustering", num_cols, default=num_cols[:3])
        k = st.slider("Clusters (K)", 2, 10, 4)

        if use_cols:
            X = df[use_cols].dropna()
            if X.empty:
                st.warning("Selected columns have only missing values after dropping NA.")
            else:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X)

                pca = PCA(2).fit_transform(X)
                d = pd.DataFrame(
                    {"PC1": pca[:, 0], "PC2": pca[:, 1], "Cluster": labels.astype(str)}
                )

                fig = px.scatter(d, x="PC1", y="PC2", color="Cluster")
                st.plotly_chart(fig, use_container_width=True)


# ======================================================
# 5️⃣ AI MENTOR
# ======================================================
with tab5:
    st.header("🧠 AI Dataset Mentor")

    @st.cache_resource(show_spinner="Loading AI model...")
    def load_model():
        # Agar machine handle kar paaye toh 'google/flan-t5-base' bhi try kar sakti ho
        return hf_pipeline("text2text-generation", model="google/flan-t5-small")

    def smart_answer(question: str, df: pd.DataFrame, dataset_name: str):
        q_clean = question.strip()
        if not q_clean:
            st.warning("Type a question.")
            return

        q_low = q_clean.lower()

        # 🔎 check kaun-kaun se columns naam se mention hue hain
        mentioned_cols = [
            c for c in df.columns
            if c.lower() in q_low
        ]
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # ------------------ 1) PURE PANDAS ANSWERS ------------------ #

        # 1️⃣ list of columns / column names
        if "column" in q_low and any(
            w in q_low for w in ["list", "name", "names", "all"]
        ):
            st.subheader("📋 Columns in this dataset")
            st.write(list(df.columns))
            return

        # 2️⃣ shape / rows / columns / size
        if (
            "shape" in q_low
            or "size" in q_low
            or "row" in q_low
            or "rows" in q_low
            or "record" in q_low
            or "records" in q_low
        ):
            st.subheader("📐 Dataset shape")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            return

        # 3️⃣ dtypes / numeric / categorical columns
        if (
            "dtype" in q_low
            or "data type" in q_low
            or "types of column" in q_low
            or "numerical column" in q_low
            or "numeric column" in q_low
            or "categorical column" in q_low
        ):
            st.subheader("🧬 Column data types")
            st.dataframe(df.dtypes.to_frame("dtype"))

            num = df.select_dtypes(include=[np.number]).columns.tolist()
            cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
            st.markdown(
                f"- **Numeric columns:** {num or 'None'}  \n"
                f"- **Categorical columns:** {cat or 'None'}"
            )
            return

        # 4️⃣ missing values / nulls
        if "missing" in q_low or "null" in q_low or "nan" in q_low or "na" in q_low:
            st.subheader("⚠️ Missing values per column")
            st.write(df.isnull().sum())
            st.info(
                "All zeros ka matlab koi missing values nahi hain. "
                "Agar kisi column ka value > 0 hai to wahan cleaning ki zarurat hai."
            )
            return

        # 5️⃣ unique values of a column
        if "unique" in q_low or "distinct" in q_low:
            if mentioned_cols:
                for c in mentioned_cols:
                    st.subheader(f"🔹 Unique values in `{c}`")
                    st.write(df[c].unique()[:50])  # 50 tak dikhayenge
            else:
                st.warning("Kaunse column ke unique values chahiye, uska naam question me likho.")
            return

        # 6️⃣ basic stats: mean / max / min / average
        if any(w in q_low for w in ["mean", "average", "avg", "maximum", "minimum", "max", "min"]):
            if not mentioned_cols:
                st.subheader("📊 Summary stats for numeric columns")
                st.write(df[num_cols].describe().T)
                return
            else:
                for c in mentioned_cols:
                    if c in num_cols:
                        st.subheader(f"📊 Stats for `{c}`")
                        st.write(df[c].describe())
                    else:
                        st.warning(f"Column `{c}` numeric nahi hai, stats nikalna meaningful nahi hoga.")
                return

        # 7️⃣ correlation between two numeric columns
        if ("correlation" in q_low or "relation" in q_low or "relationship" in q_low) and len(mentioned_cols) >= 2:
            cols = [c for c in mentioned_cols if c in num_cols]
            if len(cols) >= 2:
                a, b = cols[0], cols[1]
                corr = df[[a, b]].corr().iloc[0, 1]
                st.subheader(f"📈 Correlation between `{a}` and `{b}`")
                st.write(f"Correlation: **{corr:.3f}**")
            else:
                st.warning("Dono columns numeric hone chahiye correlation ke liye.")
            return

        # 8️⃣ overall summary / high-level explanation
        if (
            "what is this dataset" in q_low
            or "about this dataset" in q_low
            or "overview" in q_low
            or "summarize" in q_low
            or "summary" in q_low
        ):
            st.subheader("📝 Quick dataset summary")
            st.write(describe_dataset(df, dataset_name))
            return

        # ------------------ 2) FALLBACK TO LLM ------------------ #

        with st.spinner("AI Thinking..."):
            summary = describe_dataset(df, dataset_name)
            model = load_model()
            prompt = f"""
You are a helpful data analyst assistant.

Here is a summary of a pandas DataFrame named '{dataset_name}':

{summary}

User question about the dataset:
{q_clean}

Using only the information that can be logically inferred from the summary,
give a clear answer in simple 5–7 sentences. 
If something cannot be known exactly from the data, say that it is not certain
and answer in a general way.
"""
            ans = model(prompt, max_new_tokens=220)[0]["generated_text"]
            st.write(ans)

    # ---------- UI ----------
    q = st.text_area("Ask anything about the dataset:", height=90)

    if st.button("Explain"):
        smart_answer(q, df, dataset_name)
