"""
Gradsense
=========
A unified Streamlit application that:
  1. Trains a RandomForest risk-classification model on student data
     (upload your own CSV, or use the built-in realistic sample dataset).
  2. Predicts a risk level (Low / Medium / High) for individual students
     or in batch, via clean tabbed navigation.
  3. Generates plain-language, personalized explanations and study
     recommendations using a locally running Ollama LLM (llama3, mistral,
     phi3, etc.), with graceful fallback if Ollama isn't reachable.
  4. Provides interactive Plotly visualizations for data exploration,
     including a signature confidence gauge on individual predictions.

Run with:  streamlit run app.py

Requirements (see bottom of file for a copy-pasteable requirements.txt):
  streamlit, pandas, numpy, scikit-learn, plotly, requests, ollama
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# BRAND / CONFIG
# ============================================================
APP_NAME = "Gradsense"
APP_TAGLINE = "See academic risk before it becomes a report card."
APP_ICON = "🎓"

OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODELS = ["llama3", "mistral", "phi3", "phi4-mini"]

NUMERIC_FEATURES = ["study_hours", "attendance", "previous_score", "sleep_hours"]
CATEGORICAL_FEATURES = ["parental_education", "gender", "extra_curricular"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "final_score"
RISK_COL = "risk_level"
RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]

# Semantic, high-contrast palette — distinct from brand accents so
# "this is a risk signal" is never confused with "this is UI chrome".
RISK_COLORS = {
    "Low Risk": "#34D399",     # emerald
    "Medium Risk": "#FBBF24",  # amber
    "High Risk": "#F87171",    # coral / red
}
BRAND_TEAL = "#22D3EE"
BRAND_VIOLET = "#8B5CF6"
BRAND_INDIGO = "#6366F1"
TEXT_PRIMARY = "#E6EDF5"
TEXT_MUTED = "#94A3B8"

DISPLAY_NAMES = {
    "study_hours": "Study Hours/Week",
    "attendance": "Attendance (%)",
    "previous_score": "Previous Score",
    "sleep_hours": "Sleep Hours/Night",
    "parental_education": "Parental Education",
    "gender": "Gender",
    "extra_curricular": "Extra-Curricular",
    "final_score": "Final Score",
    "risk_level": "Risk Level",
    "predicted_risk": "Predicted Risk",
    "confidence": "Confidence",
}

st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — mesh-gradient backdrop + glassmorphism + type system
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, .main-header h1 {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Mesh-gradient dark canvas across the whole app */
    [data-testid="stAppViewContainer"] {
        background-color: #0A0E1A;
        background-image:
            radial-gradient(circle at 12% 8%,  rgba(34, 211, 238, 0.16) 0%, transparent 42%),
            radial-gradient(circle at 88% 15%, rgba(139, 92, 246, 0.16) 0%, transparent 45%),
            radial-gradient(circle at 50% 95%, rgba(99, 102, 241, 0.14) 0%, transparent 50%);
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(10, 14, 26, 0.92);
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Header banner */
    .main-header {
        padding: 1.8rem 2.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(34,211,238,0.22) 0%, rgba(139,92,246,0.22) 100%);
        border: 1px solid rgba(255,255,255,0.10);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        color: #F8FAFC;
        margin-bottom: 1.6rem;
    }
    .main-header h1 { margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; }
    .main-header p { margin: 0.35rem 0 0 0; opacity: 0.85; font-size: 1rem; color: #CBD5E1; }

    /* Glass cards */
    .glass-card, .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        padding: 1.1rem 1.3rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }
    .metric-card { text-align: center; }
    .metric-card .value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.7rem; font-weight: 700; color: #F8FAFC;
    }
    .metric-card .label { font-size: 0.82rem; color: #94A3B8; margin-top: 0.15rem; }

    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        color: #0A0E1A;
        letter-spacing: 0.01em;
    }

    .status-pill {
        display: inline-block;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-online  { background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52,211,153,0.4); }
    .status-offline { background: rgba(248, 113, 113, 0.15); color: #F87171; border: 1px solid rgba(248,113,113,0.4); }

    .explain-card {
        background: rgba(139, 92, 246, 0.08);
        border-left: 4px solid #8B5CF6;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        line-height: 1.6;
        color: #E6EDF5;
    }

    .gradsense-caption { color: #94A3B8; font-size: 0.85rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.14);
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #22D3EE 0%, #6366F1 100%);
        border: none;
        color: #0A0E1A;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE INIT
# ============================================================
DEFAULT_STATE = {
    "df": None,
    "model": None,
    "encoders": {},
    "trained": False,
    "accuracy": None,
    "feature_importances": None,
    "ollama_status_checked": False,
    "ollama_online": False,
    "ollama_models": [],
    "explanation_cache": {},
    "last_prediction": None,
    "used_fallback_encoding": False,
}
for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ============================================================
# HELPERS
# ============================================================
def assign_risk_level(score):
    """Map a numeric final score to a risk tier."""
    if score >= 75:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    return "High Risk"


def generate_sample_data(n=500, seed=42):
    """Recreate a realistic synthetic dataset, used when the user has no
    CSV of their own."""
    rng = np.random.default_rng(seed)

    study_hours = rng.normal(5, 2, n).clip(0, 12)
    attendance = rng.normal(80, 10, n).clip(40, 100)
    previous_score = rng.normal(65, 15, n).clip(0, 100)
    sleep_hours = rng.normal(7, 1.5, n).clip(3, 10)
    parental_education = rng.choice(
        ["High School", "Bachelors", "Masters", "PhD"], n, p=[0.35, 0.35, 0.2, 0.1]
    )
    gender = rng.choice(["Male", "Female"], n)
    extra_curricular = rng.choice(["Yes", "No"], n, p=[0.4, 0.6])

    final_score = (
        0.35 * study_hours * 8
        + 0.25 * attendance
        + 0.30 * previous_score
        + 0.10 * sleep_hours * 5
        + rng.normal(0, 6, n)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours": study_hours.round(1),
            "attendance": attendance.round(1),
            "previous_score": previous_score.round(1),
            "sleep_hours": sleep_hours.round(1),
            "parental_education": parental_education,
            "gender": gender,
            "extra_curricular": extra_curricular,
            "final_score": final_score.round(1),
        }
    )
    df[RISK_COL] = df[TARGET_COL].apply(assign_risk_level)
    return df


def validate_columns(df):
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    has_target = TARGET_COL in df.columns or RISK_COL in df.columns
    return missing, has_target


def prepare_training_frame(df):
    """Ensure a risk_level column exists, derived from final_score if needed."""
    df = df.copy()
    if RISK_COL not in df.columns:
        if TARGET_COL in df.columns:
            df[RISK_COL] = df[TARGET_COL].apply(assign_risk_level)
        else:
            raise ValueError(
                f"Data needs either a '{RISK_COL}' column or a '{TARGET_COL}' "
                f"column to derive risk levels from."
            )
    return df


def train_model(df):
    """Encode categoricals, train a RandomForestClassifier, return everything
    needed for later single/batch predictions.

    BUGFIX: stratified splitting can still raise ValueError even when
    y.nunique() > 1, if any single class has only one member (stratify
    requires >= 2 per class). We now fall back to a plain, unstratified
    split instead of crashing the whole app.
    """
    df = prepare_training_frame(df)

    encoders = {}
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        X[col] = df[col].astype(float)
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    y = df[RISK_COL]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )
    except ValueError:
        # A class had too few members to stratify on — fall back safely.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=True
    )

    return model, encoders, acc, importances, df


def run_training(df, *, show_toast=True):
    """Single source of truth for (re)training the model and updating
    session state. Consolidates what used to be two separate, duplicated
    code paths (button click vs. auto-train-on-load)."""
    if df is None or df.empty:
        return False
    try:
        with st.spinner("Training model..."):
            model, encoders, acc, importances, trained_df = train_model(df)
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.accuracy = acc
        st.session_state.feature_importances = importances
        st.session_state.df = trained_df
        st.session_state.trained = True
        st.session_state.explanation_cache = {}
        if show_toast:
            st.toast(f"Model trained — {acc*100:.1f}% accuracy.", icon="🎯")
        return True
    except Exception as e:
        st.sidebar.error(f"Training failed: {e}")
        return False


def encode_features(row_dict, encoders):
    """Turn a dict of raw feature values into a single-row encoded DataFrame,
    matching the column order/encoding used at training time. Handles unseen
    categorical values by falling back to the first known class, and flags
    that a fallback occurred so the UI can surface it."""
    data = {}
    fallback_used = False
    for col in NUMERIC_FEATURES:
        data[col] = [float(row_dict[col])]
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        val = str(row_dict[col])
        if val in le.classes_:
            data[col] = [le.transform([val])[0]]
        else:
            data[col] = [0]
            fallback_used = True
    st.session_state.used_fallback_encoding = fallback_used
    return pd.DataFrame(data, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)


def check_ollama_status(force=False):
    if st.session_state.ollama_status_checked and not force:
        return st.session_state.ollama_online, st.session_state.ollama_models
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.5)
        if resp.status_code == 200:
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            st.session_state.ollama_online = True
            st.session_state.ollama_models = models
        else:
            st.session_state.ollama_online = False
            st.session_state.ollama_models = []
    except Exception:
        st.session_state.ollama_online = False
        st.session_state.ollama_models = []
    st.session_state.ollama_status_checked = True
    return st.session_state.ollama_online, st.session_state.ollama_models


def generate_ai_explanation(model_name, student_context, top_features):
    """Call the local Ollama instance for a plain-language explanation.
    Returns (success: bool, text_or_error: str)."""
    predicted = student_context.get("predicted_risk") if isinstance(student_context, dict) else ""
    prompt = f"""A student has the following profile:
{student_context}

A machine learning model classified this student's risk level as: {predicted}

The factors the model weighs most heavily overall are: {', '.join(top_features)}.

In 3-4 plain-language sentences for a non-technical teacher, explain why this
student likely received this risk classification, and suggest one or two
concrete, actionable next steps to help them. Be specific to their numbers,
not generic advice."""

    try:
        import ollama
        response = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        return True, response["message"]["content"]
    except Exception as e:
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
                timeout=60,
            )
            if resp.status_code == 200:
                return True, resp.json().get("message", {}).get("content", "")
            return False, f"Ollama returned status {resp.status_code}: {resp.text[:200]}"
        except Exception as e2:
            return False, str(e2) or str(e)


def metric_card(label, value):
    st.markdown(
        f"""<div class="metric-card">
                <div class="value">{value}</div>
                <div class="label">{label}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def risk_badge(risk):
    color = RISK_COLORS.get(risk, "#888")
    st.markdown(
        f'<span class="risk-badge" style="background:{color}">{risk}</span>',
        unsafe_allow_html=True,
    )


def confidence_gauge(confidence, risk):
    """Signature visual: a radial confidence gauge colored to match the
    predicted risk tier, so color and shape both reinforce the same signal."""
    color = RISK_COLORS.get(risk, BRAND_TEAL)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 34, "color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94A3B8"}},
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "rgba(255,255,255,0.04)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 100], "color": "rgba(255,255,255,0.06)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=10, b=10),
        height=220,
        font=dict(color="#E6EDF5"),
    )
    return fig


# ============================================================
# HEADER
# ============================================================
st.markdown(
    f"""
    <div class="main-header">
        <h1>{APP_ICON} {APP_NAME}</h1>
        <p>{APP_TAGLINE} Predictions and plain-language explanations, powered by a
        locally-running LLM via Ollama — no data ever leaves this machine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Data & Model")

    data_source = st.radio(
        "Training data source",
        ["Use sample dataset", "Upload my own CSV"],
        help="The sample dataset is synthetic but realistic, and lets you try "
        "the app immediately without your own file.",
    )

    uploaded_train_file = None
    if data_source == "Upload my own CSV":
        uploaded_train_file = st.file_uploader(
            "Training CSV",
            type=["csv"],
            help=f"Needs columns: {', '.join(FEATURE_COLS)}, plus either "
            f"'{TARGET_COL}' or '{RISK_COL}'.",
        )

    col_a, col_b = st.columns(2)
    with col_a:
        load_clicked = st.button("📂 Load Data", width="stretch")
    with col_b:
        train_clicked = st.button("🎯 Train Model", width="stretch", type="primary")

    st.markdown("---")
    st.header("🤖 Ollama Connection")

    online, available_models = check_ollama_status()
    status_html = (
        '<span class="status-pill status-online">● Online</span>'
        if online
        else '<span class="status-pill status-offline">● Offline</span>'
    )
    st.markdown(f"Status: {status_html}", unsafe_allow_html=True)

    if st.button("🔄 Re-check connection", width="stretch"):
        check_ollama_status(force=True)
        st.rerun()

    model_choices = available_models if available_models else DEFAULT_OLLAMA_MODELS
    ollama_model = st.selectbox("Ollama model", model_choices, index=0)

    if not online:
        st.caption(
            "Ollama isn't reachable at localhost:11434. Start it with `ollama serve`, "
            "then pull a model, e.g. `ollama pull llama3`."
        )

    st.markdown("---")
    st.caption(f"{APP_NAME} · Streamlit, scikit-learn, Plotly, and Ollama (local LLM).")

# ============================================================
# LOAD DATA
# ============================================================
if load_clicked:
    if data_source == "Use sample dataset":
        st.session_state.df = generate_sample_data()
        st.session_state.trained = False
        st.toast("Sample dataset loaded.", icon="✅")
    else:
        if uploaded_train_file is None:
            st.sidebar.error("Please upload a CSV first.")
        else:
            try:
                df_new = pd.read_csv(uploaded_train_file)
                missing, has_target = validate_columns(df_new)
                if missing:
                    st.sidebar.error(f"CSV is missing required columns: {', '.join(missing)}")
                elif not has_target:
                    st.sidebar.error(
                        f"CSV needs a '{TARGET_COL}' or '{RISK_COL}' column to train on."
                    )
                else:
                    st.session_state.df = df_new
                    st.session_state.trained = False
                    st.toast(f"Loaded {len(df_new)} records.", icon="✅")
            except Exception as e:
                st.sidebar.error(f"Couldn't read this file: {e}")

# Auto-load sample data on first run so the app isn't empty.
if st.session_state.df is None:
    st.session_state.df = generate_sample_data()

# ============================================================
# TRAIN MODEL — single consolidated path (button click or first-load)
# ============================================================
if train_clicked:
    run_training(st.session_state.df)
elif not st.session_state.trained and st.session_state.df is not None:
    run_training(st.session_state.df, show_toast=False)

df = st.session_state.df

# ============================================================
# TABS
# ============================================================
tab_overview, tab_explore, tab_predict, tab_batch = st.tabs(
    ["🏠 Overview", "📊 Data Exploration", "🎯 Predict a Student", "📋 Batch Predictions"]
)

# ---------------- OVERVIEW ----------------
with tab_overview:
    if df is None or df.empty:
        st.info("Load a dataset from the sidebar to get started.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Total Records", len(df))
        with c2:
            avg_score = df[TARGET_COL].mean() if TARGET_COL in df.columns else float("nan")
            metric_card("Avg Final Score", f"{avg_score:.1f}" if pd.notna(avg_score) else "–")
        with c3:
            if RISK_COL in df.columns:
                low_pct = (df[RISK_COL] == "Low Risk").mean() * 100
                metric_card("Low Risk Students", f"{low_pct:.0f}%")
            else:
                metric_card("Low Risk Students", "–")
        with c4:
            acc = st.session_state.accuracy
            metric_card("Model Accuracy", f"{acc*100:.1f}%" if acc is not None else "–")

        st.markdown("### Preview")
        preview_cols = [c for c in FEATURE_COLS + [TARGET_COL, RISK_COL] if c in df.columns]
        st.dataframe(df[preview_cols].head(10).rename(columns=DISPLAY_NAMES), width="stretch")

        if RISK_COL in df.columns:
            st.markdown("### Risk Distribution")
            counts = df[RISK_COL].value_counts().reindex(RISK_ORDER).fillna(0)
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map=RISK_COLORS,
                hole=0.55,
            )
            fig.update_traces(textfont_color="#0A0E1A", marker=dict(line=dict(color="#0A0E1A", width=2)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")

# ---------------- DATA EXPLORATION ----------------
with tab_explore:
    if df is None or df.empty:
        st.info("Load a dataset from the sidebar to explore it.")
    else:
        st.subheader("Target Distribution")
        if TARGET_COL in df.columns:
            fig = px.histogram(
                df, x=TARGET_COL, nbins=25, marginal="box",
                color_discrete_sequence=[BRAND_TEAL],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                xaxis_title=DISPLAY_NAMES.get(TARGET_COL, TARGET_COL),
                yaxis_title="Number of Students",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Correlation Heatmap")
        numeric_present = [c for c in NUMERIC_FEATURES + [TARGET_COL] if c in df.columns]
        if len(numeric_present) >= 2:
            corr = df[numeric_present].corr()
            fig = go.Figure(
                go.Heatmap(
                    z=corr.values,
                    x=[DISPLAY_NAMES.get(c, c) for c in corr.columns],
                    y=[DISPLAY_NAMES.get(c, c) for c in corr.index],
                    colorscale=[[0, "#F87171"], [0.5, "#0A0E1A"], [1, BRAND_TEAL]],
                    zmin=-1, zmax=1,
                    text=corr.round(2).values,
                    texttemplate="%{text}",
                    textfont=dict(color="#E6EDF5"),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Feature vs. Final Score")
        feature_pick = st.selectbox(
            "Choose a feature",
            [c for c in NUMERIC_FEATURES if c in df.columns],
            format_func=lambda c: DISPLAY_NAMES.get(c, c),
        )
        if feature_pick and TARGET_COL in df.columns:
            fig = px.scatter(
                df, x=feature_pick, y=TARGET_COL,
                color=RISK_COL if RISK_COL in df.columns else None,
                color_discrete_map=RISK_COLORS,
                trendline="ols",
                opacity=0.65,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                xaxis_title=DISPLAY_NAMES.get(feature_pick, feature_pick),
                yaxis_title=DISPLAY_NAMES.get(TARGET_COL, TARGET_COL),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_present and TARGET_COL in df.columns:
            st.subheader("Final Score by Category")
            cat_pick = st.selectbox(
                "Choose a categorical feature",
                cat_present,
                format_func=lambda c: DISPLAY_NAMES.get(c, c),
                key="cat_pick",
            )
            fig = px.box(df, x=cat_pick, y=TARGET_COL, color=cat_pick,
                         color_discrete_sequence=[BRAND_TEAL, BRAND_VIOLET, BRAND_INDIGO, "#34D399"])
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                showlegend=False,
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, width="stretch")

        if st.session_state.feature_importances is not None:
            st.subheader("What Drives the Model's Predictions?")
            imp = st.session_state.feature_importances.rename(index=DISPLAY_NAMES)
            fig = go.Figure(
                go.Bar(
                    x=imp.values, y=imp.index, orientation="h",
                    marker=dict(color=imp.values, colorscale=[[0, "#3B82F6"], [1, BRAND_TEAL]]),
                    text=[f"{v:.1%}" for v in imp.values],
                    textposition="outside",
                    textfont=dict(color="#E6EDF5"),
                )
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E6EDF5"),
                xaxis_title="Relative Importance",
                xaxis=dict(tickformat=".0%", gridcolor="rgba(148,163,184,0.15)"),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")

# ---------------- PREDICT A STUDENT ----------------
with tab_predict:
    if not st.session_state.trained:
        st.info("Train a model first (sidebar) to make predictions.")
    else:
        st.subheader("Enter Student Details")
        with st.form("single_predict_form"):
            f1, f2 = st.columns(2)
            with f1:
                study_hours = st.slider("Study Hours/Week", 0.0, 12.0, 5.0, 0.5)
                attendance = st.slider("Attendance (%)", 40.0, 100.0, 80.0, 1.0)
                previous_score = st.slider("Previous Score", 0.0, 100.0, 65.0, 1.0)
                sleep_hours = st.slider("Sleep Hours/Night", 3.0, 10.0, 7.0, 0.5)
            with f2:
                parental_education = st.selectbox(
                    "Parental Education", ["High School", "Bachelors", "Masters", "PhD"]
                )
                gender = st.selectbox("Gender", ["Male", "Female"])
                extra_curricular = st.selectbox("Extra-Curricular Activities", ["Yes", "No"])

            submitted = st.form_submit_button("🔮 Predict Risk Level", type="primary")

        if submitted:
            row = {
                "study_hours": study_hours,
                "attendance": attendance,
                "previous_score": previous_score,
                "sleep_hours": sleep_hours,
                "parental_education": parental_education,
                "gender": gender,
                "extra_curricular": extra_curricular,
            }
            X_row = encode_features(row, st.session_state.encoders)
            model = st.session_state.model
            pred = model.predict(X_row)[0]
            proba = model.predict_proba(X_row)[0]
            confidence = float(np.max(proba))

            st.session_state.last_prediction = {
                **row,
                "predicted_risk": pred,
                "confidence": confidence,
            }

        if st.session_state.last_prediction is not None:
            pred_info = st.session_state.last_prediction
            st.markdown("### Result")

            if st.session_state.used_fallback_encoding:
                st.caption(
                    "⚠️ One or more selections weren't seen during training — "
                    "the model used a safe fallback encoding for that field."
                )

            r1, r2 = st.columns([1, 1])
            with r1:
                risk_badge(pred_info["predicted_risk"])
                st.caption(f"Confidence: {pred_info['confidence']*100:.0f}%")
                st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.plotly_chart(
                    confidence_gauge(pred_info["confidence"], pred_info["predicted_risk"]),
                    width="stretch",
                )
                st.markdown('</div>', unsafe_allow_html=True)
            with r2:
                top_features = (
                    st.session_state.feature_importances.sort_values(ascending=False)
                    .index[:3].tolist()
                    if st.session_state.feature_importances is not None
                    else []
                )
                top_features_display = [DISPLAY_NAMES.get(f, f) for f in top_features]

                st.markdown("#### 💬 AI Explanation")
                if st.button("Generate AI Explanation", key="single_explain_btn"):
                    online, _ = check_ollama_status()
                    if not online:
                        st.warning(
                            "⚠️ Ollama isn't reachable right now, so an AI explanation "
                            "can't be generated. Make sure `ollama serve` is running and "
                            "a model is pulled (e.g. `ollama pull llama3`), then try again."
                        )
                    else:
                        with st.spinner(f"Asking {ollama_model} for an explanation..."):
                            ok, text = generate_ai_explanation(
                                ollama_model, pred_info, top_features_display
                            )
                        if ok:
                            st.markdown(f'<div class="explain-card">{text}</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"⚠️ Couldn't generate an explanation: {text}")

# ---------------- BATCH PREDICTIONS ----------------
with tab_batch:
    if not st.session_state.trained:
        st.info("Train a model first (sidebar) to run batch predictions.")
    else:
        st.subheader("Upload Students to Score")
        st.caption(f"CSV must contain columns: {', '.join(FEATURE_COLS)}")
        batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch_upload")

        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                missing = [c for c in FEATURE_COLS if c not in batch_df.columns]
                if missing:
                    st.error(f"CSV is missing required columns: {', '.join(missing)}")
                else:
                    model = st.session_state.model
                    encoders = st.session_state.encoders

                    rows = [encode_features(r.to_dict(), encoders) for _, r in batch_df.iterrows()]
                    X_batch = pd.concat(rows, ignore_index=True)

                    batch_df["predicted_risk"] = model.predict(X_batch)
                    batch_df["confidence"] = model.predict_proba(X_batch).max(axis=1)

                    st.success(f"Scored {len(batch_df)} students.")

                    def highlight_risk(val):
                        color = RISK_COLORS.get(val, "#888")
                        return f"color: {color}; font-weight: bold"

                    show_cols = [c for c in batch_df.columns if c in FEATURE_COLS] + [
                        "predicted_risk", "confidence"
                    ]
                    rendered_risk_col = DISPLAY_NAMES.get("predicted_risk", "predicted_risk")
                    styled = (
                        batch_df[show_cols]
                        .rename(columns=DISPLAY_NAMES)
                        .style.map(highlight_risk, subset=[rendered_risk_col])
                        .format({DISPLAY_NAMES.get("confidence", "confidence"): "{:.0%}"})
                    )
                    st.dataframe(styled, width="stretch")

                    counts = batch_df["predicted_risk"].value_counts().reindex(RISK_ORDER).fillna(0)
                    fig = px.bar(
                        x=counts.index, y=counts.values,
                        color=counts.index, color_discrete_map=RISK_COLORS,
                        labels={"x": "Risk Level", "y": "Number of Students"},
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E6EDF5"),
                        showlegend=False,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig, width="stretch")

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name="gradsense_risk_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Couldn't process this file: {e}")

# ============================================================
# requirements.txt (copy into a separate file if you don't have one)
# ------------------------------------------------------------
# streamlit
# pandas
# numpy
# scikit-learn
# plotly
# requests
# ollama
# ============================================================
