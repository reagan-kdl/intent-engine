# app.py
from __future__ import annotations

import os
import json
import base64
import re
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple

import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

# Try/Except block to handle missing local modeling.py gracefully
try:
    from modeling import (
        train_model,
        score_leads,
        explain_single_lead,
        generate_demo_dataset,
        roc_points,
    )
    HAS_MODELING = True
except ImportError:
    HAS_MODELING = False

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Lead Intent Probability Engine",
    layout="wide",
)

# -----------------------------
# KDL Brand palette
# -----------------------------
KDL = {
    "purple": "#825BA9",      # primary purple
    "gold": "#DBCA72",        # gold accent
    "deep_purple": "#6F189E", # deep purple (rich)
    "light_blue": "#8FA1F7",  # light blue (accent)
    "navy": "#10289E",        # navy (anchor)
    "white": "#FFFFFF",
    # UI tints
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(219,202,114,0.12)",
    "text": "rgba(255,255,255,0.92)",
    "muted": "rgba(255,255,255,0.72)",
    "surface": "rgba(255,255,255,0.055)",
    "surface2": "rgba(255,255,255,0.035)",
    "border": "rgba(219,202,114,0.22)",
    "success": "#28a745",
    "warning": "#ffc107",
}

# -----------------------------
# Plotly defaults + KDL styling
# -----------------------------
px.defaults.template = "plotly_dark"

def _kdl_style_fig(fig, title: Optional[str] = None):
    """Apply KDL styling consistently across Plotly charts."""
    if title is not None:
        fig.update_layout(title=title)

    fig.update_layout(
        paper_bgcolor=KDL["bg"],
        plot_bgcolor=KDL["bg"],
        font=dict(
            family="Manrope, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial",
            color=KDL["text"],
        ),
        title_font=dict(size=16, color=KDL["text"]),
        colorway=[KDL["gold"], KDL["light_blue"], KDL["purple"], KDL["deep_purple"], KDL["navy"]],
        legend=dict(font=dict(color=KDL["muted"])),
        margin=dict(l=10, r=10, t=46, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=KDL["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=KDL["grid"], zeroline=False)
    return fig


# -----------------------------
# Brand font: Manrope (KDL)
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="st"], [class*="css"], * {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif !important;
    }

    h1, h2, h3, h4 {
        letter-spacing: -0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# KDL Header CSS
# -----------------------------
st.markdown(
    f"""
    <style>
    .kdl-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 18px 18px;
        border-radius: 18px;
        border: 1px solid rgba(219, 202, 114, 0.18);
        background: linear-gradient(
            135deg,
            rgba(111, 24, 158, 0.18) 0%,
            rgba(130, 91, 169, 0.14) 32%,
            rgba(16, 40, 158, 0.10) 60%,
            rgba(255,255,255,0.02) 100%
        );
        box-shadow: 0 14px 38px rgba(0,0,0,0.25);
        margin-bottom: 10px;
    }}
    .kdl-header__left {{
        display: flex;
        align-items: center;
        gap: 14px;
        min-width: 0;
    }}
    .kdl-logo {{
        height: 42px;
        display: flex;
        align-items: center;
    }}
    .kdl-logo img {{
        height: 42px;
        width: auto;
        display: block;
    }}
    .kdl-header__brand {{
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: rgba(255,255,255,0.96);
        line-height: 1.2;
        white-space: nowrap;
    }}
    .kdl-header__tagline {{
        margin-top: 2px;
        font-size: 13px;
        font-weight: 650;
        color: rgba(219, 202, 114, 0.95);
        opacity: 0.95;
    }}
    .kdl-header__sub {{
        margin: 0 2px 18px 2px;
        color: rgba(255,255,255,0.72);
        font-size: 14px;
    }}
    .kdl-pill {{
        display: inline-flex;
        align-items: center;
        padding: 8px 12px;
        border-radius: 999px;
        border: 1px solid rgba(219, 202, 114, 0.30);
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.92);
        font-weight: 800;
        font-size: 12px;
        letter-spacing: 0.02em;
        box-shadow: 0 10px 26px rgba(111, 24, 158, 0.16);
        white-space: nowrap;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# KDL Theme
# -----------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --kdl-purple: {KDL["purple"]};
        --kdl-gold: {KDL["gold"]};
        --kdl-deep: {KDL["deep_purple"]};
        --kdl-blue: {KDL["light_blue"]};
        --kdl-navy: {KDL["navy"]};
        --kdl-ink: #0B0F19;
        --kdl-surface: {KDL["surface"]};
        --kdl-surface2: {KDL["surface2"]};
        --kdl-border: {KDL["border"]};
    }}

    .stApp {{
        background:
          radial-gradient(900px 620px at 14% 10%, rgba(111, 24, 158, 0.18) 0%, rgba(11, 15, 25, 1) 58%),
          radial-gradient(780px 520px at 78% 18%, rgba(16, 40, 158, 0.14) 0%, rgba(11, 15, 25, 0) 60%),
          radial-gradient(640px 460px at 68% 78%, rgba(143, 161, 247, 0.07) 0%, rgba(11, 15, 25, 0) 65%)
          !important;
    }}

    section[data-testid="stSidebar"] {{
        background:
          linear-gradient(180deg, rgba(111, 24, 158, 0.10) 0%, rgba(11, 15, 25, 1) 80%) !important;
        border-right: 1px solid rgba(219, 202, 114, 0.18);
    }}

    .stButton > button {{
        border-radius: 12px !important;
        border: 1px solid rgba(219, 202, 114, 0.30) !important;
        background: linear-gradient(135deg, rgba(111, 24, 158, 0.95), rgba(130, 91, 169, 0.92)) !important;
        color: white !important;
        font-weight: 800 !important;
        padding: 0.6rem 1rem !important;
        transition: transform 120ms ease, box-shadow 120ms ease, filter 120ms ease;
        box-shadow: 0 10px 26px rgba(111, 24, 158, 0.22);
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.04);
        box-shadow: 0 12px 30px rgba(219, 202, 114, 0.14);
    }}
    .stButton > button:active {{
        transform: translateY(0px);
        filter: brightness(0.98);
    }}

    div[data-baseweb="select"] > div,
    .stTextInput input,
    .stNumberInput input {{
        border-radius: 12px !important;
        background: var(--kdl-surface) !important;
        border: 1px solid rgba(219, 202, 114, 0.16) !important;
    }}

    details, .stExpander {{
        border-radius: 14px !important;
        border: 1px solid rgba(219, 202, 114, 0.14) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }}

    .stDataFrame, div[data-testid="stDataFrame"] {{
        border-radius: 14px !important;
        border: 1px solid rgba(219, 202, 114, 0.12) !important;
        overflow: hidden;
    }}

    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(219, 202, 114, 0.12);
        border-radius: 14px;
        padding: 14px 14px 10px 14px;
    }}

    a {{ color: var(--kdl-gold) !important; }}
    a:hover {{ color: rgba(219, 202, 114, 0.92) !important; }}

    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(219, 202, 114, 0.28), rgba(219, 202, 114, 0.00));
        opacity: 0.8;
    }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-thumb {{
        background: rgba(111, 24, 158, 0.40);
        border-radius: 999px;
        border: 2px solid rgba(11, 15, 25, 0.60);
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(219, 202, 114, 0.35); }}

    div[data-baseweb="slider"] > div > div {{
        color: var(--kdl-gold) !important;
    }}
    div[data-baseweb="slider"] [data-testid="stTickBar"] > div {{
        background: rgba(219, 202, 114, 0.35) !important;
    }}
    div[data-baseweb="slider"] [role="slider"] {{
        box-shadow: 0 0 0 2px rgba(219, 202, 114, 0.35) !important;
        border: 1px solid rgba(219, 202, 114, 0.55) !important;
    }}
    input[type="checkbox"], input[type="radio"] {{
        accent-color: {KDL["gold"]};
    }}
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, rgba(219, 202, 114, 0.95), rgba(111, 24, 158, 0.85)) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Helpers
# -----------------------------
def _download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _read_svg_as_data_uri(path: str) -> Optional[str]:
    """Return an SVG as a data URI for embedding in HTML (works well in Streamlit)."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/svg+xml;base64,{b64}"
    except Exception:
        return None


def _safe_hist(series: pd.Series, title: str):
    """Safely render histogram, handling empty/NaN data to avoid Canvas errors."""
    if series.empty or series.isna().all():
        st.caption(f"No data available for: {title}")
        return

    try:
        fig = px.histogram(series, nbins=30, title=title)
        _kdl_style_fig(fig, title=title)
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.caption(f"Could not render chart: {title}")


def _coef_bar(coef_df: pd.DataFrame, top_n: int = 15):
    """
    Improved visual: Diverging bar chart for coefficients.
    """
    tmp = coef_df.head(top_n).copy()
    tmp["direction"] = np.where(tmp["coef"] >= 0, "Increases intent", "Decreases intent")
    
    # Sort by magnitude but keep sign separation clear
    tmp = tmp.iloc[::-1] # Reverse for plotting top-down

    fig = px.bar(
        tmp,
        x="coef",
        y="feature",
        orientation="h",
        title=f"Top {top_n} Global Drivers",
        hover_data=["abs_coef", "direction"],
        color="direction",
        color_discrete_map={
            "Increases intent": KDL["gold"], 
            "Decreases intent": KDL["light_blue"]
        }
    )
    _kdl_style_fig(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _contrib_bar(contrib_df: pd.DataFrame):
    """
    Improved visual: Diverging bar chart for single lead contribution.
    """
    tmp = contrib_df.copy().iloc[::-1]
    
    fig = px.bar(
        tmp,
        x="contribution",
        y="feature",
        orientation="h",
        title="Top Drivers for this Lead",
        hover_data=["direction"],
        color="direction",
        color_discrete_map={
            "Increases intent": KDL["gold"], 
            "Decreases intent": KDL["light_blue"]
        }
    )
    _kdl_style_fig(fig)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def _tier_badge(tier: str) -> str:
    tier = str(tier)
    if tier == "Hot":
        return "🔥 Hot"
    if tier == "Warm":
        return "🟠 Warm"
    return "🔵 Cold"


def _extract_pipeline_features(pipeline) -> Tuple[List[str], List[str]]:
    """
    Robustly extract list of features from the pipeline's Preprocessor.
    Looks at both transformer names AND types to correctly ID numeric vs categorical.
    """
    try:
        preprocessor = pipeline.named_steps.get("preprocessor")
        if not preprocessor:
            return [], []
        
        num_cols = []
        cat_cols = []
        
        for name, trans, cols in preprocessor.transformers_:
            # Skip 'remainder' or dropped columns
            if name in ("remainder", "drop"):
                continue
                
            # Logic: Check name, then check transformer type string
            name_lower = str(name).lower()
            trans_str = str(trans).lower()
            
            is_num = False
            is_cat = False
            
            # Heuristic 1: Names
            if any(x in name_lower for x in ["num", "scal", "cont"]):
                is_num = True
            elif any(x in name_lower for x in ["cat", "enc", "ord"]):
                is_cat = True
                
            # Heuristic 2: Types (stronger)
            if any(x in trans_str for x in ["standardscaler", "minmax", "robust", "polynomial"]):
                is_num = True
            if any(x in trans_str for x in ["onehot", "ordinal"]):
                is_cat = True
            
            # Assignment
            if is_num:
                num_cols.extend(cols)
            elif is_cat:
                cat_cols.extend(cols)
            else:
                # If ambiguous, add to cat to avoid median imputation on potential strings
                cat_cols.extend(cols)
        
        return list(num_cols), list(cat_cols)
    except Exception:
        return [], []


def _data_quality_flag(row: pd.Series, numeric_cols: List[str]) -> str:
    missing = row.isna().mean()
    if missing >= 0.35:
        return "🟡 Sparse data"

    valid_numeric_cols = [c for c in numeric_cols if c in row.index]
    if not valid_numeric_cols:
        return "✅ OK"
    
    signals = 0
    for c in valid_numeric_cols:
        try:
            if float(row.get(c, 0)) > 0:
                signals += 1
        except:
            pass

    if len(valid_numeric_cols) > 5 and signals < (len(valid_numeric_cols) * 0.2):
        return "🟡 Low signal"

    # Recency check
    if "days_since_last_session" in row.index:
        try:
            d = float(row.get("days_since_last_session") or 999)
            if d >= 60:
                return "🟡 Stale"
        except Exception:
            pass

    return "✅ OK"


def _self_healing_align(df: pd.DataFrame, tr, label_col: str) -> pd.DataFrame:
    """
    First pass alignment: try to match what the pipeline SAYS it has.
    """
    out = df.copy()
    
    num_cols, cat_cols = _extract_pipeline_features(tr.pipeline)
    expected_features = set(getattr(tr, "feature_names", [])) | set(num_cols) | set(cat_cols)

    # Basic fill
    for c in expected_features:
        if c == label_col: 
            continue
        if c not in out.columns:
            if c in num_cols:
                out[c] = 0
            elif c in cat_cols:
                out[c] = "Unknown"
            else:
                # Fallback: If we don't know the type, use NaN so Imputers handle it safely
                out[c] = np.nan
    return out


def score_leads_safely(df_input: pd.DataFrame, tr, label_col: str) -> pd.DataFrame:
    """
    Wraps scoring in a try/except block to handle 'columns are missing' errors
    by automatically filling them and retrying.
    """
    df_aligned = _self_healing_align(df_input, tr, label_col)
    
    try:
        return score_leads(df_aligned, pipeline=tr.pipeline, label_col=label_col)
    except ValueError as e:
        err_msg = str(e)
        
        # 1. Handle "columns are missing" error (common sklearn)
        if "columns are missing" in err_msg:
            # Regex to extract the set content
            match = re.search(r"\{([^}]+)\}", err_msg)
            if match:
                missing_str = match.group(1)
                missing_cols = [c.strip().strip("'").strip('"') for c in missing_str.split(",")]
                
                st.toast(f"Self-healing: Filling {len(missing_cols)} missing columns.", icon="🛠️")
                
                # Fill missing cols
                num_cols, _ = _extract_pipeline_features(tr.pipeline)
                for c in missing_cols:
                    if c not in df_aligned.columns:
                        if c in num_cols:
                            df_aligned[c] = 0
                        else:
                            df_aligned[c] = np.nan
                
                # Retry
                return score_leads(df_aligned, pipeline=tr.pipeline, label_col=label_col)
        
        # 2. Handle "could not convert string to float" (imputation mismatch)
        if "could not convert string to float" in err_msg:
            st.toast("Self-healing: Fixing data type mismatches...", icon="🩹")
            # Try to force numeric columns to NaN if they contain "Unknown"
            num_cols, _ = _extract_pipeline_features(tr.pipeline)
            for c in num_cols:
                if c in df_aligned.columns:
                    # Replace "Unknown" with NaN in numeric columns so Imputer handles it
                    df_aligned[c] = df_aligned[c].replace("Unknown", np.nan)
            
            return score_leads(df_aligned, pipeline=tr.pipeline, label_col=label_col)

        # If it's a different error, raise it
        raise e


# -----------------------------
# Demo profile schema controls (with caching)
# -----------------------------
BUSINESS_TYPES = ["B2B", "B2C", "SaaS", "Ecommerce", "Home Services", "Professional Services"]
COMPLEXITIES = ["Basic (few signals)", "Standard", "Advanced (full journey)"]


def demo_fields_for_profile(business_type: str, complexity: str) -> Dict[str, List[str]]:
    base_id = ["lead_id", "created_at", "channel"]
    base_persona = ["industry", "job_title_seniority", "employee_size_bucket", "geo_state"]

    basic_signals = ["form_starts_30d", "form_submits_30d", "email_opens_30d", "email_clicks_30d", "days_since_last_session"]
    standard_signals = basic_signals + ["sessions_30d", "pageviews_30d", "pricing_page_views_30d", "service_page_views_30d", "case_study_views_30d"]
    advanced_signals = standard_signals + ["webinars_attended_90d", "content_downloads_90d", "return_visits_30d"]

    extra = []
    if business_type in ["Ecommerce"]:
        extra += ["cart_adds_30d", "checkouts_30d", "purchases_30d", "aov_estimate", "last_purchase_days"]
    if business_type in ["Home Services"]:
        extra += ["calls_30d", "call_missed_30d", "quote_requests_30d", "service_area_type"]
    if business_type in ["SaaS"]:
        extra += ["trial_starts_30d", "product_logins_30d", "seats_requested", "plan_interest"]
    if business_type in ["B2B", "Professional Services"]:
        extra += ["account_fit_score", "intent_topic_match", "inbound_vs_outbound"]

    if complexity == "Basic (few signals)":
        signals = basic_signals
        traits = base_id + ["industry", "geo_state", "job_title_seniority"]
    elif complexity == "Standard":
        signals = standard_signals
        traits = base_id + base_persona
    else:
        signals = advanced_signals + extra
        traits = base_id + base_persona + ["company_name", "company_domain"]

    label = ["converted"]

    return {
        "Identity / firmographic": traits,
        "Engagement / behavioral": signals,
        "Outcome label (demo)": label,
    }


def apply_demo_profile(df: pd.DataFrame, business_type: str, complexity: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    out = df.copy()

    # Ensure a few optional columns exist
    for col, default in [
        ("industry", "General"),
        ("job_title_seniority", "IC"),
        ("employee_size_bucket", "1-10"),
        ("geo_state", "IL"),
        ("service_page_views_30d", 0),
        ("case_study_views_30d", 0),
        ("pricing_page_views_30d", 0),
        ("pageviews_30d", 0),
        ("sessions_30d", 0),
    ]:
        if col not in out.columns:
            out[col] = default

    if business_type == "Ecommerce":
        out["cart_adds_30d"] = rng.poisson(lam=1.2, size=len(out))
        out["checkouts_30d"] = rng.poisson(lam=0.6, size=len(out))
        out["purchases_30d"] = rng.poisson(lam=0.25, size=len(out))
        out["aov_estimate"] = np.clip(rng.normal(90, 35, size=len(out)), 10, 400).round(2)
        out["last_purchase_days"] = np.clip(rng.normal(45, 30, size=len(out)), 0, 365).round(0)

    elif business_type == "Home Services":
        out["calls_30d"] = rng.poisson(lam=0.9, size=len(out))
        out["call_missed_30d"] = rng.poisson(lam=0.25, size=len(out))
        out["quote_requests_30d"] = rng.poisson(lam=0.35, size=len(out))
        out["service_area_type"] = rng.choice(["Urban", "Suburban", "Rural"], size=len(out), p=[0.35, 0.5, 0.15])

    elif business_type == "SaaS":
        out["trial_starts_30d"] = rng.poisson(lam=0.35, size=len(out))
        out["product_logins_30d"] = rng.poisson(lam=2.0, size=len(out))
        out["seats_requested"] = rng.choice([1, 3, 5, 10, 25, 50], size=len(out), p=[0.25, 0.2, 0.2, 0.18, 0.12, 0.05])
        out["plan_interest"] = rng.choice(["Starter", "Pro", "Enterprise"], size=len(out), p=[0.5, 0.35, 0.15])

    else:
        out["account_fit_score"] = np.clip(rng.normal(65, 18, size=len(out)), 0, 100).round(0)
        out["intent_topic_match"] = np.clip(rng.normal(0.45, 0.22, size=len(out)), 0, 1).round(3)
        out["inbound_vs_outbound"] = rng.choice(["Inbound", "Outbound"], size=len(out), p=[0.6, 0.4])

    if complexity == "Basic (few signals)":
        keep = []
        fields = demo_fields_for_profile(business_type, complexity)
        for _, cols in fields.items():
            keep.extend(cols)
        keep = [c for c in keep if c in out.columns]
        out = out[keep].copy()

    elif complexity == "Standard":
        drop = [c for c in ["webinars_attended_90d", "content_downloads_90d", "return_visits_30d"] if c in out.columns]
        if drop:
            out = out.drop(columns=drop)

    else:
        if "webinars_attended_90d" not in out.columns:
            out["webinars_attended_90d"] = rng.poisson(lam=0.25, size=len(out))
        if "content_downloads_90d" not in out.columns:
            out["content_downloads_90d"] = rng.poisson(lam=0.35, size=len(out))
        if "return_visits_30d" not in out.columns:
            out["return_visits_30d"] = rng.poisson(lam=1.1, size=len(out))
        if "company_name" not in out.columns:
            out["company_name"] = [f"Company {i % 250:03d}" for i in range(len(out))]
        if "company_domain" not in out.columns:
            out["company_domain"] = [f"company{i % 250:03d}.com" for i in range(len(out))]

    return out

# Wrapper for caching demo data generation
@st.cache_data
def get_demo_data(n, seed, boost, b_type, complexity):
    raw = generate_demo_dataset(n, seed, boost)
    return apply_demo_profile(raw, b_type, complexity, seed)

# Wrapper for caching CSV loading
@st.cache_data
def load_csv_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


# -----------------------------
# Model persistence
# -----------------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_engine_v1.joblib")


def _ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(train_result, train_meta: dict):
    _ensure_model_dir()
    bundle = {
        "train_result": train_result,
        "train_meta": train_meta,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)


def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Lead Intent Probability Engine")
st.sidebar.caption("A portfolio-grade probability model + explainability + CRM-ready output.")

if not HAS_MODELING:
    st.error("Missing `modeling.py`. Please ensure it is in the same directory.")
    st.stop()

mode = st.sidebar.radio("Data source", ["Generate demo dataset", "Upload CSV"], index=0)

with st.sidebar.expander("Model settings", expanded=True):
    test_size = st.slider("Test size (for evaluation)", 0.1, 0.4, 0.2, 0.05)
    use_time_split = st.checkbox("Use time-based split (recommended)", value=True)
    time_col = st.text_input("Time column name", value="created_at")
    label_col = st.text_input("Label column name", value="converted")
    top_k_explain = st.slider("Per-lead explanations (top K factors)", 3, 12, 6, 1)

    st.markdown("---")
    st.caption("Tier thresholds")
    cold_max = st.slider("Cold → Warm cutoff", 0.05, 0.70, 0.30, 0.01)
    warm_max = st.slider("Warm → Hot cutoff", 0.10, 0.95, 0.60, 0.01)
    if warm_max <= cold_max:
        st.warning("Warm cutoff must be greater than Cold cutoff.")

st.sidebar.divider()
st.sidebar.subheader("Model persistence")

loaded_bundle = load_model()
has_saved_model = loaded_bundle is not None

use_saved = st.sidebar.checkbox(
    "Use saved model (if available)",
    value=True if has_saved_model else False,
    help="Loads the last saved trained model from /models.",
)

save_after_train = st.sidebar.checkbox(
    "Save model after training",
    value=True,
    help="Writes the trained model to /models so it persists between runs.",
)

if has_saved_model:
    saved_meta = loaded_bundle.get("train_meta", {})
    st.sidebar.caption(f"Saved model found: `{MODEL_PATH}`")
    st.sidebar.caption(f"Rows: {saved_meta.get('rows_trained', '?')}")
else:
    st.sidebar.caption("No saved model found yet.")

st.sidebar.divider()

# Logo control
st.sidebar.subheader("Branding")
use_logo = st.sidebar.checkbox("Use KDL logo in header", value=True)
logo_path = st.sidebar.text_input("Logo path", value="assets/kdl_long_text.svg")

# -----------------------------
# Load data
# -----------------------------
demo_profile = None
demo_fields_expected = None

if mode == "Generate demo dataset":
    st.sidebar.subheader("Demo profile")

    if "demo_applied" not in st.session_state:
        st.session_state.demo_applied = {
            "business_type": "SaaS",
            "crm_complexity": "Standard",
            "n": 1200,
            "seed": 7,
            "intent_boost": 0.35,
        }

    with st.sidebar.form("demo_profile_form"):
        business_type = st.selectbox(
            "Business type",
            BUSINESS_TYPES,
            index=BUSINESS_TYPES.index(st.session_state.demo_applied["business_type"]),
        )
        crm_complexity = st.selectbox(
            "CRM / data complexity",
            COMPLEXITIES,
            index=COMPLEXITIES.index(st.session_state.demo_applied["crm_complexity"]),
            help="Basic = few signals; Standard = adds web engagement; Advanced = full journey + business-specific fields.",
        )
        n = st.number_input(
            "Demo rows",
            min_value=200,
            max_value=5000,
            value=int(st.session_state.demo_applied["n"]),
            step=100,
        )
        seed = st.number_input(
            "Random seed",
            min_value=1,
            max_value=9999,
            value=int(st.session_state.demo_applied["seed"]),
            step=1,
        )
        intent_boost = st.slider(
            "Demo: add more warm/hot leads",
            0.0,
            1.0,
            float(st.session_state.demo_applied["intent_boost"]),
            0.05,
        )
        apply_demo = st.form_submit_button("✅ Apply demo profile", type="primary")

    if apply_demo:
        st.session_state.demo_applied = {
            "business_type": business_type,
            "crm_complexity": crm_complexity,
            "n": int(n),
            "seed": int(seed),
            "intent_boost": float(intent_boost),
        }
        st.sidebar.success("Applied ✅")

    applied = st.session_state.demo_applied
    st.sidebar.caption(
        f"**Active:** {applied['business_type']} • {applied['crm_complexity']} • {applied['n']:,} rows • seed {applied['seed']}"
    )

    # Use cached function
    df = get_demo_data(
        int(applied["n"]),
        int(applied["seed"]),
        float(applied["intent_boost"]),
        applied["business_type"],
        applied["crm_complexity"]
    )

    demo_profile = {"business_type": applied["business_type"], "crm_complexity": applied["crm_complexity"]}
    demo_fields_expected = demo_fields_for_profile(applied["business_type"], applied["crm_complexity"])

else:
    uploaded = st.sidebar.file_uploader("Upload leads CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to begin, or switch to the demo dataset.")
        st.stop()
    # Use cached function
    df = load_csv_data(uploaded)

# Clean up optional datetime column
if time_col in df.columns:
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception:
        pass

# -----------------------------
# Header
# -----------------------------
logo_uri = _read_svg_as_data_uri(logo_path) if use_logo else None
if use_logo and (logo_uri is None):
    st.sidebar.warning(f"Logo not found at: {logo_path} (header will show text instead)")

left_html = ""
if use_logo and logo_uri is not None:
    left_html = f"""
      <div class="kdl-logo"><img src="{logo_uri}" alt="King Data Lab"/></div>
      <div>
        <div class="kdl-header__tagline">We let the data do the talking.</div>
      </div>
    """
else:
    left_html = """
      <div>
        <div class="kdl-header__brand">King Data Lab</div>
        <div class="kdl-header__tagline">We let the data do the talking.</div>
      </div>
    """

st.markdown(
    f"""
    <div class="kdl-header">
      <div class="kdl-header__left">
        {left_html}
      </div>
      <div class="kdl-header__right">
        <span class="kdl-pill">Intent Engine v1</span>
      </div>
    </div>
    <div class="kdl-header__sub">Upload leads → train → score → explain → export</div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# What fields are included (demo only)
# -----------------------------
if mode == "Generate demo dataset" and demo_fields_expected:
    with st.expander("What fields are included (based on your demo profile)", expanded=True):
        st.write(f"**Business type:** {demo_profile['business_type']}  •  **Data complexity:** {demo_profile['crm_complexity']}")
        for group, cols in demo_fields_expected.items():
            cols_present = [c for c in cols if c in df.columns]
            st.markdown(f"**{group}** ({len(cols_present)} fields)")
            st.code(", ".join(cols_present), language="text")
        st.caption("Tip: If you switch profiles and you’re using a saved model, retrain for the cleanest demo results.")

# -----------------------------
# Data preview
# -----------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    st.subheader("Data preview")
    st.dataframe(df.head(25), use_container_width=True, height=330)

with c2:
    st.subheader("Quick stats")
    st.metric("Rows", f"{len(df):,}")
    st.metric("Columns", f"{df.shape[1]}")
    has_label = label_col in df.columns
    st.metric("Has label column?", "Yes" if has_label else "No")
    if has_label:
        try:
            rate = float(df[label_col].astype(int).mean())
        except Exception:
            rate = float(pd.to_numeric(df[label_col], errors="coerce").fillna(0).clip(0, 1).mean())
        st.metric("Conversion rate", f"{rate*100:.1f}%")

with c3:
    st.subheader("Missingness")
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(12)
    st.write(miss.rename("missing_%"))

st.divider()

# -----------------------------
# Train model
# -----------------------------
st.subheader("1) Train model")

btn1, btn2 = st.columns([1, 1])
with btn1:
    train_btn = st.button("Train / Re-train model", type="primary")
with btn2:
    clear_btn = st.button("Clear saved model")

if clear_btn:
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    st.session_state.pop("train_result", None)
    st.session_state.pop("train_meta", None)
    st.success("Cleared saved model.")
    st.stop()

if "train_result" not in st.session_state:
    st.session_state.train_result = None
if "train_meta" not in st.session_state:
    st.session_state.train_meta = None

if use_saved and st.session_state.train_result is None:
    bundle = load_model()
    if bundle is not None:
        st.session_state.train_result = bundle.get("train_result")
        st.session_state.train_meta = bundle.get("train_meta") or {
            "model_version": "intent-engine-v1",
            "loaded_from": MODEL_PATH,
            "loaded_at": datetime.now(timezone.utc).isoformat(),
        }

if train_btn or st.session_state.train_result is None:
    if not has_label:
        st.error(
            f"Your dataset needs a binary label column (0/1). Expected '{label_col}'. "
            "(Demo dataset includes it automatically.)"
        )
        st.stop()

    split_col = time_col if (use_time_split and time_col in df.columns) else None
    
    with st.spinner("Training calibrated model..."):
        tr = train_model(df, label_col=label_col, time_split_col=split_col, test_size=float(test_size))
    
    st.session_state.train_result = tr

    train_meta = {
        "model_version": "intent-engine-v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "rows_trained": int(len(df)),
        "num_features": int(len(tr.feature_names)),
        "label_col": label_col,
        "time_split_col": split_col,
        "metrics": tr.metrics,
    }
    st.session_state.train_meta = train_meta

    if save_after_train:
        save_model(tr, train_meta)
        st.toast("Model saved to /models", icon="💾")

tr = st.session_state.train_result
train_meta = st.session_state.train_meta or {}

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Brier score (lower is better)", f"{tr.metrics.get('brier', float('nan')):.4f}")
with m2:
    roc_auc = tr.metrics.get("roc_auc", None)
    st.metric("ROC AUC", f"{roc_auc:.3f}" if roc_auc is not None else "—")
with m3:
    st.metric("Model type", "Calibrated Logistic Regression")

with st.expander("Model metadata", expanded=False):
    st.json(train_meta)

with st.expander("Global drivers (feature coefficients)", expanded=True):
    _coef_bar(tr.coef_table, top_n=18)

st.divider()

# -----------------------------
# Score
# -----------------------------
st.subheader("2) Score leads")

num_cols_pipeline, _ = _extract_pipeline_features(tr.pipeline)

scored = score_leads_safely(df, tr=tr, label_col=label_col)

if "intent_probability" in scored.columns:
    cm = float(cold_max)
    wm = float(warm_max)
    if wm > cm:
        scored["intent_tier"] = pd.cut(
            scored["intent_probability"],
            bins=[-0.001, cm, wm, 1.001],
            labels=["Cold", "Warm", "Hot"],
        ).astype(str)
    else:
        scored["intent_tier"] = "Cold"

# Calculate data quality
scored["data_quality"] = scored.apply(lambda r: _data_quality_flag(r, num_cols_pipeline), axis=1)

tiers = ["Cold", "Warm", "Hot"]
selected_tiers = st.multiselect("Filter tiers", tiers, default=tiers)
filtered = scored[scored["intent_tier"].isin(selected_tiers)].copy()

filter_cols = []
for col in ["channel", "industry", "job_title_seniority", "employee_size_bucket", "geo_state"]:
    if col in filtered.columns:
        filter_cols.append(col)

if filter_cols:
    with st.expander("Additional filters", expanded=False):
        for col in filter_cols:
            options = sorted([x for x in filtered[col].dropna().unique().tolist()])
            pick = st.multiselect(f"{col}", options, default=options[: min(len(options), 6)])
            if pick:
                filtered = filtered[filtered[col].isin(pick)]

left, right = st.columns([1.2, 1])

with left:
    st.write("### Scored leads")

    filtered_display = filtered.copy()
    filtered_display["tier_badge"] = filtered_display["intent_tier"].apply(_tier_badge)

    key_cols = [
        c
        for c in [
            "lead_id",
            "tier_badge",
            "intent_probability",
            "intent_tier",
            "data_quality",
            "recommended_action",
        ]
        if c in filtered_display.columns
    ]
    remaining = [c for c in filtered_display.columns if c not in key_cols]
    filtered_display = filtered_display[key_cols + remaining]

    st.dataframe(
        filtered_display.sort_values("intent_probability", ascending=False).head(500),
        use_container_width=True,
        height=420,
    )

    st.download_button(
        "Download scored CSV",
        data=_download_csv(filtered),
        file_name="scored_leads.csv",
        mime="text/csv",
    )

with right:
    st.write("### Visuals")

    # Safety guard: ensure we don't plot empty data to prevent Canvas errors
    if not filtered.empty:
        if "intent_probability" in filtered.columns:
            _safe_hist(filtered["intent_probability"], "Intent probability distribution")

        tier_counts = (
            filtered["intent_tier"].value_counts().reindex(["Cold", "Warm", "Hot"]).fillna(0).astype(int)
        )
        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            title="Lead count by tier",
            labels={"x": "Tier", "y": "Leads"},
        )
        tier_color_map = {"Cold": KDL["light_blue"], "Warm": KDL["gold"], "Hot": KDL["deep_purple"]}
        fig.update_traces(marker=dict(color=[tier_color_map.get(t, KDL["purple"]) for t in tier_counts.index]))
        _kdl_style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

        if label_col in filtered.columns:
            conv = filtered.groupby("intent_tier")[label_col].mean().reindex(["Cold", "Warm", "Hot"])
            conv = conv.fillna(0)
            
            fig2 = px.bar(
                x=conv.index,
                y=(conv.values * 100),
                title="Conversion rate by tier (if labels present)",
                labels={"x": "Tier", "y": "Conversion rate (%)"},
            )
            fig2.update_traces(marker=dict(color=[tier_color_map.get(t, KDL["purple"]) for t in conv.index]))
            _kdl_style_fig(fig2)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No leads selected. Adjust filters to see visualizations.")

# -----------------------------
# Action section: CRM Sync Preview
# -----------------------------
st.divider()
st.subheader("3) Take action")

sync_col1, sync_col2 = st.columns([1, 2])

with sync_col1:
    sync_tier = st.selectbox("Tier to sync", ["Hot", "Warm", "Cold"], index=0)
    max_sync = st.slider("Max leads", 5, 200, 50, 5)
    do_sync = st.button("🚀 Sync with CRM (Preview)", type="primary")

with sync_col2:
    st.markdown(
        """
**What this does**
- Builds a CRM-ready payload you could send to HubSpot/Salesforce
- Includes probability, tier, recommended action, and model version
- **No data is sent** (safe preview)
"""
    )

if do_sync:
    to_sync = (
        filtered[filtered["intent_tier"] == sync_tier]
        .sort_values("intent_probability", ascending=False)
        .head(int(max_sync))
    )

    payload = []
    for _, r in to_sync.iterrows():
        payload.append(
            {
                "lead_id": r.get("lead_id", None),
                "properties": {
                    "intent_probability": round(float(r["intent_probability"]), 4),
                    "intent_tier": str(r["intent_tier"]),
                    "recommended_action": str(r.get("recommended_action", "")),
                    "data_quality": str(r.get("data_quality", "")),
                    "model_version": str(train_meta.get("model_version", "intent-engine-v1")),
                    "trained_at": str(train_meta.get("trained_at", "")),
                },
            }
        )

    st.success(f"Prepared {len(payload)} lead records for CRM sync preview.")
    st.write("Preview (first 5):")
    st.json(payload[:5])

    st.download_button(
        "Download CRM payload (JSON)",
        data=json.dumps(payload, indent=2),
        file_name="crm_sync_payload.json",
        mime="application/json",
    )

st.divider()

# -----------------------------
# Per-lead explanation panel
# -----------------------------
st.subheader("4) Explain a lead (the ‘why’)")

id_col = "lead_id" if "lead_id" in filtered.columns else None

if filtered.empty:
    st.info("No leads available to explain. Adjust your filters above.")
else:
    if id_col:
        lead_ids = filtered[id_col].astype(str).tolist()
        # Limit to top 2000 to avoid UI lag
        selected_id = st.selectbox("Select lead", lead_ids[:2000])
        chosen = filtered[filtered[id_col].astype(str) == str(selected_id)].iloc[0]
    else:
        idx = st.number_input(
            "Row index to explain",
            min_value=0,
            max_value=max(0, len(filtered) - 1),
            value=0,
            step=1,
        )
        chosen = filtered.iloc[int(idx)]

    p = float(chosen["intent_probability"])
    tier = str(chosen["intent_tier"])

    a1, a2, a3 = st.columns([1, 1, 2])
    with a1:
        st.metric("Intent probability", f"{p*100:.1f}%")
    with a2:
        st.metric("Tier", _tier_badge(tier))
    with a3:
        st.write("**Recommended action**")
        st.write(str(chosen.get("recommended_action", "")))

    if "data_quality" in chosen.index:
        st.caption(f"Data quality: {chosen.get('data_quality')}")

    contrib = explain_single_lead(
        chosen,
        pipeline=tr.pipeline,
        top_k=int(top_k_explain),
        label_col=label_col,
    )
    _contrib_bar(contrib)

    with st.expander("Show selected lead raw fields", expanded=False):
        st.json(json.loads(pd.Series(chosen).to_json()))

# -----------------------------
# Evaluation plots
# -----------------------------
st.divider()
st.subheader("5) Evaluation (optional)")

if label_col in df.columns:
    X = _self_healing_align(df, tr, label_col).drop(columns=[label_col], errors="ignore")
    y = df[label_col].astype(int).values
    
    try:
        proba = tr.pipeline.predict_proba(X)[:, 1]
    except Exception:
        scored_eval = score_leads_safely(df, tr=tr, label_col=label_col)
        proba = scored_eval["intent_probability"].values

    tabs = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall", "Calibration"])

    # 1. Confusion Matrix (NEW)
    with tabs[0]:
        # Use the "Warm" threshold as the cutoff for positive prediction
        threshold = float(cold_max) # Any lead > cold is considered positive for this metric
        y_pred = (proba >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        # Simple custom heatmap
        z_text = [[str(y) for y in x] for x in cm]
        x_labels = ['Predicted Cold', 'Predicted Warm+']
        y_labels = ['Actual Non-Convert', 'Actual Convert']
        
        fig_cm = px.imshow(
            cm, 
            x=x_labels, 
            y=y_labels, 
            text_auto=True,
            color_continuous_scale=[KDL['navy'], KDL['light_blue'], KDL['gold']],
            title=f"Confusion Matrix (Threshold: {threshold:.2f})"
        )
        _kdl_style_fig(fig_cm)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("This shows how many leads were correctly classified based on your 'Cold/Warm' cutoff.")

    # 2. ROC
    with tabs[1]:
        if len(np.unique(y)) > 1:
            fpr, tpr = roc_points(y, proba)
            fig = px.line(x=fpr, y=tpr, title="ROC Curve (False Alarm vs Detection)")
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
            fig.update_traces(line=dict(width=3, color=KDL["gold"]))
            _kdl_style_fig(fig)
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig, use_container_width=True)

    # 3. Precision-Recall
    with tabs[2]:
        if len(np.unique(y)) > 1:
            prec, rec, _ = precision_recall_curve(y, proba)
            ap = average_precision_score(y, proba)
            fig_pr = px.line(x=rec, y=prec, title=f"Precision-Recall Curve (AP={ap:.2f})")
            fig_pr.update_traces(line=dict(width=3, color=KDL["deep_purple"]))
            _kdl_style_fig(fig_pr)
            fig_pr.update_layout(xaxis_title="Recall (Percent of converters found)", yaxis_title="Precision (Percent of flagged leads that convert)")
            st.plotly_chart(fig_pr, use_container_width=True)

    # 4. Calibration
    with tabs[3]:
        bins = pd.qcut(proba, 10, duplicates="drop")
        cal = (
            pd.DataFrame({"p": proba, "y": y})
            .groupby(bins)
            .agg(pred=("p", "mean"), actual=("y", "mean"))
            .reset_index(drop=True)
        )
        fig2 = px.line(cal, x="pred", y="actual", title="Calibration: Predicted vs Actual")
        fig2.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig2.update_traces(line=dict(width=3, color=KDL["light_blue"]))
        _kdl_style_fig(fig2)
        fig2.update_layout(xaxis_title="Predicted Probability", yaxis_title="Actual Conversion Rate")
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No labels found — evaluation charts require a label column (e.g., 'converted').")