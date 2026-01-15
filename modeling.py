# modeling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


# -----------------------------
# Config
# -----------------------------

DEFAULT_LABEL_COLS = ["converted", "label", "converted_30d"]
DEFAULT_ID_COLS = ["lead_id", "id"]
DEFAULT_DATE_COLS = ["created_at", "lead_created_at"]

HIGH_INTENT_HINTS = {
    "pricing_page_views_30d": 2.5,
    "form_submits_30d": 3.0,
    "form_starts_30d": 1.5,
    "case_study_views_30d": 1.2,
    "email_clicks_30d": 1.1,
}


@dataclass
class TrainResult:
    pipeline: Pipeline
    feature_names: List[str]
    metrics: Dict[str, float]
    coef_table: pd.DataFrame


def _infer_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in DEFAULT_LABEL_COLS:
        if c in df.columns:
            return c
    return None


def _infer_id_column(df: pd.DataFrame) -> Optional[str]:
    for c in DEFAULT_ID_COLS:
        if c in df.columns:
            return c
    return None


def _make_preprocess_and_model(df: pd.DataFrame, y_col: str) -> Tuple[Pipeline, List[str]]:
    X = df.drop(columns=[y_col])

    # Identify numeric/categorical columns
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    base_lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # helps if conversions are rare
        solver="lbfgs",
    )

    # Use calibration to make probabilities more trustworthy (important for “probability engine” positioning)
    clf = CalibratedClassifierCV(base_lr, method="sigmoid", cv=3)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # Fit once to capture feature names
    pipe.fit(X, df[y_col].astype(int))

    # Extract feature names after preprocessing
    pre_fitted: ColumnTransformer = pipe.named_steps["pre"]
    feature_names = list(pre_fitted.get_feature_names_out())

    return pipe, feature_names


def train_model(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    time_split_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """
    Trains a calibrated logistic regression pipeline.
    If a time column is provided, uses time-based split (recommended for real lead data).
    """
    df = df.copy()

    y_col = label_col or _infer_label_column(df)
    if not y_col:
        raise ValueError("No label column found. Add a 'converted' (0/1) column or specify label_col.")

    # Ensure binary
    df[y_col] = df[y_col].astype(int).clip(0, 1)

    # Split
    if time_split_col and time_split_col in df.columns:
        tmp = df.sort_values(time_split_col)
        n_test = max(1, int(len(tmp) * test_size))
        train_df = tmp.iloc[:-n_test]
        test_df = tmp.iloc[-n_test:]
    else:
        # Random split fallback
        rng = np.random.default_rng(random_state)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n_test = max(1, int(len(df) * test_size))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    pipe, feature_names = _make_preprocess_and_model(train_df, y_col)

    # Evaluate
    X_test = test_df.drop(columns=[y_col])
    y_test = test_df[y_col].astype(int).values
    proba = pipe.predict_proba(X_test)[:, 1]

    metrics: Dict[str, float] = {}
    if len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    metrics["brier"] = float(brier_score_loss(y_test, proba))

    # Build coefficient table (approx: use underlying LR inside calibration)
    clf: CalibratedClassifierCV = pipe.named_steps["clf"]
    base_est = clf.calibrated_classifiers_[0].estimator

    coef = base_est.coef_.ravel()
    coef_table = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_table["abs_coef"] = coef_table["coef"].abs()
    coef_table = coef_table.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return TrainResult(pipeline=pipe, feature_names=feature_names, metrics=metrics, coef_table=coef_table)


def score_leads(
    df: pd.DataFrame,
    pipeline: Pipeline,
    label_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    df = df.copy()
    y_col = label_col or _infer_label_column(df)
    lead_id = id_col or _infer_id_column(df)

    X = df.drop(columns=[y_col]) if y_col and y_col in df.columns else df
    p = pipeline.predict_proba(X)[:, 1]

    out = df.copy()
    out["intent_probability"] = p
    out["intent_tier"] = pd.cut(
        out["intent_probability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Cold", "Warm", "Hot"],
    ).astype(str)

    out["recommended_action"] = out.apply(_recommended_action_row, axis=1)

    if lead_id and lead_id in out.columns:
        cols = [lead_id] + [c for c in out.columns if c != lead_id]
        out = out[cols]

    return out


def explain_single_lead(
    row: pd.Series,
    pipeline: Pipeline,
    top_k: int = 6,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Returns top positive/negative standardized contributions for one lead.
    Uses: (standardized feature value) * (LR coefficient)
    """
    y_col = label_col or _infer_label_column(pd.DataFrame([row]))
    X_row = row.drop(labels=[y_col]) if y_col and y_col in row.index else row

    pre: ColumnTransformer = pipeline.named_steps["pre"]
    clf: CalibratedClassifierCV = pipeline.named_steps["clf"]
    base_est = clf.calibrated_classifiers_[0].estimator

    Xt = pre.transform(pd.DataFrame([X_row]))
    feature_names = list(pre.get_feature_names_out())
    coef = base_est.coef_.ravel()

    contrib = Xt.ravel() * coef
    dfc = pd.DataFrame({"feature": feature_names, "contribution": contrib})
    dfc["abs_contribution"] = dfc["contribution"].abs()
    dfc = dfc.sort_values("abs_contribution", ascending=False).head(top_k).reset_index(drop=True)
    dfc["direction"] = np.where(dfc["contribution"] >= 0, "↑ increases intent", "↓ decreases intent")
    return dfc


def roc_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr


def _recommended_action_row(r: pd.Series) -> str:
    """
    Simple action rules that feel 'real'. You can customize these endlessly.
    """
    tier = r.get("intent_tier", "Cold")

    pricing = float(r.get("pricing_page_views_30d", 0) or 0)
    forms = float(r.get("form_submits_30d", 0) or 0)
    days_since = float(r.get("days_since_last_session", 999) or 999)
    email_clicks = float(r.get("email_clicks_30d", 0) or 0)

    if tier == "Hot":
        if forms >= 1 or pricing >= 2:
            return "Sales outreach ASAP (within 1 hour). Reference pricing/services viewed."
        if days_since <= 2:
            return "Sales outreach today. Personalize to pages/events from last session."
        return "Assign to SDR. 2-touch sequence (call + email) within 24 hours."

    if tier == "Warm":
        if pricing >= 1 and email_clicks >= 1:
            return "Send 1:1 email with relevant proof (case study) + soft CTA to book."
        if days_since <= 3:
            return "Enroll in intent-based nurture (service-specific) + retargeting."
        return "Nurture sequence + monitor for high-intent events."

    if days_since <= 7:
        return "Educational nurture (top-of-funnel). Suppress from SDR for now."
    return "Low-touch nurture + periodic re-engagement. Fix tracking/data if sparse."


def generate_demo_dataset(n: int = 1200, seed: int = 7, intent_boost: float = 0.0) -> pd.DataFrame:
    """
    Creates a realistic-ish demo dataset with a hidden intent factor.
    Great for portfolio demos without using client data.
    """
    rng = np.random.default_rng(seed)

    industries = ["Home Services", "SaaS", "Manufacturing", "Healthcare", "Professional Services", "Retail"]
    channels = ["Paid Search", "Organic", "Referral", "Paid Social", "Email", "Direct"]
    seniority = ["IC", "Manager", "Director", "VP", "C-Suite"]
    size_bucket = ["1-10", "11-50", "51-200", "200+"]

    latent_low = rng.beta(2, 6, size=n)
    latent_high = rng.beta(5, 2, size=n)
    b = float(np.clip(intent_boost, 0.0, 1.0))
    latent_intent = (1 - b) * latent_low + b * latent_high
    latent_intent = np.clip(latent_intent, 0, 1)

    df = pd.DataFrame(
        {
            "lead_id": [f"L{100000+i}" for i in range(n)],
            "channel": rng.choice(channels, size=n, p=[0.28, 0.25, 0.12, 0.12, 0.13, 0.10]),
            "industry": rng.choice(industries, size=n),
            "job_title_seniority": rng.choice(seniority, size=n, p=[0.20, 0.30, 0.25, 0.18, 0.07]),
            "employee_size_bucket": rng.choice(size_bucket, size=n, p=[0.25, 0.35, 0.25, 0.15]),
            "business_email": rng.binomial(1, 0.78, size=n),
            "sessions_30d": rng.poisson(lam=1 + 6 * latent_intent, size=n),
            "pageviews_30d": rng.poisson(lam=2 + 18 * latent_intent, size=n),
            "service_page_views_30d": rng.poisson(lam=0.8 + 6 * latent_intent, size=n),
            "case_study_views_30d": rng.poisson(lam=0.2 + 2.5 * latent_intent, size=n),
            "pricing_page_views_30d": rng.poisson(lam=0.1 + 2.8 * (latent_intent**1.4), size=n),
            "form_starts_30d": rng.poisson(lam=0.15 + 2.2 * (latent_intent**1.6), size=n),
            "form_submits_30d": rng.binomial(2, np.clip(0.03 + 0.45 * latent_intent, 0, 0.85), size=n),
            "email_opens_30d": rng.poisson(lam=0.4 + 3.0 * latent_intent, size=n),
            "email_clicks_30d": rng.binomial(3, np.clip(0.02 + 0.25 * latent_intent, 0, 0.6), size=n),
        }
    )

    df["days_since_last_session"] = np.clip((18 - 30 * latent_intent + rng.normal(0, 4, size=n)).round(), 0, 60).astype(
        int
    )

    fit_boost = (
        (df["job_title_seniority"].isin(["Director", "VP", "C-Suite"])).astype(int) * 0.10
        + (df["employee_size_bucket"].isin(["51-200", "200+"])).astype(int) * 0.06
        + (df["industry"].isin(["Home Services", "Professional Services", "SaaS"])).astype(int) * 0.05
    )

    behavior_boost = (
        0.10 * np.tanh(df["pricing_page_views_30d"] / 2)
        + 0.15 * np.tanh(df["form_submits_30d"])
        + 0.08 * np.tanh(df["email_clicks_30d"])
        - 0.06 * np.tanh(df["days_since_last_session"] / 10)
    )

    base = -2.5 + 3.2 * latent_intent + fit_boost + behavior_boost
    p = 1 / (1 + np.exp(-base))
    df["converted"] = rng.binomial(1, np.clip(p, 0, 1))

    start = np.datetime64("2025-01-01")
    df["created_at"] = start + rng.integers(0, 365, size=n).astype("timedelta64[D]")

    return df