"""
fairness_metrics.py  (fixed)
Fix: consistent LabelEncoder-based encoding for all columns before model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def _encode_features(X: pd.DataFrame) -> np.ndarray:
    """Encode all columns to numeric, fill NaN."""
    from sklearn.preprocessing import LabelEncoder
    X_enc = X.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == object or str(X_enc[col].dtype) == "category":
            X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
        else:
            X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce")
    return X_enc.fillna(0).astype(float).values


def _encode_series(s: pd.Series) -> np.ndarray:
    """Encode a target series to integer labels."""
    from sklearn.preprocessing import LabelEncoder
    if s.dtype == object or str(s.dtype) == "category":
        return LabelEncoder().fit_transform(s.astype(str))
    return s.astype(int).values


def _train_and_evaluate(X: np.ndarray, y: np.ndarray) -> Dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    if len(np.unique(y)) < 2:
        return {"error": "Target column has only one class — cannot train a model."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "model_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "model": model, "scaler": scaler,
        "X_test": X_test, "y_test": y_test, "y_pred": y_pred,
    }


def _compute_fairness_metrics(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: Optional[str],
) -> Dict[str, Any]:
    result = {}
    try:
        df_clean = df.dropna().copy().reset_index(drop=True)
        X = _encode_features(df_clean.drop(columns=[target_col]))
        y = _encode_series(df_clean[target_col])

        eval_r = _train_and_evaluate(X, y)
        if "error" in eval_r:
            return eval_r

        result["model_accuracy"] = eval_r["model_accuracy"]
        result["accuracy"]       = eval_r["model_accuracy"]

        if sensitive_col and sensitive_col in df_clean.columns:
            from fairlearn.metrics import (
                demographic_parity_difference,
                equalized_odds_difference,
                MetricFrame,
            )
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score

            scaler2 = StandardScaler()
            X_scaled = scaler2.fit_transform(X)
            model2 = LogisticRegression(max_iter=1000, random_state=42)
            model2.fit(X_scaled, y)
            y_pred_full = model2.predict(X_scaled)

            # Encode sensitive attribute
            sens = _encode_series(df_clean[sensitive_col]) if df_clean[sensitive_col].dtype == object \
                   else df_clean[sensitive_col].fillna(0).astype(int).values

            dp_diff = demographic_parity_difference(y, y_pred_full, sensitive_features=sens)
            eo_diff = equalized_odds_difference(y, y_pred_full, sensitive_features=sens)

            result["demographic_parity_diff"] = round(abs(float(dp_diff)), 4)
            result["equalized_odds_diff"]      = round(abs(float(eo_diff)), 4)

            # Subgroup breakdown using original (readable) values of sensitive col
            sens_readable = df_clean[sensitive_col].astype(str).values
            mf = MetricFrame(
                metrics=accuracy_score,
                y_true=y,
                y_pred=y_pred_full,
                sensitive_features=sens_readable,
            )
            sg = mf.by_group.reset_index()
            sg.columns = [sensitive_col, "accuracy"]
            sg["accuracy"] = sg["accuracy"].round(4)

            # Positive outcome rate per group
            outcome_df = df_clean[[sensitive_col, target_col]].copy()
            outcome_df[target_col] = y
            rate = outcome_df.groupby(sensitive_col)[target_col].mean().reset_index()
            rate.columns = [sensitive_col, "positive_outcome_rate"]
            rate["positive_outcome_rate"] = rate["positive_outcome_rate"].round(4)
            merged = sg.merge(rate, on=sensitive_col, how="left")
            result["subgroup_breakdown"] = merged.to_dict(orient="records")

    except Exception as e:
        result["error"] = str(e)
    return result


def run_fairness_analysis(
    df: pd.DataFrame,
    target_col: str,
    detection_results: Dict[str, Any],
) -> Dict[str, Any]:
    protected     = detection_results.get("protected", [])
    sensitive_col = protected[0]["column"] if protected else None

    before = _compute_fairness_metrics(df, target_col, sensitive_col)

    # After-SMOTE comparison
    after = {}
    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import LabelEncoder

        df_clean = df.dropna().copy().reset_index(drop=True)
        y        = _encode_series(df_clean[target_col])
        X        = _encode_features(df_clean.drop(columns=[target_col]))

        k = min(5, int((y == np.bincount(y).argmin()).sum()) - 1)
        if k < 1:
            raise ValueError("Not enough minority samples for SMOTE")

        smote        = SMOTE(sampling_strategy="auto", random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X, y)

        col_names = [c for c in df_clean.columns if c != target_col]
        df_after  = pd.DataFrame(X_res, columns=col_names)
        df_after[target_col] = y_res

        after = _compute_fairness_metrics(df_after, target_col, sensitive_col)

    except Exception as e:
        after = {"error": f"After-SMOTE metrics unavailable: {e}"}

    return {
        "before":             before,
        "after":              after,
        "subgroup_breakdown": before.get("subgroup_breakdown"),
    }
