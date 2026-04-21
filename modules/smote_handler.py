"""
smote_handler.py  (fixed)
SMOTE rebalancing with quality validation.
Fix: use LabelEncoder for ALL object/category columns before passing to SMOTE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Encode ALL non-numeric columns to integers. Robust to mixed dtypes."""
    from sklearn.preprocessing import LabelEncoder
    X_enc = X.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == object or str(X_enc[col].dtype) == "category":
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        else:
            X_enc[col] = pd.to_numeric(X_enc[col], errors="coerce")
    X_enc = X_enc.fillna(0).astype(float)
    return X_enc


def _distribution_comparison(
    real_minority: pd.DataFrame,
    synthetic: pd.DataFrame,
    n_bins: int = 10,
) -> Dict[str, Any]:
    records = []
    total_kl = 0.0
    n_compared = 0

    for col in real_minority.columns:
        try:
            combined = pd.concat([real_minority[col], synthetic[col]]).dropna()
            if combined.nunique() < 2:
                continue
            bins = np.linspace(combined.min(), combined.max(), n_bins + 1)
            real_hist, _ = np.histogram(real_minority[col].dropna(), bins=bins)
            syn_hist,  _ = np.histogram(synthetic[col].dropna(),  bins=bins)
            kl = _kl_divergence(real_hist, syn_hist)
            total_kl += kl
            n_compared += 1
            records.append({
                "column":         col,
                "real_mean":      round(float(real_minority[col].mean()), 4),
                "synthetic_mean": round(float(synthetic[col].mean()), 4),
                "real_std":       round(float(real_minority[col].std()), 4),
                "synthetic_std":  round(float(synthetic[col].std()), 4),
                "kl_divergence":  round(kl, 4),
                "quality":        "Good" if kl < 0.5 else "Moderate" if kl < 1.5 else "Poor",
            })
        except Exception:
            continue

    avg_kl = total_kl / max(n_compared, 1)
    quality_score = max(0.0, 1.0 - min(avg_kl / 3.0, 1.0))
    return {"records": records, "quality_score": round(quality_score, 3), "avg_kl": round(avg_kl, 4)}


def apply_smote_with_validation(
    df: pd.DataFrame,
    target_col: str,
    detection_results: Dict[str, Any],
    sampling_strategy: str = "auto",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    try:
        from imblearn.over_sampling import SMOTE
        from sklearn.preprocessing import LabelEncoder

        df_clean = df.dropna().copy().reset_index(drop=True)

        # Encode target
        y_raw = df_clean[target_col]
        if y_raw.dtype == object or str(y_raw.dtype) == "category":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        else:
            y = y_raw.astype(int).reset_index(drop=True)

        # Encode ALL features (key fix - handles 'Female', 'Male', etc.)
        X_raw = df_clean.drop(columns=[target_col])
        X = _encode_features(X_raw)

        class_dist_before = y.value_counts().to_dict()
        result["class_dist_before"] = {str(k): int(v) for k, v in class_dist_before.items()}
        result["original_count"]    = len(df_clean)

        minority_class = y.value_counts().idxmin()
        minority_mask  = (y == minority_class)
        real_minority  = X[minority_mask].copy()

        k = min(5, int(minority_mask.sum()) - 1)
        if k < 1:
            raise ValueError(f"Not enough minority samples for SMOTE (need at least 2, got {minority_mask.sum()})")

        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X.values, y.values)

        n_original     = len(X)
        synthetic_mask = np.arange(len(X_res)) >= n_original
        synthetic_X    = pd.DataFrame(X_res[synthetic_mask], columns=X.columns)

        class_dist_after = pd.Series(y_res).value_counts().to_dict()
        result["class_dist_after"]  = {str(k): int(v) for k, v in class_dist_after.items()}
        result["synthetic_count"]   = int(synthetic_mask.sum())

        if len(real_minority) > 0 and len(synthetic_X) > 0:
            dist_result = _distribution_comparison(real_minority, synthetic_X)
            result["quality_score"]           = dist_result["quality_score"]
            result["distribution_comparison"] = dist_result["records"]
        else:
            result["quality_score"]           = 1.0
            result["distribution_comparison"] = []

        alerts:       List[str] = []
        alternatives: List[str] = []
        qs = result["quality_score"]

        if qs < 0.4:
            alerts.append("SMOTE quality is LOW. Synthetic samples diverge significantly from real minority data.")
            alternatives.append("Class Weighting: Use class_weight='balanced' in your model.")
            alternatives.append("Adversarial Reweighting: Assign sample weights to minimize fairness disparity.")
        if qs < 0.7:
            alerts.append("Some synthetic features show high KL divergence. Check the distribution table.")
            alternatives.append("ADASYN: Focuses synthetic generation on hard-to-learn minority samples.")

        poor_cols = [r["column"] for r in result.get("distribution_comparison", []) if r.get("kl_divergence", 0) > 1.5]
        if poor_cols:
            alerts.append(f"Poor synthetic quality in columns: {', '.join(poor_cols)}.")
        if minority_mask.sum() < 10:
            alerts.append(f"Very few real minority samples ({minority_mask.sum()}). Collect more real data if possible.")

        result["quality_alerts"] = alerts
        result["alternatives"]   = alternatives

        df_rebalanced = pd.DataFrame(X_res, columns=X.columns)
        df_rebalanced[target_col] = y_res
        result["df_rebalanced"] = df_rebalanced

    except Exception as e:
        result["error"]               = str(e)
        result.setdefault("quality_score", 0.0)
        result.setdefault("synthetic_count", 0)
        result.setdefault("original_count", len(df))
        result.setdefault("class_dist_before", {})
        result.setdefault("class_dist_after", {})

    return result
