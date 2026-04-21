"""
attribute_detector.py
Automatic detection of protected attributes and proxy variables using:
  - NLP keyword matching on column names
  - Value distribution analysis (binary skew, cardinality, specific patterns)
  - Correlation-based proxy detection
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# ── Protected attribute keyword taxonomy ─────────────────────────────────────
PROTECTED_KEYWORDS = {
    "gender":      (["gender", "sex", "male", "female", "man", "woman", "nonbinary", "genderid"], "Gender is a protected attribute under EEOC, ECOA, and most anti-discrimination laws."),
    "race":        (["race", "ethnicity", "ethnic", "racial", "black", "white", "hispanic", "asian", "origin", "nationality"], "Race/ethnicity is protected under the Civil Rights Act, ECOA, and GDPR."),
    "age":         (["age", "dob", "birthdate", "birth_year", "born", "yob", "year_of_birth"], "Age is a protected attribute under the Age Discrimination in Employment Act (ADEA) and ECOA."),
    "religion":    (["religion", "religious", "faith", "church", "mosque", "temple", "belief", "denomination"], "Religion is protected under Title VII and many international regulations."),
    "disability":  (["disability", "disabled", "handicap", "impairment", "condition", "medical"], "Disability status is protected under the ADA and ECOA."),
    "marital":     (["marital", "married", "single", "divorced", "spouse", "civil_status", "relationship_status"], "Marital status is protected under ECOA."),
    "pregnancy":   (["pregnant", "pregnancy", "maternity", "parental", "parental_leave"], "Pregnancy is protected under the Pregnancy Discrimination Act and EEOC."),
    "national":    (["country", "nation", "citizen", "citizenship", "immigrant", "visa", "residency", "birthplace"], "National origin is protected under Title VII and ECOA."),
    "political":   (["political", "party", "vote", "affiliation"], "Political affiliation is protected in many jurisdictions under GDPR and state laws."),
    "income":      (["income", "salary", "wage", "earnings", "compensation", "pay"], "Income proxies can encode socioeconomic discrimination correlated with protected attributes."),
    "zipcode":     (["zip", "zipcode", "zip_code", "postal", "postcode", "address", "neighborhood", "district"], "Geographic codes (ZIP, neighborhood) are well-documented proxies for race and national origin."),
}

# Values that are strong signals of protected attributes
PROTECTED_VALUE_PATTERNS = {
    "gender":  [{"m", "f"}, {"male", "female"}, {"0", "1"}, {"man", "woman"}, {"m", "f", "nb", "other"}],
    "race":    [{"white", "black", "asian", "hispanic", "other"}, {"caucasian", "african american", "latino"}],
    "marital": [{"married", "single", "divorced", "widowed"}, {"s", "m", "d", "w"}],
}


def _column_name_score(col: str) -> Dict[str, Any]:
    """Score a column name against the protected attribute keyword taxonomy."""
    col_lower = col.lower().replace("_", " ").replace("-", " ")
    for category, (keywords, explanation) in PROTECTED_KEYWORDS.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', col_lower) or col_lower == kw:
                confidence = 0.95 if col_lower in keywords else 0.80
                return {"category": category, "confidence": confidence, "reason": explanation, "matched_keyword": kw}
            # Partial match
            if kw in col_lower:
                return {"category": category, "confidence": 0.65, "reason": explanation, "matched_keyword": kw}
    return {}


def _value_distribution_score(series: pd.Series) -> Dict[str, Any]:
    """Inspect value distributions for protected-attribute patterns."""
    try:
        clean = series.dropna()
        if len(clean) == 0:
            return {}

        # Categorical / object column
        if series.dtype == object or series.nunique() < 20:
            unique_lower = {str(v).lower().strip() for v in clean.unique()}
            for category, pattern_list in PROTECTED_VALUE_PATTERNS.items():
                for pattern in pattern_list:
                    if unique_lower == pattern or unique_lower.issubset(pattern | {"other", "unknown", "n/a"}):
                        return {
                            "category": category,
                            "confidence": 0.75,
                            "reason": f"Values exactly match known {category} patterns: {unique_lower}",
                        }

        # Binary numeric with skewed distribution — suspect
        if series.dtype in [np.int64, np.float64] and series.nunique() == 2:
            counts = series.value_counts(normalize=True)
            minority_pct = counts.min()
            if 0.05 < minority_pct < 0.45:
                return {
                    "category": "binary_protected",
                    "confidence": 0.55,
                    "reason": f"Binary column with minority group at {minority_pct:.1%} — may encode a protected attribute.",
                }

    except Exception:
        pass
    return {}


def _compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise Cramér's V for categorical + numeric correlations."""
    # Encode all columns as numeric for correlation
    df_enc = df.copy()
    for col in df_enc.columns:
        if df_enc[col].dtype == object:
            try:
                df_enc[col] = pd.Categorical(df_enc[col]).codes
            except Exception:
                df_enc[col] = 0
    df_enc = df_enc.fillna(0)
    try:
        corr = df_enc.corr().abs()
    except Exception:
        corr = pd.DataFrame()
    return corr


def detect_protected_attributes(
    df: pd.DataFrame,
    target_col: str,
    correlation_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Main detection pipeline.
    Returns dict with keys: protected, proxies, safe
    """
    protected: List[Dict] = []
    all_flags: Dict[str, Dict] = {}   # col -> flag info

    non_target_cols = [c for c in df.columns if c != target_col]

    # Step 1: Name + value analysis
    for col in non_target_cols:
        name_result = _column_name_score(col)
        val_result  = _value_distribution_score(df[col])

        if name_result or val_result:
            best = name_result if name_result.get("confidence", 0) >= val_result.get("confidence", 0) else val_result
            reason = best.get("reason", "")
            confidence = best.get("confidence", 0.5)
            category   = best.get("category", "unknown")

            sample_vals = ", ".join(str(v) for v in df[col].dropna().unique()[:8])
            flag = {
                "column":        col,
                "category":      category,
                "confidence":    confidence,
                "reason":        reason,
                "sample_values": sample_vals,
                "detection":     "name+value" if (name_result and val_result) else ("name" if name_result else "value"),
            }
            all_flags[col] = flag
            protected.append(flag)

    # Step 2: Correlation-based proxy detection
    corr = _compute_correlations(df)
    proxies: List[Dict] = []
    protected_cols = {f["column"] for f in protected}

    for protected_col in protected_cols:
        if protected_col not in corr.index:
            continue
        corr_row = corr[protected_col].drop(index=[protected_col, target_col], errors="ignore")
        high_corr = corr_row[corr_row >= correlation_threshold]
        for proxy_col, corr_val in high_corr.items():
            if proxy_col in protected_cols:
                continue  # already flagged directly
            proxies.append({
                "column":          proxy_col,
                "correlated_with": protected_col,
                "correlation":     round(float(corr_val), 3),
                "reason": (
                    f"`{proxy_col}` has a {corr_val:.2f} correlation with `{protected_col}`. "
                    f"Including this variable in a model could result in indirect (proxy) discrimination even if "
                    f"`{protected_col}` is excluded from training."
                ),
            })

    # Step 3: Safe columns
    flagged_all = protected_cols | {p["column"] for p in proxies} | {target_col}
    safe = [c for c in df.columns if c not in flagged_all]

    return {
        "protected":  protected,
        "proxies":    proxies,
        "safe":       safe,
        "corr_matrix": corr,
    }
