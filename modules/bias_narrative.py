"""
bias_narrative.py
Generates plain-English explanations of:
  - Protected attribute findings
  - Proxy discrimination risks
  - Fairness metric interpretations
  - Actionable recommendations per regulatory context
"""

from typing import Dict, List, Any


# ── Regulatory context per category ───────────────────────────────────────────
LEGAL_CONTEXT = {
    "gender":      "This is illegal under Title VII of the Civil Rights Act, EEOC guidelines, and ECOA for credit decisions.",
    "race":        "Racial discrimination is prohibited by Title VII, the Fair Housing Act, ECOA, and EU GDPR Article 22.",
    "age":         "Age discrimination against persons 40+ is prohibited under ADEA. ECOA also prohibits age-based credit decisions.",
    "religion":    "Religious discrimination is illegal under Title VII and protected in the EU under GDPR Article 9.",
    "disability":  "Disability discrimination violates the ADA (Americans with Disabilities Act) and Section 508.",
    "marital":     "Marital status discrimination is prohibited under ECOA for credit/lending decisions.",
    "national":    "National origin discrimination violates Title VII, IRCA, and EU anti-discrimination directives.",
    "pregnancy":   "Pregnancy-based discrimination violates the Pregnancy Discrimination Act and Title VII.",
    "income":      "Socioeconomic proxies can indirectly encode racial or national-origin bias, violating disparate impact doctrine.",
    "zipcode":     "ZIP code and neighborhood data are well-documented proxies for race and national origin under fair lending laws (ECOA, CRA).",
    "political":   "Political affiliation is protected under GDPR and various state-level regulations.",
    "binary_protected": "This binary column may encode a protected characteristic. Investigate value semantics before including it in model training.",
}

RECOMMENDATIONS = {
    "gender":      ["Remove the column from training features.", "If necessary, apply fairness constraints (e.g., adversarial debiasing).", "Audit model outputs for disparate impact."],
    "race":        ["Remove column from features.", "Apply disparate impact testing (80% rule).", "Consider fairness-aware algorithms."],
    "age":         ["Remove or bin age into broad categories that don't encode protected status.", "Test model for age-group disparate impact."],
    "zipcode":     ["Consider removing ZIP code or replacing with non-proxied geographic features.", "Apply fairness constraints during training."],
    "income":      ["If income is a legitimate feature, test carefully for proxy effects.", "Apply equalized odds post-processing if disparities exist."],
    "binary_protected": ["Inspect value semantics carefully.", "If it encodes a protected attribute, remove it from training."],
    "default":     ["Review this attribute carefully with a fairness specialist.", "Run disparate impact analysis before deploying any model trained on this data."],
}


def _format_recommendation(recs: List[str]) -> str:
    return "\n".join(f"  → {r}" for r in recs)


def generate_bias_narrative(
    df,
    detection_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    smote_results: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Return a list of narrative dicts with keys: title, body, severity
    """
    narratives = []

    # ── 1. Protected attribute narratives ─────────────────────────────────
    for attr in detection_results.get("protected", []):
        col      = attr["column"]
        category = attr.get("category", "unknown")
        conf     = attr.get("confidence", 0.5)
        legal    = LEGAL_CONTEXT.get(category, LEGAL_CONTEXT["default"] if "default" in LEGAL_CONTEXT else "Consult a fairness specialist.")
        recs     = RECOMMENDATIONS.get(category, RECOMMENDATIONS["default"])

        severity = "HIGH" if conf > 0.75 else "MEDIUM"
        body = (
            f"Column: `{col}`\n"
            f"Category: {category.replace('_',' ').title()}  |  Detection confidence: {conf:.0%}\n"
            f"Detection method: {attr.get('detection','analysis')}\n\n"
            f"Why this matters:\n"
            f"  {legal}\n\n"
            f"What was found:\n"
            f"  Sample values observed: {attr.get('sample_values', 'N/A')}\n"
            f"  {attr.get('reason', '')}\n\n"
            f"Recommended actions:\n{_format_recommendation(recs)}"
        )
        narratives.append({"title": f"Protected Attribute Detected: `{col}`", "body": body, "severity": severity})

    # ── 2. Proxy variable narratives ─────────────────────────────────────
    for proxy in detection_results.get("proxies", []):
        col      = proxy["column"]
        prot_col = proxy["correlated_with"]
        corr_val = proxy["correlation"]

        severity = "HIGH" if corr_val > 0.75 else "MEDIUM"
        legal_context = ""
        for category, (keywords, explanation) in {
            "race":    (["race", "ethnicity", "zipcode"], "This may constitute illegal proxy discrimination under fair lending and civil rights laws."),
            "gender":  (["gender", "sex"], "Using proxy variables for gender may violate EEOC guidelines."),
        }.items():
            if any(kw in prot_col.lower() for kw in keywords[0:1]):
                legal_context = keywords[-1] if isinstance(keywords[-1], str) else ""
                break

        body = (
            f"Column: `{col}`  →  correlated with protected attribute `{prot_col}` (r = {corr_val:.2f})\n\n"
            f"What this means:\n"
            f"  `{col}` has a {corr_val:.2f} correlation with the `{prot_col}` column.\n"
            f"  This means your model could effectively discriminate by {prot_col} through {col} —\n"
            f"  even if `{prot_col}` is excluded from training. This is called PROXY DISCRIMINATION.\n\n"
            f"Legal implications:\n"
            f"  Proxy discrimination is actionable under disparate impact doctrine (Griggs v. Duke Power Co.).\n"
            f"  It is illegal under ECOA in credit/lending, Title VII in employment, and\n"
            f"  challenged under GDPR Article 22 in automated decision-making.\n\n"
            f"Recommended actions:\n"
            f"  → Remove `{col}` from training features, OR\n"
            f"  → Apply fairness constraints (adversarial debiasing, reweighting) during training.\n"
            f"  → Run a disparate impact analysis (4/5ths rule) to quantify the risk.\n"
            f"  → Consult legal counsel if this model affects credit, employment, or housing."
        )
        narratives.append({"title": f"Proxy Discrimination Risk: `{col}` ↔ `{prot_col}`", "body": body, "severity": severity})

    # ── 3. Fairness metric narrative ──────────────────────────────────────
    before = fairness_results.get("before", {})
    after  = fairness_results.get("after", {})
    dp_before = before.get("demographic_parity_diff")
    dp_after  = after.get("demographic_parity_diff")

    if dp_before is not None:
        passed = dp_before < 0.10
        severity = "LOW" if passed else ("MEDIUM" if dp_before < 0.20 else "HIGH")
        improvement = ""
        if dp_after is not None:
            delta = dp_before - dp_after
            improvement = (
                f"\nAfter SMOTE rebalancing: {dp_after:.3f} (improvement of {abs(delta):.3f})\n"
                + ("  ✅ The bias has been reduced to an acceptable level." if dp_after < 0.10 else "  ⚠️ Bias persists after SMOTE — consider adversarial reweighting.")
            )
        body = (
            f"Demographic Parity Difference (before SMOTE): {dp_before:.3f}\n"
            f"Threshold required by ECOA/EEOC: < 0.10 (10 percentage points)\n"
            f"Status: {'✅ PASS' if passed else '❌ FAIL — Disparate impact detected'}\n"
            f"{improvement}\n\n"
            f"What this means:\n"
            f"  A demographic parity difference of {dp_before:.3f} means the model's positive outcome\n"
            f"  rate differs by {dp_before*100:.1f} percentage points between demographic groups.\n\n"
            f"Recommended actions:\n"
            + ("  → Current bias level meets ECOA threshold. Monitor for drift over time." if passed else
               "  → Apply SMOTE or class weighting to balance training data.\n"
               "  → Consider post-processing fairness constraints (e.g., threshold optimization).\n"
               "  → Audit feature set for proxy variables before deployment.")
        )
        narratives.append({"title": "Demographic Parity Fairness Assessment", "body": body, "severity": severity})

    # ── 4. SMOTE quality narrative ────────────────────────────────────────
    quality_score = smote_results.get("quality_score", None)
    if quality_score is not None:
        severity = "LOW" if quality_score > 0.7 else ("MEDIUM" if quality_score > 0.4 else "HIGH")
        body = (
            f"SMOTE synthetic data quality score: {quality_score:.2f}\n"
            f"Interpretation: {'Good — synthetic samples closely mirror real minority distributions.' if quality_score > 0.7 else 'Moderate — some distributional drift in synthetic samples.' if quality_score > 0.4 else 'Poor — synthetic samples diverge significantly from real data. Consider alternative strategies.'}\n\n"
            f"How quality is measured:\n"
            f"  We compare the distribution of key attributes within synthetic minority samples\n"
            f"  versus real minority samples using KL-divergence and statistical distance metrics.\n\n"
        )
        alerts = smote_results.get("quality_alerts", [])
        if alerts:
            body += "Specific issues detected:\n" + "\n".join(f"  ⚠️ {a}" for a in alerts) + "\n\n"
        alternatives = smote_results.get("alternatives", [])
        if alternatives:
            body += "Recommended alternatives:\n" + "\n".join(f"  → {a}" for a in alternatives)
        narratives.append({"title": "SMOTE Synthetic Data Quality Report", "body": body, "severity": severity})

    # ── 5. Summary narrative if no issues found ───────────────────────────
    if not narratives:
        narratives.append({
            "title": "No Major Bias Issues Detected",
            "body": (
                "Initial analysis did not identify high-confidence protected attributes or proxy variables.\n\n"
                "Note: This does not guarantee the dataset is bias-free.\n"
                "  → Review column semantics manually, especially for fields like ZipCode, School, or NeighborhoodName.\n"
                "  → Run model-level disparate impact testing before deployment.\n"
                "  → Consider a third-party fairness audit for high-stakes decisions."
            ),
            "severity": "LOW",
        })

    return narratives
