"""
report_generator.py
Generates a PDF compliance report mapping fairness metrics to regulatory requirements.
Format per regulation:
  "Metric: Value (Threshold required by REG: X) — PASS/FAIL"
Includes: dataset stats, methodology, findings, remediation steps.
"""

import io
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


# ── Regulatory threshold definitions ─────────────────────────────────────────
REGULATIONS = {
    "ECOA": {
        "full_name": "Equal Credit Opportunity Act (ECOA)",
        "jurisdiction": "United States (Federal)",
        "scope": "Credit and lending decisions",
        "metrics": {
            "demographic_parity_diff": {
                "threshold": 0.10,
                "direction": "below",
                "label": "Demographic Parity Difference",
                "description": "Outcome rate difference between demographic groups must be < 10pp",
            },
            "equalized_odds_diff": {
                "threshold": 0.10,
                "direction": "below",
                "label": "Equalized Odds Difference",
                "description": "True positive rate difference across groups must be < 10pp",
            },
        },
    },
    "EEOC": {
        "full_name": "Equal Employment Opportunity Commission Guidelines",
        "jurisdiction": "United States (Federal)",
        "scope": "Employment screening and hiring decisions",
        "metrics": {
            "demographic_parity_diff": {
                "threshold": 0.20,
                "direction": "below",
                "label": "Adverse Impact Ratio (4/5ths Rule)",
                "description": "Selection rate for protected group must be ≥ 80% of highest-selected group (≈ Demographic Parity Diff < 0.20)",
            },
        },
    },
    "GDPR Art.22": {
        "full_name": "EU General Data Protection Regulation Article 22",
        "jurisdiction": "European Union",
        "scope": "Automated decision-making with significant effects on individuals",
        "metrics": {
            "equalized_odds_diff": {
                "threshold": 0.15,
                "direction": "below",
                "label": "Equalized Odds Difference",
                "description": "Model must produce equitable error rates across protected groups",
            },
            "demographic_parity_diff": {
                "threshold": 0.15,
                "direction": "below",
                "label": "Demographic Parity Difference",
                "description": "Outcome rates across protected groups must be within 15pp",
            },
        },
    },
    "HHS Healthcare": {
        "full_name": "HHS Office for Civil Rights — AI Fairness in Healthcare",
        "jurisdiction": "United States (Federal)",
        "scope": "Healthcare algorithms and clinical decision support",
        "metrics": {
            "demographic_parity_diff": {
                "threshold": 0.10,
                "direction": "below",
                "label": "Demographic Parity Difference",
                "description": "Clinical outcome allocation must be demographically equitable",
            },
        },
    },
}


def _pass_fail(value: float, threshold: float, direction: str) -> str:
    if direction == "below":
        return "✅ PASS" if value < threshold else "❌ FAIL"
    return "✅ PASS" if value > threshold else "❌ FAIL"


def generate_pdf_report(
    df: pd.DataFrame,
    target_col: str,
    detection_results: Dict[str, Any],
    fairness_results: Dict[str, Any],
    smote_results: Dict[str, Any],
    narratives: List[Dict[str, Any]],
    active_regs: List[str],
) -> bytes:
    """
    Generate and return a PDF compliance report as bytes using fpdf2.
    """
    return _generate_with_fpdf(df, target_col, detection_results, fairness_results, smote_results, narratives, active_regs)


def _generate_with_fpdf(df, target_col, detection_results, fairness_results, smote_results, narratives, active_regs) -> bytes:
    from fpdf import FPDF

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(15, 32, 39)
            self.set_fill_color(0, 229, 200)
            self.cell(0, 12, "  FairLens — AI Bias & Fairness Compliance Report", ln=True, fill=True)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 6, f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}   |   Target column: {target_col}", ln=True)
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"FairLens Compliance Report — Page {self.page_no()} — CONFIDENTIAL", align="C")

        def section_title(self, title: str):
            self.ln(4)
            self.set_font("Helvetica", "B", 12)
            self.set_text_color(0, 100, 140)
            self.cell(0, 8, title, ln=True)
            self.set_draw_color(0, 229, 200)
            self.set_line_width(0.5)
            self.line(self.get_x(), self.get_y(), self.get_x() + 180, self.get_y())
            self.ln(2)
            self.set_text_color(30, 30, 30)

        def body_text(self, txt: str, bold: bool = False):
            self.set_font("Helvetica", "B" if bold else "", 9)
            self.multi_cell(0, 5, txt)
            self.ln(1)

        def kv_row(self, key: str, value: str, highlight: bool = False):
            self.set_font("Helvetica", "B", 9)
            self.cell(70, 6, key + ":", ln=False)
            self.set_font("Helvetica", "", 9)
            if highlight:
                color = (80, 200, 120) if "PASS" in value else (220, 80, 80)
                self.set_text_color(*color)
            self.cell(0, 6, value, ln=True)
            self.set_text_color(30, 30, 30)

    pdf = PDF()
    pdf.set_margins(15, 20, 15)
    pdf.add_page()

    # ── Executive Summary ──────────────────────────────────────────────────
    pdf.section_title("1. Executive Summary")
    n_protected = len(detection_results.get("protected", []))
    n_proxies   = len(detection_results.get("proxies", []))
    before      = fairness_results.get("before", {})
    dp          = before.get("demographic_parity_diff")

    pdf.body_text(
        f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        f"Protected attributes detected: {n_protected}\n"
        f"Proxy variables detected: {n_proxies}\n"
        f"Demographic Parity Difference: {f'{dp:.3f}' if dp is not None else 'N/A'}\n"
        f"Model Accuracy (baseline): {before.get('model_accuracy', 'N/A')}\n"
        f"SMOTE quality score: {smote_results.get('quality_score', 'N/A')}\n"
        f"Synthetic samples generated: {smote_results.get('synthetic_count', 0):,}"
    )

    # ── Dataset Statistics ─────────────────────────────────────────────────
    pdf.section_title("2. Dataset Statistics & Methodology")
    numeric_df = df.select_dtypes(include="number")
    pdf.body_text(
        f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}  |  Numeric columns: {numeric_df.shape[1]}\n"
        f"Missing values: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/df.size*100:.1f}%)\n\n"
        "Methodology:\n"
        "  • Protected attribute detection: NLP keyword matching + value distribution analysis + correlation thresholding\n"
        "  • Fairness metrics: fairlearn MetricFrame — Demographic Parity Difference & Equalized Odds Difference\n"
        "  • Baseline model: Logistic Regression (scikit-learn, max_iter=1000)\n"
        "  • Rebalancing: SMOTE (imbalanced-learn) with quality validation via KL divergence\n"
        "  • PDF report: fpdf2"
    )

    # ── Protected Attributes ───────────────────────────────────────────────
    pdf.section_title("3. Protected Attributes & Proxy Variables")
    for attr in detection_results.get("protected", []):
        pdf.kv_row("Column", attr["column"])
        pdf.kv_row("Category", attr.get("category", "").replace("_", " ").title())
        pdf.kv_row("Confidence", f"{attr.get('confidence', 0):.0%}")
        pdf.body_text(f"  Reason: {attr.get('reason', '')}")
        pdf.ln(1)

    if not detection_results.get("protected"):
        pdf.body_text("No protected attributes detected at high confidence.")

    pdf.body_text("Proxy Variables:", bold=True)
    for p in detection_results.get("proxies", []):
        pdf.body_text(f"  • {p['column']} — correlation {p['correlation']:.2f} with {p['correlated_with']}: {p['reason']}")
    if not detection_results.get("proxies"):
        pdf.body_text("  None detected above correlation threshold.")

    # ── Regulatory Compliance ──────────────────────────────────────────────
    pdf.section_title("4. Regulatory Compliance Assessment")
    for reg_name in active_regs:
        reg = REGULATIONS.get(reg_name)
        if not reg:
            continue
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"{reg_name} — {reg['full_name']}", ln=True)
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 5, f"Jurisdiction: {reg['jurisdiction']}  |  Scope: {reg['scope']}", ln=True)
        pdf.ln(1)

        for metric_key, metric_info in reg["metrics"].items():
            val = before.get(metric_key)
            thr = metric_info["threshold"]
            lbl = metric_info["label"]
            if val is not None:
                status = _pass_fail(val, thr, metric_info["direction"])
                pdf.kv_row(lbl, f"{val:.3f}  (Required: {'< ' if metric_info['direction']=='below' else '> '}{thr})  —  {status}", highlight=True)
            else:
                pdf.kv_row(lbl, "N/A — insufficient data for measurement")
            pdf.body_text(f"    {metric_info['description']}")
        pdf.ln(2)

    # ── Bias Narratives ────────────────────────────────────────────────────
    pdf.section_title("5. Bias Findings & Narratives")
    for narrative in narratives:
        severity_label = {"HIGH": "[HIGH RISK]", "MEDIUM": "[MEDIUM RISK]", "LOW": "[LOW RISK]"}.get(narrative.get("severity"), "")
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, f"{severity_label} {narrative.get('title', '')}", ln=True)
        pdf.set_font("Courier", "", 8)
        body_clean = narrative.get("body", "").replace("→", "->").replace("✅", "[PASS]").replace("❌", "[FAIL]").replace("⚠️", "[WARN]")
        pdf.multi_cell(0, 4.5, body_clean)
        pdf.ln(3)

    # ── SMOTE Quality ──────────────────────────────────────────────────────
    pdf.section_title("6. SMOTE Rebalancing & Quality Report")
    pdf.kv_row("Original samples", f"{smote_results.get('original_count', 0):,}")
    pdf.kv_row("Synthetic samples generated", f"{smote_results.get('synthetic_count', 0):,}")
    pdf.kv_row("Quality score", f"{smote_results.get('quality_score', 0):.2f} / 1.00")
    for alert in smote_results.get("quality_alerts", []):
        pdf.body_text(f"  WARNING: {alert}")
    for alt in smote_results.get("alternatives", []):
        pdf.body_text(f"  Alternative: {alt}")

    # ── Remediation Steps ──────────────────────────────────────────────────
    pdf.section_title("7. Remediation Steps & Next Actions")
    pdf.body_text(
        "1. Remove or anonymize confirmed protected attributes from training features.\n"
        "2. Investigate and remove high-correlation proxy variables.\n"
        "3. Apply fairness constraints (adversarial debiasing, reweighting) if disparities persist.\n"
        "4. Re-run this audit after each model update.\n"
        "5. Consult legal counsel before deploying in regulated domains (credit, employment, healthcare).\n"
        "6. Document all findings and remediation steps for regulatory record-keeping."
    )

    pdf.section_title("8. Disclaimer")
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5,
        "This report is generated by an automated fairness analysis tool (FairLens). "
        "It is intended to assist developers and compliance teams in identifying potential bias risks. "
        "It does not constitute legal advice. Regulatory requirements vary by jurisdiction and context. "
        "Always consult qualified legal counsel and domain experts before making compliance decisions."
    )

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


def _generate_with_reportlab(df, target_col, detection_results, fairness_results, smote_results, narratives, active_regs) -> bytes:
    """Reportlab-based PDF generation (higher quality typography)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()

    TEAL   = colors.HexColor("#00e5c8")
    DARK   = colors.HexColor("#0f2027")
    RED    = colors.HexColor("#ff6b6b")
    GREEN  = colors.HexColor("#51cf66")
    ORANGE = colors.HexColor("#ffa94d")

    title_style = ParagraphStyle("title", fontSize=20, textColor=TEAL, spaceAfter=4, fontName="Helvetica-Bold")
    h2_style    = ParagraphStyle("h2", fontSize=13, textColor=DARK, spaceAfter=3, spaceBefore=12, fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", fontSize=9, spaceAfter=3, leading=14, fontName="Helvetica")
    mono_style  = ParagraphStyle("mono", fontSize=8, fontName="Courier", spaceAfter=2, leading=12)
    caption     = ParagraphStyle("caption", fontSize=8, textColor=colors.grey, fontName="Helvetica-Oblique")

    story = []

    # Header
    story.append(Paragraph("⚖ FairLens — AI Fairness Compliance Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  Target: <b>{target_col}</b>", caption))
    story.append(HRFlowable(width="100%", thickness=2, color=TEAL, spaceAfter=10))

    # Executive Summary table
    story.append(Paragraph("1. Executive Summary", h2_style))
    before = fairness_results.get("before", {})
    dp     = before.get("demographic_parity_diff")
    summary_data = [
        ["Metric", "Value"],
        ["Dataset size", f"{df.shape[0]:,} rows × {df.shape[1]} columns"],
        ["Protected attributes detected", str(len(detection_results.get("protected", [])))],
        ["Proxy variables detected", str(len(detection_results.get("proxies", [])))],
        ["Demographic Parity Difference", f"{dp:.3f}" if dp is not None else "N/A"],
        ["Model accuracy (baseline)", str(before.get("model_accuracy", "N/A"))],
        ["SMOTE quality score", str(smote_results.get("quality_score", "N/A"))],
        ["Synthetic samples generated", f"{smote_results.get('synthetic_count', 0):,}"],
    ]
    t = Table(summary_data, colWidths=[9*cm, 8*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("TEXTCOLOR",  (0, 0), (-1, 0), TEAL),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f8ff"), colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("PADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3*cm))

    # Regulatory Compliance
    story.append(Paragraph("2. Regulatory Compliance", h2_style))
    for reg_name in active_regs:
        reg = REGULATIONS.get(reg_name)
        if not reg:
            continue
        story.append(Paragraph(f"<b>{reg_name}</b> — {reg['full_name']}", body_style))
        story.append(Paragraph(f"Scope: {reg['scope']}", caption))
        reg_rows = [["Metric", "Value", "Threshold", "Status"]]
        for metric_key, info in reg["metrics"].items():
            val = before.get(metric_key)
            thr = info["threshold"]
            if val is not None:
                status = _pass_fail(val, thr, info["direction"])
                reg_rows.append([info["label"], f"{val:.3f}", f"{'< ' if info['direction']=='below' else '> '}{thr}", status])
            else:
                reg_rows.append([info["label"], "N/A", f"{thr}", "⚠ N/A"])
        rt = Table(reg_rows, colWidths=[7*cm, 3*cm, 3*cm, 4*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), DARK),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 8),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("PADDING",    (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff9f0"), colors.white]),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.2*cm))

    # Findings
    story.append(Paragraph("3. Findings & Narratives", h2_style))
    for narrative in narratives:
        color_map = {"HIGH": "#ff6b6b", "MEDIUM": "#ffa94d", "LOW": "#51cf66"}
        c_hex = color_map.get(narrative.get("severity", "LOW"), "#aaaaaa")
        story.append(Paragraph(f'<font color="{c_hex}">[{narrative.get("severity","INFO")}]</font> <b>{narrative.get("title","")}</b>', body_style))
        body_clean = narrative.get("body", "").replace("→", "->").replace("✅", "[PASS]").replace("❌", "[FAIL]").replace("⚠️", "[WARN]").replace("⚠", "[WARN]")
        story.append(Paragraph(body_clean.replace("\n", "<br/>"), mono_style))
        story.append(Spacer(1, 0.2*cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=8))
    story.append(Paragraph(
        "Disclaimer: This report is generated by an automated tool (FairLens) for informational purposes only. "
        "It does not constitute legal advice. Consult qualified legal counsel before making compliance decisions.",
        caption
    ))

    doc.build(story)
    return buf.getvalue()
