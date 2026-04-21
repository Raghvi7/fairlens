"""
FairLens — AI Bias Detection & Fairness Audit Dashboard
Main Streamlit entry point
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import html
import traceback

from modules.attribute_detector import detect_protected_attributes
from modules.fairness_metrics import run_fairness_analysis
from modules.bias_narrative import generate_bias_narrative
from modules.smote_handler import apply_smote_with_validation
from modules.report_generator import generate_pdf_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens — AI Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0f2027 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        border-left: 5px solid #00e5c8;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: #00e5c8; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-header p { color: #9ecfdf; margin: 0.4rem 0 0 0; font-size: 1rem; }

    .metric-card {
        background: #1a2535;
        border: 1px solid #2a3f55;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #00e5c8; }
    .metric-card .label { color: #7a99b3; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #fff; font-size: 2rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
    .metric-card .sub { color: #9ecfdf; font-size: 0.75rem; }

    .flag-card {
        background: #1e1a2e;
        border-left: 4px solid #ff6b6b;
        border-radius: 6px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
    }
    .flag-card.proxy { border-left-color: #ffa94d; }
    .flag-card.clear { border-left-color: #51cf66; }

    .narrative-box {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1.5rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #c9e3f5;
        line-height: 1.7;
        white-space: pre-wrap;
    }

    .section-header {
        color: #00e5c8;
        font-size: 1.2rem;
        font-weight: 700;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 0.4rem;
        margin: 2rem 0 1rem 0;
    }

    .badge-high   { background:#ff6b6b; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-medium { background:#ffa94d; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }
    .badge-low    { background:#51cf66; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:700; }

    .stButton > button {
        background: linear-gradient(90deg, #00e5c8, #0099ff);
        color: #000;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ FairLens</h1>
    <p>AI Bias Detection & Fairness Audit Platform — Automated demographic parity, proxy discrimination analysis, SMOTE rebalancing & compliance reporting</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scales.png", width=60)
    st.markdown("## ⚙️ Configuration")

    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    st.markdown("---")
    st.markdown("### 🎯 Analysis Settings")
    target_col = st.text_input("Target Column (label to predict)", placeholder="e.g. hired, approved, loan_granted")
    correlation_threshold = st.slider("Proxy Correlation Threshold", 0.3, 0.95, 0.6, 0.05,
        help="Columns correlated above this value with a protected attribute are flagged as proxy variables.")
    smote_strategy = st.selectbox("SMOTE Sampling Strategy", ["auto", "minority", "not minority", "all"])
    generate_report = st.checkbox("📄 Generate PDF Compliance Report", value=True)

    st.markdown("---")
    st.markdown("### 🏛️ Regulations to Check")
    reg_ecoa  = st.checkbox("ECOA (Equal Credit Opportunity Act)", value=True)
    reg_eeoc  = st.checkbox("EEOC (Employment Fairness)", value=True)
    reg_gdpr  = st.checkbox("GDPR Article 22", value=True)
    reg_hipaa = st.checkbox("HHS AI Fairness (Healthcare)", value=False)

    active_regs = []
    if reg_ecoa:  active_regs.append("ECOA")
    if reg_eeoc:  active_regs.append("EEOC")
    if reg_gdpr:  active_regs.append("GDPR Art.22")
    if reg_hipaa: active_regs.append("HHS Healthcare")

    run_btn = st.button("🚀 Run Full Audit")

# ── Main body ─────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.info("👆 Upload a CSV dataset in the sidebar to begin the fairness audit.")
    # Demo section
    st.markdown('<div class="section-header">🗂️ What This Tool Does</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        **🔍 Protected Attribute Detection**
        - NLP-based column name analysis
        - Value distribution inspection
        - Proxy variable identification
        - Confidence scores & explanations
        """)
    with cols[1]:
        st.markdown("""
        **📊 Fairness Metrics**
        - Demographic Parity Difference
        - Equalized Odds comparison
        - Before vs. After SMOTE analysis
        - Model-integrated testing
        """)
    with cols[2]:
        st.markdown("""
        **📄 Compliance Reports**
        - ECOA / EEOC / GDPR mapping
        - Plain-English bias narratives
        - SMOTE quality validation
        - Exportable PDF report
        """)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.markdown(f'<div class="section-header">📂 Dataset Overview — {df.shape[0]:,} rows × {df.shape[1]} columns</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="label">Rows</div><div class="value">{df.shape[0]:,}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="label">Columns</div><div class="value">{df.shape[1]}</div></div>', unsafe_allow_html=True)
with col3:
    missing_pct = df.isnull().sum().sum() / df.size * 100
    st.markdown(f'<div class="metric-card"><div class="label">Missing %</div><div class="value">{missing_pct:.1f}%</div></div>', unsafe_allow_html=True)
with col4:
    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    st.markdown(f'<div class="metric-card"><div class="label">Numeric Cols</div><div class="value">{num_cols}</div></div>', unsafe_allow_html=True)

with st.expander("📋 Preview Dataset", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

if not run_btn:
    st.info("Configure settings in the sidebar and click **🚀 Run Full Audit** to begin.")
    st.stop()

if not target_col:
    st.warning("⚠️ Please enter the **Target Column** name in the sidebar before running.")
    st.stop()

if target_col not in df.columns:
    st.error(f"Column `{target_col}` not found in dataset. Available columns: {list(df.columns)}")
    st.stop()

# ── Run audit ─────────────────────────────────────────────────────────────────
with st.spinner("🔍 Detecting protected attributes..."):
    detection_results = detect_protected_attributes(df, target_col, correlation_threshold)

with st.spinner("📊 Running fairness analysis..."):
    fairness_results = run_fairness_analysis(df, target_col, detection_results)

with st.spinner("🧬 Applying SMOTE & validating quality..."):
    smote_results = apply_smote_with_validation(df, target_col, detection_results, smote_strategy)

with st.spinner("✍️ Generating bias narratives..."):
    narratives = generate_bias_narrative(df, detection_results, fairness_results, smote_results)

# ═══════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Attribute Detection",
    "📊 Fairness Metrics",
    "🧬 SMOTE Rebalancing",
    "📝 Bias Narratives",
    "📄 Compliance Report",
])

# ─── TAB 1: Attribute Detection ───────────────────────────────────
with tab1:
    st.markdown('<div class="section-header">🔍 Protected Attribute & Proxy Variable Detection</div>', unsafe_allow_html=True)

    protected = detection_results.get("protected", [])
    proxies   = detection_results.get("proxies", [])
    safe      = detection_results.get("safe", [])

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="label">Protected Attrs</div><div class="value" style="color:#ff6b6b">{len(protected)}</div><div class="sub">Directly flagged</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="label">Proxy Variables</div><div class="value" style="color:#ffa94d">{len(proxies)}</div><div class="sub">Indirect discrimination risk</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="label">Clear Columns</div><div class="value" style="color:#51cf66">{len(safe)}</div><div class="sub">Low risk</div></div>', unsafe_allow_html=True)

    st.markdown("#### 🚨 Protected Attributes Detected")
    if protected:
        for attr in protected:
            badge = "high" if attr["confidence"] > 0.75 else "medium"
            st.markdown(f"""
            <div class="flag-card">
                <strong style="color:#ff6b6b">{attr['column']}</strong>
                &nbsp;&nbsp;<span class="badge-{badge}">Confidence: {attr['confidence']:.0%}</span>
                <br><small style="color:#aaa">Reason: {attr['reason']}</small>
                <br><small style="color:#9ecfdf">Values: {attr.get('sample_values', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No obvious protected attributes detected by name analysis.")

    st.markdown("#### ⚠️ Proxy Variables (Potential Indirect Discrimination)")
    if proxies:
        for proxy in proxies:
            st.markdown(f"""
            <div class="flag-card proxy">
                <strong style="color:#ffa94d">{proxy['column']}</strong>
                &nbsp;&nbsp;<span class="badge-medium">Correlation: {proxy['correlation']:.2f} with {proxy['correlated_with']}</span>
                <br><small style="color:#aaa">{proxy['reason']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No proxy variables detected above the correlation threshold.")

    # Correlation heatmap
    st.markdown("#### 🔥 Correlation Heatmap")
    import plotly.graph_objects as go
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        corr = num_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, text=corr.values.round(2),
            texttemplate="%{text}", hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>",
        ))
        fig.update_layout(
            template="plotly_dark", height=500,
            title="Feature Correlation Matrix (Numeric Columns)",
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ─── TAB 2: Fairness Metrics ──────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">📊 Model-Integrated Fairness Analysis</div>', unsafe_allow_html=True)

    if fairness_results.get("error"):
        st.warning(f"Fairness analysis issue: {fairness_results['error']}")
    else:
        before = fairness_results.get("before", {})
        after  = fairness_results.get("after", {})

        st.markdown("#### Before vs. After SMOTE Rebalancing")
        metrics_to_show = ["demographic_parity_diff", "equalized_odds_diff", "accuracy", "model_accuracy"]

        cols = st.columns(len(metrics_to_show))
        metric_labels = {
            "demographic_parity_diff": ("Demographic Parity Diff", "lower is fairer"),
            "equalized_odds_diff":     ("Equalized Odds Diff", "lower is fairer"),
            "accuracy":                ("Overall Accuracy", "model accuracy"),
            "model_accuracy":          ("Model Accuracy", "on test set"),
        }

        for i, key in enumerate(metrics_to_show):
            bval = before.get(key)
            aval = after.get(key)
            label, sub = metric_labels.get(key, (key, ""))
            if bval is not None:
                delta_str = ""
                if aval is not None:
                    delta = aval - bval
                    arrow = "▲" if delta > 0 else "▼"
                    delta_str = f"<div class='sub'>{arrow} {abs(delta):.3f} after SMOTE</div>"
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">{label}</div>
                        <div class="value">{bval:.3f}</div>
                        <div class="sub">{sub}</div>
                        {delta_str}
                    </div>
                    """, unsafe_allow_html=True)

        # Bar chart: fairness metric comparison
        if before and after:
            import plotly.graph_objects as go
            shared_keys = [k for k in before if k in after and isinstance(before[k], (int, float))]
            if shared_keys:
                fig2 = go.Figure(data=[
                    go.Bar(name="Before SMOTE", x=shared_keys, y=[before[k] for k in shared_keys], marker_color="#ff6b6b"),
                    go.Bar(name="After SMOTE",  x=shared_keys, y=[after[k]  for k in shared_keys], marker_color="#51cf66"),
                ])
                fig2.update_layout(
                    barmode="group", template="plotly_dark",
                    title="Fairness Metrics: Before vs After SMOTE",
                    height=400, margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Per-subgroup breakdown
        subgroup = fairness_results.get("subgroup_breakdown")
        if subgroup:
            st.markdown("#### Subgroup Outcome Rates")
            sg_df = pd.DataFrame(subgroup)
            st.dataframe(sg_df, use_container_width=True)

# ─── TAB 3: SMOTE Rebalancing ─────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">🧬 SMOTE Rebalancing & Quality Validation</div>', unsafe_allow_html=True)

    if smote_results.get("error"):
        st.warning(f"SMOTE issue: {smote_results['error']}")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="label">Original Samples</div><div class="value">{smote_results.get("original_count", "N/A"):,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="label">Synthetic Samples</div><div class="value" style="color:#00e5c8">{smote_results.get("synthetic_count", 0):,}</div></div>', unsafe_allow_html=True)
        with c3:
            quality = smote_results.get("quality_score", 0)
            color = "#51cf66" if quality > 0.7 else "#ffa94d" if quality > 0.4 else "#ff6b6b"
            st.markdown(f'<div class="metric-card"><div class="label">SMOTE Quality Score</div><div class="value" style="color:{color}">{quality:.2f}</div><div class="sub">Distribution similarity</div></div>', unsafe_allow_html=True)

        # Class distribution chart
        dist_before = smote_results.get("class_dist_before", {})
        dist_after  = smote_results.get("class_dist_after", {})
        if dist_before and dist_after:
            import plotly.graph_objects as go
            all_classes = sorted(set(list(dist_before.keys()) + list(dist_after.keys())))
            fig3 = go.Figure(data=[
                go.Bar(name="Before", x=[str(c) for c in all_classes], y=[dist_before.get(c, 0) for c in all_classes], marker_color="#ff6b6b"),
                go.Bar(name="After",  x=[str(c) for c in all_classes], y=[dist_after.get(c, 0)  for c in all_classes], marker_color="#51cf66"),
            ])
            fig3.update_layout(
                barmode="group", template="plotly_dark",
                title=f"Class Distribution Before vs After SMOTE (Target: {target_col})",
                height=380, margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Quality alerts & alternatives
        quality_alerts = smote_results.get("quality_alerts", [])
        if quality_alerts:
            st.markdown("#### ⚠️ SMOTE Quality Alerts")
            for alert in quality_alerts:
                st.warning(alert)

        alternatives = smote_results.get("alternatives", [])
        if alternatives:
            st.markdown("#### 💡 Alternative Strategies Recommended")
            for alt in alternatives:
                st.info(alt)

        # Synthetic vs real distribution comparison
        dist_compare = smote_results.get("distribution_comparison")
        if dist_compare:
            st.markdown("#### Synthetic vs Real Minority Sample Distributions")
            st.dataframe(pd.DataFrame(dist_compare), use_container_width=True)

# ─── TAB 4: Bias Narratives ───────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">📝 Plain-English Bias Narratives</div>', unsafe_allow_html=True)
    st.caption("AI-generated explanations of detected biases, proxy discrimination risks, and actionable recommendations.")

    for narrative in narratives:
        severity_color = {"HIGH": "#ff6b6b", "MEDIUM": "#ffa94d", "LOW": "#51cf66"}.get(narrative.get("severity", "LOW"), "#aaa")
        # Escape body text so special chars don't break HTML, then convert newlines to <br>
        body_escaped = html.escape(narrative.get("body", "")).replace("\n", "<br>")
        title_escaped = html.escape(narrative.get("title", "Finding"))
        severity_escaped = html.escape(narrative.get("severity", "INFO"))
        st.markdown(f"""
        <div style="margin-bottom:1.5rem; border-radius:10px; overflow:hidden;">
            <div style="background:{severity_color}22; padding:0.6rem 1rem; border-left:4px solid {severity_color};">
                <strong style="color:{severity_color}">[{severity_escaped}]</strong>
                &nbsp;<span style="color:#eee">{title_escaped}</span>
            </div>
            <div class="narrative-box" style="border-top:none; border-radius:0 0 10px 10px; white-space:normal;">
{body_escaped}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 5: Compliance Report ─────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">📄 Compliance Report</div>', unsafe_allow_html=True)

    if generate_report:
        with st.spinner("Generating PDF compliance report..."):
            try:
                pdf_bytes = generate_pdf_report(
                    df=df,
                    target_col=target_col,
                    detection_results=detection_results,
                    fairness_results=fairness_results,
                    smote_results=smote_results,
                    narratives=narratives,
                    active_regs=active_regs,
                )
                st.success("✅ Compliance report generated!")
                st.download_button(
                    label="📥 Download PDF Compliance Report",
                    data=pdf_bytes,
                    file_name="fairlens_compliance_report.pdf",
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")
                st.code(traceback.format_exc())

    # Inline summary table
    st.markdown("#### Regulatory Compliance Summary")
    reg_data = []
    before = fairness_results.get("before", {})
    dp = before.get("demographic_parity_diff", None)
    eo = before.get("equalized_odds_diff", None)

    thresholds = {
        "ECOA":          {"metric": "Demographic Parity Diff", "value": dp, "threshold": 0.10, "direction": "below"},
        "EEOC":          {"metric": "Demographic Parity Diff", "value": dp, "threshold": 0.20, "direction": "below"},
        "GDPR Art.22":   {"metric": "Equalized Odds Diff",     "value": eo, "threshold": 0.15, "direction": "below"},
        "HHS Healthcare":{"metric": "Demographic Parity Diff", "value": dp, "threshold": 0.10, "direction": "below"},
    }

    for reg in active_regs:
        info = thresholds.get(reg, {})
        val  = info.get("value")
        thr  = info.get("threshold")
        if val is not None and thr is not None:
            passed = val < thr
            status = "✅ PASS" if passed else "❌ FAIL"
            reg_data.append({
                "Regulation": reg,
                "Metric": info["metric"],
                "Current Value": f"{val:.3f}",
                "Required Threshold": f"< {thr}",
                "Status": status,
            })
        else:
            reg_data.append({
                "Regulation": reg,
                "Metric": info.get("metric", "N/A"),
                "Current Value": "N/A",
                "Required Threshold": f"< {info.get('threshold','N/A')}",
                "Status": "⚠️ INSUFFICIENT DATA",
            })

    if reg_data:
        st.dataframe(pd.DataFrame(reg_data), use_container_width=True, hide_index=True)
    else:
        st.info("Select regulations in the sidebar to see compliance status.")
