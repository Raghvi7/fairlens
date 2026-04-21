# ⚖️ FairLens — AI Bias Detection & Fairness Audit Platform

A Streamlit dashboard that automatically detects bias in datasets, measures fairness metrics, rebalances with SMOTE, and generates regulatory compliance reports.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py

# 3. Open browser at http://localhost:8501
```

---

## 📂 Project Structure

```
fairness_audit/
├── app.py                        # Main Streamlit dashboard
├── requirements.txt
├── README.md
├── generate_sample_data.py       # Generate synthetic demo dataset
└── modules/
    ├── __init__.py
    ├── attribute_detector.py     # NLP + correlation-based protected attr detection
    ├── fairness_metrics.py       # Fairlearn model-integrated fairness testing
    ├── bias_narrative.py         # Plain-English bias explanations
    ├── smote_handler.py          # SMOTE with quality validation
    └── report_generator.py       # PDF compliance report generator
```

---

## ✨ Features

### 🔍 Protected Attribute Detection
- NLP keyword matching against a taxonomy of 10+ protected categories
- Value distribution inspection (binary skew, pattern matching)
- Correlation-based proxy variable detection with configurable threshold
- Confidence scores and explanations for every flagged column

### 📊 Model-Integrated Fairness Metrics
- Trains a baseline Logistic Regression on your dataset
- Measures **Demographic Parity Difference** and **Equalized Odds Difference** via `fairlearn`
- Compares metrics **before and after SMOTE** rebalancing
- Per-subgroup accuracy and positive outcome rate breakdown

### 📝 Plain-English Bias Narratives
- Human-readable explanations for each detected protected attribute
- Proxy discrimination analysis with legal citations
- Actionable remediation steps tailored to each finding

### 🧬 SMOTE Quality Validation
- Applies SMOTE for minority class rebalancing
- Validates synthetic sample quality using KL-divergence per feature
- Flags columns with poor synthetic distributions
- Recommends alternatives: class weighting, adversarial reweighting, ADASYN

### 📄 PDF Compliance Reports
- Maps each fairness metric to ECOA / EEOC / GDPR Art.22 / HHS Healthcare thresholds
- Format: `Metric: Value (Threshold: X) — PASS/FAIL`
- Includes dataset stats, methodology, findings, and remediation steps
- Downloadable directly from the dashboard

---

## 🗂️ Supported Regulations

| Regulation       | Jurisdiction   | Scope                    | Key Threshold        |
|------------------|---------------|--------------------------|----------------------|
| ECOA             | USA Federal   | Credit & lending         | DPD < 0.10           |
| EEOC             | USA Federal   | Employment & hiring      | DPD < 0.20 (4/5 rule)|
| GDPR Article 22  | European Union| Automated decisions      | DPD & EOD < 0.15     |
| HHS Healthcare   | USA Federal   | Clinical decision support| DPD < 0.10           |

---

## 🧪 Generate Sample Data

A sample hiring dataset is provided for demo purposes:

```bash
python generate_sample_data.py
# Outputs: sample_hiring_dataset.csv
```

Then upload this CSV to the dashboard to see all features in action.

---

## ⚙️ Configuration (Sidebar)

| Setting                        | Description                                              |
|-------------------------------|----------------------------------------------------------|
| Target Column                 | The outcome variable your model predicts (e.g., `hired`)|
| Proxy Correlation Threshold   | Flag columns correlated above this value with protected attrs |
| SMOTE Sampling Strategy       | auto / minority / not minority / all                    |
| Generate PDF Report           | Produce downloadable compliance PDF                      |
| Regulations                   | Select which regulations to check compliance against     |

---

## 📜 Disclaimer

FairLens is an automated analysis tool for developer education and regulatory assistance. It does not constitute legal advice. Consult qualified legal counsel before deploying AI systems in regulated domains.
