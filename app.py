# app.py -- HR Attrition Prediction Dashboard (Final Polished Version)
# Save this file as app.py in your repo root and deploy.

import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, io, warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from fpdf import FPDF
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")
st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

# -----------------------
# TITLE / BADGES / LINKS
# -----------------------
st.title("🚀 HR Attrition Prediction Dashboard")

st.markdown(
    """
    ![Python](https://img.shields.io/badge/Python-3.10-blue)
    ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
    ![Machine Learning](https://img.shields.io/badge/ML-LogReg%20%7C%20RF%20%7C%20XGB-green)
    ![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-yellow)
    """
)

st.markdown(
    """
    **By Amlan Mishra**  
    [![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?logo=github&style=for-the-badge)](https://github.com/AMBOT-pixel96/hr-tech-portfolio)  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin&style=for-the-badge)](https://www.linkedin.com/in/amlan-mishra-7aa70894)
    """
)

# -----------------------
# Paths & Directories
# -----------------------
MODEL_PATH = "models_joblib"
TMP_DIR = "temp_charts"
os.makedirs(TMP_DIR, exist_ok=True)

# -----------------------
# Load Models / Artifacts
# -----------------------
def load_artifacts():
    models = {"Logistic Regression": None, "Random Forest": None, "XGBoost": None}
    scaler = None
    trained_cols = None

    # Try loads; if missing, leave as None
    try:
        p = os.path.join(MODEL_PATH, "logistic_attrition_model.joblib")
        if os.path.exists(p): models["Logistic Regression"] = joblib.load(p)
    except Exception as e:
        st.warning(f"Could not load Logistic model: {e}")

    try:
        p = os.path.join(MODEL_PATH, "random_forest_attrition_model.joblib")
        if os.path.exists(p): models["Random Forest"] = joblib.load(p)
    except Exception as e:
        st.warning(f"Could not load Random Forest model: {e}")

    try:
        p = os.path.join(MODEL_PATH, "xgboost_attrition_model.joblib")
        if os.path.exists(p): models["XGBoost"] = joblib.load(p)
    except Exception as e:
        st.warning(f"Could not load XGBoost model: {e}")

    try:
        p = os.path.join(MODEL_PATH, "scaler.joblib")
        if os.path.exists(p): scaler = joblib.load(p)
    except Exception as e:
        st.info("Scaler not found or failed to load; scaling steps may error if required.")

    try:
        p = os.path.join(MODEL_PATH, "trained_columns.joblib")
        if os.path.exists(p): trained_cols = joblib.load(p)
    except Exception as e:
        st.info("trained_columns not found; ensure model feature alignment is available.")

    # Validated baseline metrics fallback
    validated = {
        "Logistic Regression": {"accuracy": 0.75, "roc_auc": 0.80},
        "Random Forest": {"accuracy": 0.84, "roc_auc": 0.77},
        "XGBoost": {"accuracy": 0.86, "roc_auc": 0.77},
    }

    return models, scaler, trained_cols, validated

models, scaler, trained_cols, validated_metrics = load_artifacts()

# -----------------------
# Upload CSV (main page)
# -----------------------
st.subheader("📂 Upload Employee Dataset")
st.markdown(
    "Upload CSV file (refer [Sample CSV Format](https://github.com/AMBOT-pixel96/hr-tech-portfolio/raw/main/data/processed_hr_data.csv))."
)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv", accept_multiple_files=False)

if not uploaded_file:
    st.info("⬆️ Upload a CSV to start. Use the sample link above for the expected columns.")
    st.stop()

# Load uploaded CSV
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Failed to read CSV: " + str(e))
    st.stop()

st.write("### Preview Uploaded Data")
st.dataframe(raw_df.head(), use_container_width=True)

# -----------------------
# Basic sanity checks
# -----------------------
if trained_cols is None:
    st.error("trained_columns.joblib not found in models_joblib/. This file is required for feature alignment and to run models. Upload it to the models_joblib folder.")
    st.stop()
if scaler is None:
    st.warning("Scaler not found. If your models require scaled inputs you may get errors or poor results. Add scaler.joblib to models_joblib/")

# -----------------------
# Feature alignment & scaling
# -----------------------
X_df = raw_df.copy()
# ensure all trained cols exist (create zeros for missing)
for c in trained_cols:
    if c not in X_df.columns:
        X_df[c] = 0
# keep only trained_cols (preserves order)
X_df = X_df[trained_cols]

# Attempt scaling if scaler available
try:
    X_scaled = scaler.transform(X_df)
except Exception:
    # fallback: use raw features if scaler not available
    X_scaled = X_df.values
    st.warning("Using unscaled features (scaler.transform failed). Models expecting scaled input may misbehave.")

# -----------------------
# Predictions for each model
# -----------------------
results = {}
for name, mdl in models.items():
    if mdl is None:
        results[name] = {"pred": np.full(len(X_df), np.nan), "prob": np.full(len(X_df), np.nan)}
        continue
    try:
        preds = mdl.predict(X_scaled)
        probs = mdl.predict_proba(X_scaled)[:, 1]
    except Exception:
        # try non-scaled input
        preds = mdl.predict(X_df.values)
        probs = mdl.predict_proba(X_df.values)[:, 1]
    results[name] = {"pred": preds, "prob": probs}
    # append back to raw_df for convenience
    raw_df[f"{name}_Pred"] = preds
    raw_df[f"{name}_Prob"] = probs

# -----------------------
# Metrics & summary table
# -----------------------
summary_rows = []
for name in results:
    pred = results[name]["pred"]
    prob = results[name]["prob"]
    if "Attrition" in raw_df.columns:
        gt = raw_df["Attrition"].map({"Yes": 1, "No": 0})
        try:
            acc = accuracy_score(gt, pred)
            auc = roc_auc_score(gt, prob)
            source = "Calculated on uploaded data"
        except Exception:
            acc, auc, source = np.nan, np.nan, "Insufficient labels"
    else:
        acc = validated_metrics[name]["accuracy"]
        auc = validated_metrics[name]["roc_auc"]
        source = "Validated on test set"
    summary_rows.append({"Model": name, "Accuracy": acc, "ROC_AUC": auc, "Source": source})

summary_df = pd.DataFrame(summary_rows)

# -----------------------
# Ensemble (mean of available model probs)
# -----------------------
prob_arrays = [results[m]["prob"] for m in results if not np.isnan(results[m]["prob"]).all()]
if prob_arrays:
    ensemble_prob = np.nanmean(np.vstack(prob_arrays), axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
else:
    ensemble_prob = np.full(len(X_df), np.nan)
    ensemble_pred = np.full(len(X_df), np.nan)

total = len(raw_df)
at_risk = int(np.nansum(ensemble_pred)) if not np.isnan(ensemble_pred).all() else 0
avg_prob = float(np.nanmean(ensemble_prob)) * 100 if not np.isnan(ensemble_prob).all() else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Total Employees", total)
c2.metric("At-Risk (ensemble)", at_risk)
c3.metric("Avg Risk (ensemble)", f"{avg_prob:.1f}%")

st.write("### 📊 Model Comparison Summary")
st.dataframe(summary_df.style.format({"Accuracy": "{:.2f}", "ROC_AUC": "{:.2f}"}), use_container_width=True)

# -----------------------
# Model comparison chart (Accuracy vs ROC AUC)
# -----------------------
try:
    # Ensure numeric and present
    summary_plot_df = summary_df.melt(id_vars=["Model"], value_vars=["Accuracy", "ROC_AUC"],
                                      var_name="Metric", value_name="Score")
    fig_cmp, ax_cmp = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Model", y="Score", hue="Metric", data=summary_plot_df, ax=ax_cmp)
    ax_cmp.set_ylim(0, 1)
    ax_cmp.set_ylabel("Score (0 - 1)")
    ax_cmp.set_title("Model Comparison: Accuracy vs ROC AUC")
    st.pyplot(fig_cmp)
    fig_cmp.savefig(os.path.join(TMP_DIR, "model_comparison.png"), bbox_inches="tight", dpi=200)
except Exception as e:
    st.warning("Model comparison chart skipped: " + str(e))

# -----------------------
# Visuals: Donut, Dept stacked, ROC
# -----------------------
# Donut
fig_d, ax_d = plt.subplots(figsize=(4, 4))
safe_count = total - at_risk
ax_d.pie([at_risk, safe_count], labels=["At Risk", "Safe"], autopct="%1.1f%%",
         colors=["#ef4444", "#10b981"], startangle=90)
circle = plt.Circle((0, 0), 0.7, fc="white")
ax_d.add_artist(circle)
ax_d.axis("equal")
ax_d.set_title("At Risk vs Safe (Ensemble)")
st.pyplot(fig_d)
fig_d.savefig(os.path.join(TMP_DIR, "donut_chart.png"), bbox_inches="tight", dpi=200)

# Department stacked
if "Department" in raw_df.columns:
    raw_df["_ens"] = ensemble_pred
    dept = raw_df.groupby(["Department", "_ens"]).size().unstack(fill_value=0)
    fig_s, ax_s = plt.subplots(figsize=(8, 4))
    dept.plot(kind="bar", stacked=True, ax=ax_s, color=["#10b981", "#ef4444"])
    ax_s.set_xlabel("Department")
    ax_s.set_ylabel("Count")
    ax_s.set_title("Department-wise Safe vs At Risk (Ensemble)")
    plt.xticks(rotation=30)
    st.pyplot(fig_s)
    fig_s.savefig(os.path.join(TMP_DIR, "dept_stacked.png"), bbox_inches="tight", dpi=200)

# ROC curves (if labels exist)
if "Attrition" in raw_df.columns:
    try:
        fig_r, ax_r = plt.subplots(figsize=(6, 5))
        gt = raw_df["Attrition"].map({"Yes": 1, "No": 0})
        plotted = False
        for name in results:
            probs = results[name]["prob"]
            if not np.isnan(probs).all():
                fpr, tpr, _ = roc_curve(gt, probs)
                auc_val = summary_df.loc[summary_df.Model == name, "ROC_AUC"].values
                auc_val = auc_val[0] if len(auc_val) else np.nan
                ax_r.plot(fpr, tpr, label=f"{name} (AUC {auc_val:.2f})")
                plotted = True
        if plotted:
            ax_r.plot([0, 1], [0, 1], "k--")
            ax_r.set_xlabel("False Positive Rate")
            ax_r.set_ylabel("True Positive Rate")
            ax_r.set_title("ROC Curves")
            ax_r.legend()
            st.pyplot(fig_r)
            fig_r.savefig(os.path.join(TMP_DIR, "roc_curve.png"), bbox_inches="tight", dpi=200)
    except Exception as e:
        st.warning("ROC chart error: " + str(e))

# -----------------------
# Feature importances / coefficients
# -----------------------
st.write("### 🔑 Top Features (per model)")
feats = {}
for name, mdl in models.items():
    if mdl is None:
        continue
    try:
        if hasattr(mdl, "coef_"):
            vals = mdl.coef_[0]
            fn = np.array(trained_cols)
            df = pd.DataFrame({"feature": fn, "importance": vals})
            df["abs_imp"] = df.importance.abs()
            df = df.sort_values("abs_imp", ascending=False).drop("abs_imp", axis=1).head(10)
            feats[name] = df
        elif hasattr(mdl, "feature_importances_"):
            vals = mdl.feature_importances_
            fn = np.array(trained_cols)
            df = pd.DataFrame({"feature": fn, "importance": vals})
            df = df.sort_values("importance", ascending=False).head(10)
            feats[name] = df
    except Exception as e:
        st.warning(f"Feature importance failed for {name}: {e}")

for n, f in feats.items():
    st.write(f"**{n}**")
    st.dataframe(f, use_container_width=True)

# -----------------------
# SHAP (XGBoost only, subsampled)
# -----------------------
shap_bar = shap_swarm = None
try:
    import shap
    if models.get("XGBoost") is not None:
        st.write("### 🧩 SHAP Explainability (XGBoost)")
        # subsample for speed
        idx = np.random.choice(range(len(X_df)), size=min(200, len(X_df)), replace=False)
        Xs = X_df.iloc[idx]
        # shap expects original feature array passed to explainer if using wrapped model
        try:
            explainer = shap.Explainer(models["XGBoost"], Xs)
            shap_values = explainer(Xs)
        except Exception:
            # fallback - try with scaled data if needed
            explainer = shap.Explainer(models["XGBoost"], scaler.transform(Xs))
            shap_values = explainer(scaler.transform(Xs))
        # Global bar
        fig_shap_bar = plt.figure(figsize=(7, 4))
        shap.plots.bar(shap_values, show=False, max_display=15)
        st.pyplot(fig_shap_bar)
        shap_bar = os.path.join(TMP_DIR, "shap_bar.png")
        fig_shap_bar.savefig(shap_bar, bbox_inches="tight", dpi=200)
        # Beeswarm (may be slow)
        fig_shap_swarm = plt.figure(figsize=(7, 5))
        shap.plots.beeswarm(shap_values, show=False, max_display=15)
        st.pyplot(fig_shap_swarm)
        shap_swarm = os.path.join(TMP_DIR, "shap_swarm.png")
        fig_shap_swarm.savefig(shap_swarm, bbox_inches="tight", dpi=200)
except Exception as e:
    st.info("SHAP not available or failed: " + str(e))

# -----------------------
# Download images for PPT decks
# -----------------------
st.subheader("📥 Download images to plug into PPT decks")
download_map = {
    "Donut Chart (At Risk vs Safe)": "donut_chart.png",
    "Department-wise Risk (stacked)": "dept_stacked.png",
    "ROC Curve": "roc_curve.png",
    "Model Comparison (Acc vs ROC AUC)": "model_comparison.png",
    "SHAP Bar Importance": "shap_bar.png",
    "SHAP Beeswarm": "shap_swarm.png"
}
for label, fn in download_map.items():
    p = os.path.join(TMP_DIR, fn)
    if os.path.exists(p):
        with open(p, "rb") as f:
            st.download_button(f"⬇️ {label}", f.read(), file_name=fn, mime="image/png")
    else:
        st.write(f"- {label}: (not available)")

# -----------------------
# Polished PDF Export
# -----------------------
class PDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, "HR Attrition Prediction Report", ln=True, align="C")

    def footer(self):
        self.set_y(-20)
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 100, 100)
        txt = "Prepared with <3 by Amlan Mishra - HR Tech, People Analytics & C&B Specialist at KPMG India"
        safe = txt.encode("latin-1", "replace").decode("latin-1")
        self.multi_cell(0, 10, safe, align="C")
        self.set_text_color(30, 100, 200)
        self.set_font("Arial", "U", 8)
        self.cell(0, 10, "Connect on LinkedIn", ln=True, align="C", link="https://www.linkedin.com/in/amlan-mishra-7aa70894")

if st.button("📑 Generate Polished PDF"):
    pdf = PDF()
    pdf.add_page()

    # Cover + Executive Summary
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, "HR Attrition Prediction Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}", ln=True, align="C")
    pdf.ln(12)

    pdf.set_font("Arial", "", 11)
    exec_text = (
        "Executive Summary:\n\n"
        "This report provides attrition risk predictions using Logistic Regression, Random Forest, "
        "and XGBoost models. An ensemble view combines their strengths. SHAP explainability "
        "highlights drivers of attrition so HR leaders can move from predictions to actionable decisions.\n\n"
        "Business context:\n"
        "- Attrition costs companies in productivity, rehiring and retraining.\n"
        "- Predictive analytics enables proactive retention actions.\n"
        "- Explainability builds trust and supports evidence-based HR decisions.\n"
    )
    pdf.multi_cell(0, 7, exec_text, align="L")

    # Key metrics
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Key Metrics", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Total Employees: {total}", ln=True)
    pdf.cell(0, 8, f"At-Risk Employees (Ensemble): {at_risk}", ln=True)
    pdf.cell(0, 8, f"Avg Attrition Risk (Ensemble): {avg_prob:.1f}%", ln=True)
    pdf.ln(6)

    # Model comparison table (shaded header)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Model Comparison", ln=True)
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font("Arial", "B", 10)
    headers = ["Model", "Accuracy", "ROC AUC", "Source"]
    for h in headers:
        pdf.cell(45, 8, h, 1, 0, "C", fill=True)
    pdf.ln(8)
    pdf.set_font("Arial", "", 10)
    for _, r in summary_df.iterrows():
        pdf.cell(45, 8, str(r.Model), 1)
        pdf.cell(45, 8, f"{r.Accuracy:.2f}", 1, 0, "C")
        pdf.cell(45, 8, f"{r.ROC_AUC:.2f}", 1, 0, "C")
        pdf.cell(45, 8, str(r.Source), 1, 0, "C")
        pdf.ln(8)

    # Top-10 snapshot table
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Top 10 Employees (Ensemble)", ln=True)
    pdf.set_fill_color(220, 220, 220)
    cols = ["JobRole", "Dept", "Age", "Income", "Risk%", "Prediction"]
    for c in cols:
        pdf.cell(30, 8, c, 1, 0, "C", fill=True)
    pdf.ln(8)
    pdf.set_font("Arial", "", 9)

    snap_df = pd.DataFrame({
        "JobRole": raw_df.get("JobRole", [""] * len(raw_df)),
        "Dept": raw_df.get("Department", [""] * len(raw_df)),
        "Age": raw_df.get("Age", [""] * len(raw_df)),
        "Income": raw_df.get("MonthlyIncome", [""] * len(raw_df)),
        "Risk%": (ensemble_prob * 100).round(1),
        "Prediction": np.where(ensemble_pred == 1, "At Risk", "Safe")
    })

    for _, row in snap_df.head(10).iterrows():
        pdf.cell(30, 8, str(row.JobRole)[:12], 1)
        pdf.cell(30, 8, str(row.Dept)[:10], 1)
        pdf.cell(30, 8, str(row.Age), 1, 0, "C")
        pdf.cell(30, 8, str(row.Income), 1, 0, "R")
        if row["Risk%"] > 50:
            pdf.set_text_color(200, 30, 30)
        pdf.cell(30, 8, f"{row['Risk%']:.1f}", 1, 0, "R")
        pdf.set_text_color(0, 0, 0)
        pdf.cell(30, 8, row.Prediction, 1, 0, "C")
        pdf.ln(8)

    # Charts (include model comparison)
    chart_list = [
        ("donut_chart.png", "At Risk vs Safe (Ensemble)"),
        ("dept_stacked.png", "Department-wise Risk"),
        ("roc_curve.png", "ROC Curves"),
        ("model_comparison.png", "Model Comparison (Accuracy vs ROC AUC)"),
        ("shap_bar.png", "SHAP Global Importance"),
        ("shap_swarm.png", "SHAP Beeswarm")
    ]
    for fn, title in chart_list:
        p = os.path.join(TMP_DIR, fn)
        if os.path.exists(p):
            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, title, ln=True, align="C")
            pdf.image(p, x=20, y=40, w=170)

    # Export download
    buf = io.BytesIO()
    pdf.output(buf, "S")
    st.download_button(
        "📥 Download Polished PDF",
        data=buf.getvalue(),
        file_name="attrition_report_polished.pdf",
        mime="application/pdf",
    )

# -----------------------
# END
# -----------------------