# app.py -- HR Attrition Prediction Dashboard (All Models + SHAP + PDF)

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
st.title("🚀 HR Attrition Prediction Dashboard")

# ===============================
# Load Models & Artifacts
# ===============================
MODEL_PATH = "models_joblib/"
TMP_DIR = "temp_charts"; os.makedirs(TMP_DIR, exist_ok=True)

def load_artifacts():
    models, scaler, trained_cols = {}, None, None
    try: models["Logistic Regression"] = joblib.load(os.path.join(MODEL_PATH,"logistic_attrition_model.joblib"))
    except: models["Logistic Regression"]=None
    try: models["Random Forest"] = joblib.load(os.path.join(MODEL_PATH,"random_forest_attrition_model.joblib"))
    except: models["Random Forest"]=None
    try: models["XGBoost"] = joblib.load(os.path.join(MODEL_PATH,"xgboost_attrition_model.joblib"))
    except: models["XGBoost"]=None
    try: scaler = joblib.load(os.path.join(MODEL_PATH,"scaler.joblib"))
    except: pass
    try: trained_cols = joblib.load(os.path.join(MODEL_PATH,"trained_columns.joblib"))
    except: pass
    validated = {
        "Logistic Regression":{"accuracy":0.75,"roc_auc":0.80},
        "Random Forest":{"accuracy":0.84,"roc_auc":0.77},
        "XGBoost":{"accuracy":0.86,"roc_auc":0.77}
    }
    return models, scaler, trained_cols, validated

models, scaler, trained_cols, validated_metrics = load_artifacts()

# ===============================
# File Upload
# ===============================
st.sidebar.header("⚙️ Settings")
uploaded_file = st.file_uploader("📂 Upload CSV (HRIS-style export)", type="csv")
sample_headers=["Age","Department","JobRole","MonthlyIncome","YearsAtCompany","OverTime"]
st.sidebar.download_button("Download Sample CSV Template",
    data=pd.DataFrame(columns=sample_headers).to_csv(index=False),
    file_name="sample_hris_upload.csv")

if not uploaded_file:
    st.info("⬆️ Upload a CSV to start (use template).")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
st.write("### Preview Uploaded Data")
st.dataframe(raw_df.head(), use_container_width=True)

# ===============================
# Feature Alignment
# ===============================
X = raw_df.copy()
for col in trained_cols:
    if col not in X.columns: X[col]=0
X = X[trained_cols]
X_scaled = scaler.transform(X)

# ===============================
# Predictions for all models
# ===============================
results={}
for name,model in models.items():
    if model is None: 
        results[name]={"pred":np.repeat(np.nan,len(X)),"prob":np.repeat(np.nan,len(X))}
        continue
    try:
        pred=model.predict(X_scaled); prob=model.predict_proba(X_scaled)[:,1]
    except: 
        pred=model.predict(X); prob=model.predict_proba(X)[:,1]
    results[name]={"pred":pred,"prob":prob}
    raw_df[f"{name}_Pred"]=pred; raw_df[f"{name}_Prob"]=prob

# ===============================
# Metrics & Comparison
# ===============================
summaries=[]
for name in results:
    pred,prob=results[name]["pred"],results[name]["prob"]
    if "Attrition" in raw_df.columns:
        gt=raw_df["Attrition"].map({"Yes":1,"No":0})
        try:
            acc=accuracy_score(gt,pred); auc=roc_auc_score(gt,prob); src="Calculated on uploaded data"
        except: acc,auc,src=np.nan,np.nan,"Insufficient labels"
    else:
        acc=validated_metrics[name]["accuracy"]; auc=validated_metrics[name]["roc_auc"]; src="Validated on test set"
    summaries.append({"Model":name,"Accuracy":acc,"ROC_AUC":auc,"Source":src})
summary_df=pd.DataFrame(summaries)

# Ensemble
ensemble_prob=np.nanmean(np.array([results[m]["prob"] for m in results if not np.isnan(results[m]["prob"]).all()]),axis=0)
ensemble_pred=(ensemble_prob>=0.5).astype(int)
total=len(raw_df); at_risk=int(np.nansum(ensemble_pred)); avg_prob=np.nanmean(ensemble_prob)*100

c1,c2,c3=st.columns(3)
c1.metric("Total Employees",total)
c2.metric("At-Risk (ensemble)",at_risk)
c3.metric("Avg Risk (ensemble)",f"{avg_prob:.1f}%")

st.write("### 📊 Model Comparison Summary")
st.dataframe(summary_df.style.format({"Accuracy":"{:.2f}","ROC_AUC":"{:.2f}"}),use_container_width=True)

# ===============================
# Visuals
# ===============================
# Donut
fig_d,ax_d=plt.subplots(figsize=(4,4))
ax_d.pie([at_risk,total-at_risk],labels=["At Risk","Safe"],autopct="%1.1f%%",
         colors=["#ef4444","#10b981"],startangle=90)
plt.Circle((0,0),0.7,fc='white'); ax_d.axis('equal')
st.pyplot(fig_d); fig_d.savefig(os.path.join(TMP_DIR,"donut.png"))

# Dept stacked
if "Department" in raw_df.columns:
    raw_df["_Ens"]=ensemble_pred
    dept=raw_df.groupby(["Department","_Ens"]).size().unstack(fill_value=0)
    fig_s,ax_s=plt.subplots(figsize=(7,4))
    dept.plot(kind="bar",stacked=True,ax=ax_s,color=["#10b981","#ef4444"])
    plt.xticks(rotation=30); st.pyplot(fig_s)
    fig_s.savefig(os.path.join(TMP_DIR,"dept.png"))

# ROC curves if labels
if "Attrition" in raw_df.columns:
    fig_r,ax_r=plt.subplots()
    gt=raw_df["Attrition"].map({"Yes":1,"No":0})
    for name in results:
        try:
            fpr,tpr,_=roc_curve(gt,results[name]["prob"])
            auc=summary_df[summary_df.Model==name].ROC_AUC.values[0]
            ax_r.plot(fpr,tpr,label=f"{name} AUC {auc:.2f}")
        except: pass
    ax_r.plot([0,1],[0,1],"k--"); ax_r.legend(); st.pyplot(fig_r)
    fig_r.savefig(os.path.join(TMP_DIR,"roc.png"))

# ===============================
# Feature Importance
# ===============================
st.write("### 🔑 Top Features")
feats={}
for name,model in models.items():
    if model is None: continue
    if hasattr(model,"coef_"):
        vals=model.coef_[0]; fn=np.array(trained_cols)
        df=pd.DataFrame({"feature":fn,"importance":vals}).assign(abs_imp=lambda d:d.importance.abs())
        feats[name]=df.sort_values("abs_imp",ascending=False).drop("abs_imp",1).head(10)
    elif hasattr(model,"feature_importances_"):
        vals=model.feature_importances_; fn=np.array(trained_cols)
        df=pd.DataFrame({"feature":fn,"importance":vals})
        feats[name]=df.sort_values("importance",ascending=False).head(10)
for n,f in feats.items(): st.write(f"**{n}**"); st.dataframe(f)

# ===============================
# SHAP for XGB
# ===============================
shap_bar=shap_swarm=shap_dep=shap_water=None
try:
    import shap
    if models["XGBoost"] is not None:
        st.write("### 🧩 SHAP Explainability (XGBoost)")
        idx=np.random.choice(len(X),min(200,len(X)),replace=False)
        Xs=X.iloc[idx]; explainer=shap.Explainer(models["XGBoost"],scaler.transform(Xs),feature_names=trained_cols)
        vals=explainer(scaler.transform(Xs))
        fig1=plt.figure(); shap.plots.bar(vals,show=False,max_display=15); st.pyplot(fig1)
        shap_bar=os.path.join(TMP_DIR,"shap_bar.png"); fig1.savefig(shap_bar)
        fig2=plt.figure(); shap.plots.beeswarm(vals,show=False,max_display=15); st.pyplot(fig2)
        shap_swarm=os.path.join(TMP_DIR,"shap_swarm.png"); fig2.savefig(shap_swarm)
except Exception as e: st.warning("SHAP skipped: "+str(e))

# ===============================
# PDF Export
# ===============================
class PDF(FPDF):
    def header(self): 
        if self.page_no()>1: self.set_font("Arial","B",12); self.cell(0,10,"HR Attrition Prediction Report",ln=True,align="C")
    def footer(self):
        self.set_y(-20); self.set_font("Arial","I",8); self.set_text_color(100,100,100)
        txt="Prepared with <3 by Amlan Mishra - HR Tech, People Analytics & C&B Specialist at KPMG India"
        safe=txt.encode("latin-1","replace").decode("latin-1"); self.multi_cell(0,10,safe,align="C")
        self.set_text_color(30,100,200); self.set_font("Arial","U",8)
        self.cell(0,10,"Connect on LinkedIn",ln=True,align="C",link="https://www.linkedin.com/in/amlan-mishra-7aa70894")

if st.button("📑 Generate Executive PDF"):
    pdf=PDF(); pdf.add_page()
    pdf.set_font("Arial","B",20); pdf.cell(0,15,"HR Attrition Prediction Report",ln=True,align="C")
    pdf.set_font("Arial","",12); pdf.cell(0,10,f"Generated {datetime.now().strftime('%d-%b-%Y %H:%M')}",ln=True,align="C")
    pdf.ln(10); pdf.multi_cell(0,8,"Executive Summary:\nThis report provides attrition risk predictions across Logistic Regression, Random Forest, and XGBoost models, with ensemble analysis and SHAP explainability.",align="L")
    # Metrics page
    pdf.add_page(); pdf.set_font("Arial","B",14); pdf.cell(0,10,"Key Metrics",ln=True)
    pdf.set_font("Arial","",12); pdf.cell(0,8,f"Total Employees: {total}",ln=True)
    pdf.cell(0,8,f"At-Risk Employees (Ensemble): {at_risk}",ln=True)
    pdf.cell(0,8,f"Avg Attrition Risk: {avg_prob:.1f}%",ln=True); pdf.ln(5)
    # Model comparison
    pdf.set_font("Arial","B",12); pdf.cell(0,8,"Model Comparison",ln=True)
    pdf.set_font("Arial","B",10)
    for h in ["Model","Accuracy","ROC AUC"]: pdf.cell(60,8,h,border=1,align="C")
    pdf.ln(8); pdf.set_font("Arial","",10)
    for _,r in summary_df.iterrows():
        pdf.cell(60,8,str(r.Model),border=1); pdf.cell(60,8,f"{r.Accuracy:.2f}",border=1,align="C"); pdf.cell(60,8,f"{r.ROC_AUC:.2f}",border=1,align="C"); pdf.ln(8)
    # Top 10 snapshot
    pdf.add_page(); pdf.set_font("Arial","B",12); pdf.cell(0,8,"Top 10 Employees (Ensemble)",ln=True)
    snap=pd.DataFrame({"JobRole":raw_df.get("JobRole",[""]*len(raw_df)),
        "Dept":raw_df.get("Department",[""]*len(raw_df)),"Age":raw_df.get("Age",[""]*len(raw_df)),
        "Income":raw_df.get("MonthlyIncome",[""]*len(raw_df)),"Risk%":(ensemble_prob*100).round(1),
        "Pred":np.where(ensemble_pred==1,"At Risk","Safe")})
    for _,row in snap.head(10).iterrows():
        pdf.cell(50,8,str(row.JobRole)[:20],1); pdf.cell(40,8,str(row.Dept)[:15],1)
        pdf.cell(20,8,str(row.Age),1,align="C"); pdf.cell(30,8,str(row.Income),1,align="R")
        if row["Risk%"]>50: pdf.set_text_color(200,30,30)
        pdf.cell(20,8,f"{row['Risk%']:.1f}",1,align="R"); pdf.set_text_color(0,0,0)
        pdf.cell(30,8,row.Pred,1,align="C"); pdf.ln(8)
    # Charts & SHAP
    for fn in ["donut.png","dept.png","roc.png","shap_bar.png","shap_swarm.png"]:
        p=os.path.join(TMP_DIR,fn)
        if os.path.exists(p): pdf.add_page(); pdf.image(p,x=20,y=30,w=170)
    buf=io.BytesIO(); pdf.output(buf); st.download_button("📥 Download PDF",data=buf.getvalue(),file_name="attrition_report.pdf",mime="application/pdf")
