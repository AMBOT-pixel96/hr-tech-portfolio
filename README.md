# HR Tech Portfolio
---
> 🧑‍💻 HR Tech & People Analytics Portfolio — from descriptive insights → predictive models → explainable AI.  
> 🌍 Live Demo: 
[AttritionDashboard](https://hr-tech-portfolio.streamlit.app/)
---
![PDF Available](https://img.shields.io/badge/PDF-Available-brightgreen?logo=adobeacrobatreader)
---

## 📑 Reports Panel

[![Download Latest Report](https://img.shields.io/badge/PDF-Download%20Latest-brightgreen?style=for-the-badge&logo=adobeacrobatreader)](https://github.com/AMBOT-pixel96/hr-tech-portfolio/raw/main/reports/Attrition_Project_Summary.pdf?download=1)
[![View in Repo](https://img.shields.io/badge/View-Reports-blue?style=for-the-badge&logo=github)](reports/)
[![Download All](https://img.shields.io/badge/ZIP-Download%20All-orange?style=for-the-badge&logo=files)](https://github.com/AMBOT-pixel96/hr-tech-portfolio/archive/refs/heads/main.zip)

---

![Python](https://img.shields.io/badge/Python-3.9-blue)  
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)  
![Pandas](https://img.shields.io/badge/Library-Pandas-150458?logo=pandas)  
![Seaborn](https://img.shields.io/badge/Library-Seaborn-3776AB)  
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-11557c)  
![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-green)  

This repository showcases my journey into **HR Tech & People Analytics**.

This repository showcases my journey into **HR Tech & People Analytics**.  
It contains hands-on projects where I apply **Python, Pandas, Seaborn, and People Analytics concepts** to real-world HR data.

---

## 📋 Project Overview  

| Project | Notebook | Objective | Key Results |
|---------|----------|------------|-------------|
| **1. Attrition Risk Analyzer (v1.0)** | [Day4-AttritionRiskAnalyzer.ipynb](notebooks/Day4-AttritionRiskAnalyzer.ipynb) | Descriptive analytics of attrition patterns (age, dept, job role) | Attrition rate **16.1%**, highest in Sales Reps (39%) |
| **2. Predictive Attrition Model (Logistic v2.0 / v3.0)** | [Attrition_PredictiveModel_V2.ipynb](notebooks/Attrition_PredictiveModel_V2.ipynb)<br>[Attrition_PredictiveModel_V3.ipynb](notebooks/Attrition_PredictiveModel_V3.ipynb) | Baseline + tuned Logistic Regression | Accuracy ~**75%**, ROC AUC ~**0.80**, key drivers: Overtime, JobRole, Marital Status |
| **3. HR Data Cleaning Utility (Sidequest)** | [HR_Data_Cleaning_Utility_V1.ipynb](sidequests/HR_Data_Cleaning_Utility_V1.ipynb) | Cleaning messy HR data (duplicates, missing values, casing) | Automated pipeline → [cleaned_hr_data.csv](data/cleaned_hr_data.csv) |
| **4. Model Comparison (Logistic vs Random Forest)** | [Attrition_ModelComparision.ipynb](notebooks/Attrition_ModelComparision.ipynb) | Compare interpretability vs non-linear power | Logistic: **75%** acc, RF: **83%** acc, RF captures income + age patterns |
| **5. Advanced Models (Tuned RF + XGBoost)** | [Attrition_AdvancedModels.ipynb](notebooks/Attrition_AdvancedModels.ipynb) | Hyperparameter tuned RF + XGBoost benchmarking | XGBoost best performer → **86.4% acc**, ROC AUC **0.774** |
| **6. Explainability with SHAP** | [Attrition_ModelExplainability.ipynb](notebooks/Attrition_ModelExplainability.ipynb) | Add global & local interpretability (XGBoost) | Global drivers: Overtime, JobRole, Income; Local plots explain individuals |

---

## 📂 Repository Structure  

<!-- REPO_TREE_START -->
```text
├── README.md
├── app.py
├── data
│   ├── cleaned_hr_data.csv
│   ├── logistic_top_features.csv
│   ├── messy_hr_data.csv
│   ├── model_comparison_results.csv
│   ├── processed_hr_data.csv
│   ├── rf_top_features.csv
│   └── xgb_top_features.csv
├── images
│   ├── Attrition_by_dept.jpg
│   ├── attrition_by_age.jpg
│   ├── attrition_by_jobrole.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_xgb.png
│   ├── missing_values_after.png
│   ├── missing_values_before.png
│   ├── missing_values_collage.png
│   ├── model_comparision.png
│   ├── model_comparison_barplot.png
│   ├── roc_curve_comparison.png
│   ├── screenshot1.png
│   ├── shap_dependence_plot.png
│   ├── shap_feature_importance.png
│   ├── shap_local_bar_plot.png
│   ├── shap_summary_plot.png
│   ├── shap_waterfall_plot.png
│   ├── top_features.png
│   ├── top_features_post_enhancement.png
│   ├── top_features_tuned.png
│   ├── top_features_xgb.png
│   └── top_features_xgb_heatmap.png
├── models
│   ├── Attrition_AdvancedModels.ipynb
│   ├── logistic_attrition_model.pkl
│   ├── logistic_attrition_model_tuned.pkl
│   ├── placeholder
│   ├── random_forest_attrition_model.pkl
│   ├── random_forest_tuned.pkl
│   ├── scaler.pkl
│   ├── trained_columns.pkl
│   └── xgboost_attrition_model.pkl
├── models_joblib
│   └── test.txt
├── notebooks
│   ├──  Attrition_PredictiveModel.ipynb
│   ├── Attrition_AdvancedModels.ipynb
│   ├── Attrition_ModelComparision.ipynb
│   ├── Attrition_ModelExplainability.ipynb
│   ├── Attrition_PredictiveModel_V2.ipynb
│   ├── Attrition_PredictiveModel_V3.ipynb
│   ├── Attrition_Streamlit_Dashboard.ipynb
│   ├── Day4-AttritionRiskAnalyzer.ipynb
│   ├── Day4-AttritionRiskAnalyzer_v2.0.ipynb
│   ├── baby_steps
│   │   ├── Day1-Hello-Amlan.ipynb
│   │   ├── Day2-Basics.ipynb
│   │   └── Day3-DataTypesAndControl.ipynb
│   └── baby_steps.gitkeep
├── reports
│   ├── Attrition_AdvancedModels_aa41c1310f907082a92a9e51b94010b5.pdf
│   ├── Attrition_ModelComparision_009086358233572ffb338cf8f4ee6ab7.pdf
│   ├── Attrition_ModelExplainability_52d61660976d1481ab051617939779a2.pdf
│   ├── Attrition_Project_Summary.pdf
│   ├── Day4-AttritionRiskAnalyzer_v2.0_058b58b9bf5e96ac0c1bdaac1e7abdcc.pdf
│   ├── HR_Data_Cleaning_Utility_V1_260622ce.pdf
│   ├── HR_Data_Cleaning_Utility_V1_6501dfa8.pdf
│   ├── HR_Data_Cleaning_Utility_V1_86b38b95.pdf
│   └── HR_Data_Cleaning_Utility_V1_87664d4cc3eefa67f9fe549d33ef7c57.pdf
├── requirements.txt
├── runtime.txt
├── scripts
│   ├── export_pdf.py
│   └── gen_tree.py
└── sidequests
    └── HR_Data_Cleaning_Utility_V1.ipynb
```
<!-- REPO_TREE_END -->

---

## 📊 Project 1: Attrition Risk Analyzer (v1.0)

### v1.0 — Descriptive Insights
**Objective:**  
Analyze IBM’s HR Attrition dataset to identify patterns of employee attrition, create a risk flag, and visualize insights.

---

**Why This Matters (Business Context):**  
Employee attrition directly impacts business costs through lost productivity, rehiring, and retraining.  
By identifying high-risk roles and departments, organizations can proactively design retention strategies,  
reduce turnover costs, and improve workforce stability.

---

**Key Steps:**  
1. Load and explore the dataset (1470 employees, 35 features).  
2. Analyze attrition distribution overall and by job role.  
3. Create a binary attrition risk flag (Yes = 1, No = 0).  
4. Build visualizations:
   - Overall attrition counts  
   - Attrition by department  
   - Attrition by age distribution  
   - Attrition % by job role  

📓 [View the Jupyter Notebook](notebooks/Day4-AttritionRiskAnalyzer.ipynb)  

---

### v2.0 — Predictive Modeling (Logistic Regression)  

**Objective:**  
Move from descriptive analytics → predictive insights by using **Logistic Regression** to forecast employee attrition risk. 

**Key Steps:**  
1. Data preprocessing  
   - Dropped irrelevant features (EmployeeNumber, Over18, etc.)  
   - Encoded categorical variables via one-hot encoding  
   - Scaled numeric features  
   - Train-test split (70/30)  
2. Model training with Logistic Regression (`scikit-learn`)  
3. Model evaluation  
   - Accuracy score  
   - Confusion matrix  
   - Classification report (precision, recall, F1)  
   - ROC-AUC score  
4. Feature importance analysis → which factors most influence attrition risk  
5. Saved trained model + scaler into `/models/`  
6. Fixed Data Leakage issue and added enhancements
   
📓 [View the Predictive Notebook](notebooks/Attrition_PredictiveModel_V2.ipynb)  

---


## 🖼️ Visuals & Outputs  

### Descriptive Analytics (v1.0)
# Attrition by Age:

![Attrition by Age](images/attrition_by_age.jpg)  

*Attrition counts (Yes/No) by Age*  

# Attrition by Department:  

![Attrition by Department](images/Attrition_by_dept.jpg)  

*Attrition counts (Yes/No) per Department*  

# Attrition % by Job Role:  

![Attrition % by Job Role](images/attrition_by_jobrole.png)  

*Percentage of employees leaving by Job Role*  

### Predictive Analytics (v2.0)  
![Confusion Matrix](images/confusion_matrix.png)  
*Confusion Matrix — Logistic Regression performance*  

![Top Features](images/top_features.png)  
*Top 10 features influencing attrition risk* 

![Top Features After Enhancement of Model](images/top_features_post_enhancement.png)  
*Top 10 features influencing attrition risk - Enhanced Model*

---

## 🔍 Key Insights  

- Overall attrition rate: **16.1%**  
- Job roles with highest attrition:  
  - Sales Representatives → 39% attrition (83 out of 220 left)  
  - Laboratory Technicians → 23% attrition (62 out of 259 left)  
- Lowest attrition: **R&D (13.8%)** → strongest retention  
- Highest attrition: **Sales (20.6%)** → weakest retention  
- HR is small (63 employees), but attrition rate is relatively high (19%)  
- Age groups younger than ~30 show elevated risk compared to older cohorts  

**From Predictive Model:**  
- Logistic Regression achieved ~`75.9%` accuracy, ROC-AUC = `0.817`.  
- Key positive attrition drivers: *Overtime, JobRole_SalesRep, MaritalStatus_Single, etc.*  
- Key retention drivers: *JobLevel, YearsAtCompany, MonthlyIncome*. 

---
## Project 2- Predictive Attrition Model (v3.0) - Tuned Models

**Objective:**  
Enhance the baseline Logistic Regression model by applying **cross-validation** and **hyperparameter tuning** to improve stability and interpretability.  

**Key Steps:**  
1. Split dataset into training & test sets (80/20)  
2. Standardized numeric features with `StandardScaler`  
3. Trained baseline Logistic Regression (Day-5)  
4. Applied **5-fold cross-validation** to validate performance consistency  
5. Used **GridSearchCV** to tune hyperparameters (`C`, `penalty`, `solver`)  
6. Compared tuned vs baseline performance  

**Results:**  
- Baseline Accuracy: **XX%**  
- Tuned CV Accuracy: **XX%**  
- ROC AUC: **XX**  
- Best Parameters: `{ 'C': X, 'penalty': 'l1', 'solver': 'liblinear' }`  

**Sample Visuals:**  

Top Features Driving Attrition:  
![Top Features](images/top_features_tuned.png)  

**Insights:**  
- OverTime, Laboratory Technician roles, and Frequent Travel rank as top predictors.  
- Hyperparameter tuning improved model generalization, reducing overfitting risk.  
- Cross-validation confirmed stability of results across folds.
**Model Artifacts:**  
[logistic_top_features.csv](data/logistic_top_features.csv)
  
📓 [View the Predictive Notebook](notebooks/Attrition_PredictiveModel_V3.ipynb)

---

## 📊 Project 3: HR Data Cleaning Utility (v1.0) (Side Quest-1)

This notebook demonstrates how to **simulate messy HR data** and then build a cleaning pipeline to make it analysis-ready.  
Data cleaning is a critical step in People Analytics — poor quality data = misleading insights.
# ✅ Conclusions

- Automated pipeline successfully cleaned the dataset.  
- Issues fixed: duplicates, missing values, inconsistent casing, invalid dates.  
- Outputs saved in:
  - [messy_hr_data.csv](data/messy_hr_data.csv)  
  - [cleaned_hr_data.csv](data/cleaned_hr_data.csv)  

**Hero Visual: Data Cleaning Impact**  

![Before vs After Cleaning](images/missing_values_collage.png)

📓 [View the Utility Notebook](sidequests/HR_Data_Cleaning_Utility_V1.ipynb)  


## 📊 Project 4: Attrition Model Comparison (v3.0)

**Objective:**  
Benchmark **Logistic Regression** against a **Random Forest classifier** to see if tree-based models improve prediction of employee attrition.

**Key Steps:**  
1. Prepared dataset with clean features (no leakage)  
2. Trained baseline Logistic Regression (linear, interpretable)  
3. Trained Random Forest (non-linear, ensemble)  
4. Compared performance using Accuracy & ROC AUC  
5. Visualized confusion matrices and feature importance  

**Results:**  
- Logistic Regression → Accuracy: **75%**, ROC AUC: **79%**  
- Random Forest → Accuracy: **83%**, ROC AUC: **77%**  
- Random Forest showed stronger performance on non-linear features, while Logistic remains more interpretable.  

**Sample Visuals:**  
Confusion Matrix Comparison:  
![Model Comparison](images/model_comparision.png)  

Top Features (Logistic vs Random Forest):  
- Logistic: OverTime, SalesRep role, MaritalStatus=Single  
- Random Forest: OverTime, MonthlyIncome, Age buckets  
📓 [View the Model Comparison Notebook](notebooks/Attrition_ModelComparision.ipynb)

---

**Model Artifacts:**  
***Check Below for all Artifacts*** 
---

**Insights:**  
- Logistic = simple, transparent model (good for executive storytelling)  
- Random Forest = higher accuracy, captures complex patterns (good for prediction)  
- Next: try **XGBoost** and add **SHAP interpretability** for business-ready insights.

---

## 📊 Project 5: Advanced Attrition Models (Tuned RF + XGBoost)

**Objective:**  
Take predictive modeling beyond Logistic Regression by tuning Random Forest and introducing **XGBoost**, a gradient boosting algorithm widely used for structured/tabular datasets.  

---

**Key Steps:**  
1. Prepared dataset with cleaned features (no leakage).  
2. Trained and tuned Random Forest (grid search for depth, estimators, features).  
3. Trained XGBoost model with optimized hyperparameters.  
4. Compared model performance against Logistic Regression baseline.  
5. Visualized feature importances, ROC curves, and confusion matrices.  
6. Exported key data artifacts (top features, comparison metrics).  

---

**Results (Test Set):**  
- Logistic Regression → Accuracy: **75.2%**, ROC AUC: **0.798**  
- Tuned Random Forest → Accuracy: **83.7%**, ROC AUC: **0.769**  
- XGBoost → Accuracy: **86.4%**, ROC AUC: **0.774**  

---

### 🔥 Sample Visuals  

**1. ROC Curve Comparison (Logistic vs RF vs XGBoost)**  
![ROC Curve Comparison](images/roc_curve_comparison.png)  
📈 *Shows how Logistic, RF, and XGBoost trade off sensitivity vs specificity — XGBoost edges ahead on balance.*  

**2. Model Comparison (Accuracy vs ROC AUC)**  
![Model Comparison](images/model_comparison_barplot.png)  
📊 *Side-by-side accuracy vs ROC AUC highlights overall performance differences across models.*  

**3. Top Features (XGBoost Heatmap)**  
![Top Features - XGBoost Heatmap](images/top_features_xgb_heatmap.png)  
🔥 *Top 15 features by importance — OverTime, JobRole, and MonthlyIncome dominate attrition risk signals.*  

**4. Confusion Matrix (XGBoost)**  
![Confusion Matrix - XGBoost](images/confusion_matrix_xgb.png)  
🧩 *Visual breakdown of predictions — where XGBoost gets it right (and where it misses).*  

---
### Notebook 📓 [Attrition_AdvancedModels.ipynb → Advanced Models (RF + XGBoost)](notebooks/Attrition_AdvancedModels.ipynb)  
---
### 📦 Data Artifacts

- [rf_top_features.csv](data/rf_top_features.csv)  
- [xgb_top_features.csv](data/xgb_top_features.csv)  
- [model_comparison_results.csv](data/model_comparison_results.csv)
---
**Insights:**  
- Logistic Regression: strong interpretability but weaker predictive power.  
- Random Forest: higher accuracy (84%), robust to non-linearities.  
- XGBoost: best performer overall (86% accuracy), consistently picks up complex patterns (OverTime, MonthlyIncome, JobRole).  
---
**Next Steps:**  
- Add **SHAP interpretability** for XGBoost.  
- Deploy best model in a **Streamlit dashboard**.  
- Integrate SQL + ML for end-to-end HR analytics workflows.
---
## 📊 Project 6: Model Explainability with SHAP

**Objective:**  
Use SHAP values to interpret XGBoost predictions, showing both global and local drivers of attrition.  

---

### 🔑 Key Steps  
1. **Load & preprocess data** → Dropped leakage columns, encoded categoricals, ensured numeric types.  
2. **Load trained XGBoost model** → From Project 5 (`xgboost_attrition_model.pkl`).  
3. **Subsample data (100 rows)** → Keeps SHAP efficient & avoids kernel crashes.  
4. **Compute SHAP values** → Using new `shap.Explainer` API.  
5. **Generate visuals**:  
   - Global feature importance & summary plots.  
   - Local explanations (waterfall + bar).  
   - Dependence plot for feature interactions.  
6. **Export artifacts** → Saved plots in `/images/` for portfolio clarity.  

---

### 📈 Results  
- **Global Drivers:**  
  - Age, Distance from Home, and Daily Rate (Compensation )are the most influential features.  
- **Local Explanations:**  
  - For an individual employee, SHAP shows exactly which features push their attrition risk up or down.  
- **Feature Interactions:**  
  - Dependence plots reveal how attrition risk rises at lower income bands and with frequent overtime.  
- **Takeaway:**  
  - Moves the model from “black box” → **explainable AI**, building trust with HR leaders.  

---

### 🔥 Sample Visuals  

**1. SHAP Summary Plot (Global Drivers)**  
![SHAP Summary Plot](images/shap_summary_plot.png)  
🌍 *Shows which features consistently drive attrition across the workforce.*  

**2. SHAP Feature Importance (Bar Plot)**  
![SHAP Feature Importance](images/shap_feature_importance.png)  
📊 *Highlights top features by mean absolute SHAP value — OverTime, JobRole, and MonthlyIncome dominate.*  

**3. SHAP Local Explanations (Individual Employee)**  
- Waterfall Plot: ![SHAP Waterfall](images/shap_waterfall_plot.png)  
- Bar Plot: ![SHAP Local Bar](images/shap_local_bar_plot.png)  
👤 *Explains why one specific employee is predicted as at-risk, showing top contributing features.*  

**4. SHAP Dependence Plot (MonthlyIncome)**  
![SHAP Dependence Plot](images/shap_dependence_plot.png)  
🔀 *Shows how attrition risk changes with MonthlyIncome while interacting with other features.*  

---

### 📓 Notebook  
- [Attrition_ModelExplainability.ipynb](Attrition_ModelExplainability.ipynb)  
---

### 📦 Artifacts  
- [xgboost_attrition_model.pkl](models/xgboost_attrition_model.pkl)  
- Visuals:  
  - [shap_summary_plot.png](images/shap_summary_plot.png)  
  - [shap_feature_importance.png](images/shap_feature_importance.png)  
  - [shap_waterfall_plot.png](images/shap_waterfall_plot.png)  
  - [shap_local_bar_plot.png](images/shap_local_bar_plot.png)  
  - [shap_dependence_plot.png](images/shap_dependence_plot.png)
---
### ✅ Conclusions  
- **Global Drivers:** Overtime, Job Role, and Monthly Income stand out as top predictors.  
- **Local Explanations:** Individual-level SHAP plots build confidence in predictions.  
- **Feature Interactions:** Dependence plots highlight nuanced patterns beyond simple correlations.  

**Why this matters:**  
- SHAP adds transparency → critical for HR applications where fairness & accountability are key.  
- Executives can see not just *who* is at risk, but *why*.  
- Enhances business trust in predictive HR analytics.  

---
## 📊 Project 7: Interactive Attrition Dashboard (Streamlit)

**Objective:**  
Deploy an interactive web app where HR leaders can upload data, run attrition predictions, and explore explainability visuals.

---

### 🔑 Key Steps  
1. Built a **Streamlit app (`app.py`)** to host trained models (Logistic, RF, XGBoost).  
2. Created `trained_columns.pkl` for consistent feature alignment during predictions.  
3. Added upload functionality → users can test with their own HR datasets (CSV).  
4. Integrated SHAP for **global & local explanations** inside the dashboard.  
5. Deployed app on **Streamlit Cloud** → shareable public link.  

---

### 📈 Results  
- End-to-end ML workflow is now **live & interactive**.  
- Users can:  
  - Upload data  
  - Select model  
  - View predictions + probabilities  
  - Explore explainability plots  
- Moves this portfolio from **static analysis → deployable HR Tech product**.  

---

### 🖼️ Sample Dashboard  
![Streamlit Dashboard](images/streamlit_dashboard.png)  

---

### 🌍 Live App  
- [Open Dashboard](https://hr-tech-portfolio.streamlit.app/)  

---

### 📂 Files  
- [app.py](app.py) → main dashboard app  
- [trained_columns.pkl](models/trained_columns.pkl) → ensures feature alignment  
- [requirements.txt](requirements.txt) → dependencies for deployment  

---

### ✅ Conclusions  
- Streamlit deployment shows the ability to go beyond notebooks and create **real-world HR tools**.  
- Adds credibility for consulting/analytics roles by showcasing full-stack delivery.  
- Sets foundation for future dashboards (Compensation, SQL+ML, etc.).  

---

### 🚀 Next Steps  
- Build **SQL + ML pipeline** to query databases + run predictions (Project 8).  
- Extend dashboard with **Compensation Analytics** modules.

## ⚒️ Tech Stack  

- Python (Pandas, Matplotlib, Seaborn, scikit-learn, Jupyter Notebook)  
- SQL (SQLite for queries on HR dataset)
- Dataset: [IBM HR Analytics Attrition Dataset (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Model artifacts:(All Models)
- [logistic_attrition_model.pkl](models/logistic_attrition_model.pkl)
- [scaler.pkl](models/scaler.pkl)
- [logistic_attrition_model_tuned.pkl](models/logistic_attrition_model_tuned.pkl)
- [random_forest_attrition_model.pkl](models/random_forest_attrition_model.pkl)
- [random_forest_tuned.pkl](models/random_forest_tuned.pkl)  
- [xgboost_attrition_model.pkl](models/xgboost_attrition_model.pkl)

---

## 🛠️ How to Run This Project

Follow these steps to reproduce the analysis on your own system:

1. **Clone the repository**
   ```bash
   git clone https://github.com/AMBOT-pixel96/hr-tech-portfolio.git
   cd hr-tech-portfolio
2. **Create a virtual environment Using Conda:**
```bash
conda create -n hrtech python=3.10 -y
conda activate hrtech
```
3. **Install required packages**
```
pip install -r requirements.txt
```
4. Launch Jupyter Notebook

jupyter notebook


5. Open and run the notebooks

### 📘 Baby Steps (Learning Path)
- [Day1-Hello-Amlan.ipynb](notebooks/Day1-Hello-Amlan.ipynb)  
- [Day2-Basics.ipynb](notebooks/Day2-Basics.ipynb)  
- [Day3-DataTypesAndControl.ipynb](notebooks/Day3-DataTypesAndControl.ipynb)  

### 📊 Main Projects
- [Day4-AttritionRiskAnalyzer.ipynb → Descriptive Analytics](notebooks/Day4-AttritionRiskAnalyzer.ipynb)  
- [Day4-AttritionRiskAnalyzer_v2.0.ipynb → Enhanced Descriptive Analytics](notebooks/Day4-AttritionRiskAnalyzer_v2.0.ipynb)  
- [Attrition_PredictiveModel_V2.ipynb → Predictive Modeling (Logistic Regression)](notebooks/Attrition_PredictiveModel_V2.ipynb)  
- [Attrition_PredictiveModel_V3.ipynb → Predictive Modeling (Tuned Logistic Regression)](notebooks/Attrition_PredictiveModel_V3.ipynb)  
- [Attrition_ModelComparision.ipynb → Logistic Regression vs Random Forest](notebooks/Attrition_ModelComparision.ipynb)  
- [Attrition_AdvancedModels.ipynb → Advanced Models (RF + XGBoost)](notebooks/Attrition_AdvancedModels.ipynb)

---

### 🧹 Side Quests
-[HR_Data_Cleaning_Utility_V1.ipynb → HR Data Cleanup Utility](sidequests/HR_Data_Cleaning_Utility_V1.ipynb)

---
## 🚀 Upcoming Projects  

- Compensation Analytics Dashboard
- SQL query library for HR datasets (attrition by job role, tenure, etc.)  
- Feature engineering + cross-validation for predictive modeling  
- Streamlit dashboard for interactive attrition prediction  

---

## 🧑‍💻 About Me  

I’m exploring the intersection of **Compensation & Benefits, HR Tech, and People Analytics**.  
This repo is my hands-on portfolio — tracking progress as I move from HR practitioner → HR Tech consultant.  

---

⭐️ If you find this interesting, follow my journey here or connect with me on LinkedIn.

---

<!-- trigger: repo-tree -->

---
