# 📊 Compensation & Benefits Dashboard  
_Executive HR Analytics Application — Streamlit + Python_  

An end-to-end HR Compensation Analytics App built for boardroom-ready insights — from pay equity and quartile trends to market benchmarking and automated executive reports.

---

## 🧭 Overview
**Developer:** Amlan Mishra  
**Version:** v5.0 Stable (Oct 2025)  
**Stack:** Streamlit • Plotly • Pandas • NumPy • ReportLab • WeasyPrint  
**Theme:** Matte Dark Mode (Professional UI)  
**Location:** India 🇮🇳  

---

## ⚙️ Features
| Category | Capability |
|-----------|-------------|
| 🧩 Data Upload | Import internal & benchmark datasets (CSV/XLSX) |
| 🧮 Hierarchy Setup | Drag-drop or dropdown job level sequencing |
| 💾 Session Memory | Smart save, auto-save & auto-restore modes |
| 📈 Metrics | 7+ key HR metrics with charts & insights |
| 🧠 Chatbot | Query data conversationally (beta) |
| 📰 Reports | One-click PDF generation (executive layout) |
| 🖤 Design | Full dark mode with animated headers & badges |

---

## 🗂️ Repository Structure

```text
cb_dashboard_artifacts/ 
│ 
├── data/ 
│   ├── employee_compensation_data.csv 
│   └── benchmarking_data.csv 
│ ├── exports/ 
│   ├── CB_User_Guide.pdf 
│   └── CB_Report_consolidated.pdf 
│ ├── images/ 
│   ├── app_layout/ 
│   ├── metrics/ 
│   └── chatbot/ 
│ └── README.md   ← (this file)
```
---

## 🖥️ Application Layout

### 🔹 Step Flow
| Step | Description | Screenshot |
|------|--------------|-------------|
| 1️⃣ | App Landing Page | ![Landing](./images/app_layout/1-App-Landing-Page.jpg) |
| 2️⃣ | Download Templates | ![Templates](./images/app_layout/2-Step-1-Download-Templates-Guides.jpg) |
| 3️⃣ | Upload Data | ![Upload](./images/app_layout/4-Upload-Data.jpg) |
| 4️⃣ | Define Job Hierarchy | ![Hierarchy](./images/app_layout/5-Set-Job-Order.jpg) |
| 5️⃣ | Apply Custom Order | ![Apply](./images/app_layout/6-Appy-Order.jpg) |
| 6️⃣ | Session Persistence | ![Persistence](./images/app_layout/7-Session-Persistence.jpg) |
| 7️⃣ | Export Reports | ![PDFs](./images/app_layout/8-PDF-Downloads-Section.jpg) |
| 8️⃣ | Export Charts (PNGs) | ![PDFs](./images/app_layout/9-Image-Downloads-Section.jpg) |
---

## 📊 Key Metrics Showcase

| Metric | Visualization | Table |
|---------|----------------|--------|
| Average CTC | ![Avg CTC](./images/metrics/Metric-1-Graph.jpg) | ![Table](./images/metrics/Metric-1-Table-B.jpg) |
| Median CTC | ![Median](./images/metrics/Metric-2-Graph.jpg) | ![Table](./images/metrics/Metric-2-Table.jpg) |
| Bonus % of CTC | ![Bonus](./images/metrics/Metric-3-Graph.jpg) | ![Table](./images/metrics/Metric-3-Table.jpg) |
| Quartile Distribution | ![Quartile](./images/metrics/Metric-4-Graph.jpg) | ![Table](./images/metrics/Metric-4-Table.jpg) |
| Gender Pay Gap | ![Gender](./images/metrics/Metric-5-Graph.jpg) | ![Table](./images/metrics/Metric-5-Table.jpg) |
| Performance & Pay | ![Perf](./images/metrics/Metric-6-Graph.jpg) | ![Table](./images/metrics/Metric-6-Table.jpg) |
| Market Comparison | ![Market](./images/metrics/Metric-7-Graph.jpg) | ![Table](./images/metrics/Metric-7-Table.jpg) |

---

## 💬 Chatbot Preview
| Screenshot | Description |
|-------------|--------------|
| ![Chatbot1](./images/chatbot/Chatbot-1.jpg) | Query data by natural questions |
| ![Chatbot2](./images/chatbot/Chatbot-3.jpg) | Compare roles or levels interactively |
| ![Chatbot3](./images/chatbot/Chatbot-6.jpg) | Visual + text summary responses |

---

## 📘 Executive Exports

| File | Description |
|------|--------------|
| [📗 CB_User_Guide.pdf](./exports/CB_User_Guide.pdf) | User Guide — Layout, Metrics, and Usage Rules |
| [📙 CB_Report_Consolidated.pdf](./exports/Cb_Report_consolidated.pdf) | Automatically generated executive summary report |

---

## 🚀 How to Use the Dashboard

Access the live hosted version directly on Streamlit Cloud:  
🔗 **[Launch Dashboard](https://cb-dashboard.streamlit.app)**

Once opened:
1. Download the provided **Internal** and **Benchmark** templates.
2. Upload your HR data files (CSV/XLSX format) using the **Upload Data** step.
3. Define your **Job Level Hierarchy** via dropdown interface.
4. Explore key metrics:
   - Average / Median CTC  
   - Bonus % of CTC  
   - Quartile Distribution  
   - Gender Pay Gap  
   - Market Benchmarking  
   - Performance vs Pay  
5. Export your results to professional **PDF reports** or query the **Chatbot** for insights.

The app automatically saves your session data (ordering, chat history, configurations) so you can resume where you left off.

---

## 💻 How to Clone & Run Locally

Follow the steps below to set up and run the dashboard locally on your system.

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AMBOT-pixel96/hr-tech-portfolio.git
cd hr-tech-portfolio
```
### 2️⃣ Install Requirements

Make sure you have Python 3.10+ installed.
Then run:
```bash
pip install -r requirements.txt
```
3️⃣ Launch the App

Run Streamlit:
```bash
streamlit run cb_dashboard.py
```
The dashboard will open automatically at:
🌐 http://localhost:8501


---

## 👤 Author

### Amlan Mishra
### Assistant Manager – Compensation & Benefits (Tech HR), KPMG India
### 🔗[LinkedIn](https://www.linkedin.com/in/amlan-mishra-7aa70894)
### 💻 [GitHub Portfolio](https://github.com/AMBOT-pixel96/hr-tech-portfolio)

*Developer, HR Tech Strategist, and People Analytics Enthusiast.*
*Building intelligent HR systems that blend analytics, automation, and design.*


---

## 📂 Back to Main Portfolio

Return to the main portfolio for additional HR Tech & People Analytics projects:
[HR Tech Portfolio — Main Repository](../README.md)


---