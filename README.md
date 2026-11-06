# Student Performance Prediction Engine

[![ML Track](https://img.shields.io/badge/ML_Bubble_2026-TE--BE_Advanced_Track-0284C7.svg)](https://unstop.com)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-34D399.svg)](LICENSE)

An end-to-end Machine Learning pipeline and interactive EdTech analytics dashboard engineered for **ML Bubble 2026 – Machine Learning Awareness & Skill Building Challenge** organized by the **Army Institute of Technology (AIT), Pune**.

- 💻 **GitHub Repository**: [https://github.com/laxmisanas07/student-performance-prediction](https://github.com/laxmisanas07/student-performance-prediction)
- 🌐 **Live Web Application**: [https://student-performance-prediction-klyac6rznvfsk5j6k6j6gt.streamlit.app](https://student-performance-prediction-klyac6rznvfsk5j6k6j6gt.streamlit.app)
- 👩‍💻 **Developer**: [Laxmi Sanas](https://github.com/laxmisanas07) (B.Tech AI & Data Science, SITRC Nashik)

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Key Engineering Features](#key-engineering-features)
- [Dataset Architecture & Provenance](#dataset-architecture--provenance)
- [Machine Learning Architecture](#machine-learning-architecture)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Streamlit Dashboard Architecture](#streamlit-dashboard-architecture)
- [Installation & Local Execution](#installation--local-execution)
- [Directory Blueprint](#directory-blueprint)
- [Deployment on Streamlit Community Cloud](#deployment-on-streamlit-community-cloud)
- [Responsible AI & Academic Disclaimer](#responsible-ai--academic-disclaimer)
- [Acknowledgments & Citation](#acknowledgments--citation)

---

## Executive Summary

Early identification of students vulnerable to academic underperformance allows educators and academic counselors to deliver targeted interventions before cumulative exam periods. This project designs, evaluates, and deploys a reproducible regression pipeline forecasting secondary school students' final mathematics grade ($G_3$, on a 0–20 point scale). 

The pipeline benchmarks multiple regression families (**Linear Regression**, **Random Forest Regressor**, and **Gradient Boosting Regressor**) using standardized feature transformers, evaluates them on an independent holdout test split ($20\%$), persists the optimal serialized pipeline, and surfaces predictions through a production-grade Streamlit web application.

---

## Key Engineering Features

- **Reproducible Machine Learning Pipeline**: Integrated `sklearn.pipeline.Pipeline` with `ColumnTransformer` packaging numerical standard scaling (`StandardScaler`) and categorical one-hot encoding (`OneHotEncoder(handle_unknown='ignore')`) into a unified artifact.
- **Authentic UCI Benchmark Data**: Ingests and caches the authentic UCI Machine Learning Student Performance dataset ($395$ samples for Portuguese secondary mathematics) with automated fallback verification.
- **Empirical Model Selection**: Compares three candidate regression architectures using Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Coefficient of Determination (**R²**), selecting the best model based on holdout validation.
- **Dynamic Metrics Persistence**: Exports verified evaluation metrics and candidate benchmarks to `models/model_metrics.json`, ensuring the frontend dashboard displays real empirical results with zero hardcoded statistics.
- **Interactive EdTech Dashboard**: Features 6 comprehensive modules, interactive Plotly comparison charts, Portuguese grading classification rubrics, and downloadable student assessment reports.
- **Deployment Ready**: Fully configured with `.streamlit/config.toml`, strict dependencies in `requirements.txt`, and zero local path dependencies.

---

## Dataset Architecture & Provenance

The system models the **UCI Student Performance Dataset** (Mathematics course, `student-mat.csv`), collected from two secondary schools in Alentejo, Portugal (*Gabriel Pereira* and *Mousinho da Silveira*):

- **Instances**: $395$ students
- **Input Predictors**: $32$ sociodemographic, academic, and behavioral features
- **Target Variable**: $G_3$ (Final Mathematics Grade, continuous score from $0$ to $20$)

### Feature Taxonomy

| Category | Input Features | Engineering Description |
| :--- | :--- | :--- |
| **Academic History** | `G1`, `G2`, `failures`, `schoolsup`, `paid` | Previous period marks ($0$–$20$), past class failures ($0$–$3$), institutional extra support |
| **Study Habits** | `studytime`, `absences`, `higher`, `internet` | Weekly study hours ($<2$, $2$–$5$, $5$–$10$, $>10$), school absences ($0$–$93$), aspirations |
| **Demographics** | `school`, `sex`, `age`, `address`, `famsize`, `Pstatus` | School code, sex, student age ($15$–$22$), urban/rural, family size, cohabitation |
| **Family Background** | `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `guardian` | Parental education ($0$–$4$), parental job categories, school selection reason, guardian |
| **Social & Health** | `famrel`, `freetime`, `goout`, `Dalc`, `Walc`, `health`, `romantic` | Family relationship ($1$–$5$), leisure time, peer outings, alcohol usage, health status ($1$–$5$) |

---

## Machine Learning Architecture

```
[Input Vector (32 Features)]
            │
            ├── Numerical Features (15)  ───> StandardScaler()
            │
            └── Categorical Features (17) ───> OneHotEncoder(handle_unknown='ignore')
                                                      │
                                                      ▼
                                            [ColumnTransformer]
                                                      │
                                                      ▼
                                       [Regressor Architecture]
                                      (Random Forest / Gradient Boosting)
                                                      │
                                                      ▼
                                          [Predicted Grade G3 (0–20)]
```

### Preprocessing Protocol
- **Numerical Scaling**: $15$ numeric features scaled to zero mean and unit variance.
- **Categorical Encoding**: $17$ nominal features one-hot encoded with `handle_unknown='ignore'` to guarantee zero runtime failures when encountering unseen categories in production.
- **Data Partitioning**: $80\%$ Training ($316$ samples), $20\%$ Holdout Testing ($79$ samples), using `random_state=42` for exact reproducibility.

---

## Empirical Benchmark Results

All models were evaluated on the identical $20\%$ unseen holdout test split. The model with the lowest Root Mean Squared Error (**RMSE**) and highest **R²** was automatically selected for production inference:

| Model Architecture | MAE (Marks) | RMSE (Marks) | R² Score | Deployment Status | Selection Rationale |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Random Forest Regressor** | **1.1858** | **1.9706** | **0.8106** | **★ Selected Best Model** | Lowest prediction variance & highest fit on unseen test set |
| **Gradient Boosting Regressor** | 1.1526 | 1.9829 | 0.8083 | Candidate Benchmark | Competitive error fit; slightly higher residual penalty |
| **Linear Regression** | 1.6467 | 2.3784 | 0.7241 | Baseline Reference | Susceptible to non-linear interaction terms |

> *Note: Evaluation metrics are dynamically loaded from `models/model_metrics.json`. R² is explicitly reported as the Coefficient of Determination ($R^2$), not classification accuracy.*

---

## Streamlit Dashboard Architecture

The frontend application (`app.py`) is styled in a modern dark navy palette (`#0B1120`, `#0F172A`, `#1E293B`, `#38BDF8`) and structured into four dedicated functional modules:

1. **🎯 Predictive Assessment**:
   - Organized student profile inputs (Academic History, Study Habits, Demographics, Family & Social Factors).
   - Real-time model inference computing estimated $G_3$ score.
   - Portuguese secondary school grading rubric classification:
     - **16–20**: *Very Good / Excellent (Distinction)*
     - **14–15**: *Good (Merit)*
     - **12–13**: *Satisfactory (Average)*
     - **10–11**: *Sufficient (Passing Grade)*
     - **0–9**: *Insufficient (Academic Support Recommended)*
   - One-click exportable academic assessment summary report (`.txt`).
2. **📊 Model Analytics & Benchmarks**:
   - Dynamic empirical metrics table highlighting best model performance.
   - Interactive Plotly visualizations comparing $R^2$, RMSE, and MAE across candidate architectures.
3. **💡 Educational Insights**:
   - Analysis of temporal assessment continuity ($G_1$ and $G_2$ as leading indicators).
   - Non-linear absence thresholds and study hour elasticity.
   - Methodological reminder distinguishing correlation from causality.
4. **ℹ️ About & Provenance**:
   - Institutional context (ML Bubble 2026, Army Institute of Technology, Pune).
   - Full citation of the UCI Student Performance benchmark dataset.

---

## Installation & Local Execution

### 1. Clone the Repository
```bash
git clone https://github.com/laxmisanas07/student-performance-prediction.git
cd student-performance-prediction
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train & Benchmark Models
To run the complete data ingestion, model comparison, and pipeline serialization:
```bash
python main.py
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
The web application will launch at `http://localhost:8501`.

---

## Directory Blueprint

```text
student-performance-prediction/
├── .devcontainer/
│   └── devcontainer.json            # VS Code remote container configuration
├── .streamlit/
│   └── config.toml                  # Streamlit dark navy theme & server settings
├── dataset/
│   └── student_data.csv             # Ingested UCI Student Performance dataset (395 rows)
├── docs/
│   ├── academic_report.md           # Formal project report
│   ├── ppt_slides_content.md        # Presentation slide deck content
│   ├── presentation_script.md       # Verbal presentation script
│   └── unstop_checklist.md          # Submission compliance checklist
├── models/
│   ├── best_student_model.joblib    # Serialized scikit-learn inference pipeline
│   └── model_metrics.json           # Persisted empirical evaluation benchmark results
├── app.py                           # Production Streamlit EdTech dashboard
├── main.py                          # Machine learning training & benchmarking pipeline
├── requirements.txt                 # Pinned dependencies for production deployment
└── README.md                        # Project documentation
```

---

## Deployment on Streamlit Community Cloud

The repository is configured for zero-configuration continuous deployment on **Streamlit Community Cloud**:

- **Repository**: `laxmisanas07/student-performance-prediction`
- **Branch**: `main`
- **Main file path**: `app.py`
- **Dependencies**: Auto-installed from `requirements.txt`
- **Custom Theme**: Loaded from `.streamlit/config.toml`

Whenever changes are pushed to `main`, Streamlit Community Cloud automatically triggers a rebuild and deploys the updated application.

---

## Responsible AI & Academic Disclaimer

Predicted final grades generated by this machine learning model are regression estimates based on historical student distributions from Portuguese secondary schools. These estimates are designed to serve as **early-warning diagnostic indicators** to help educators provide timely academic counseling. They must **never** be used as the sole basis for academic evaluations, grading determinations, disciplinary actions, or high-stakes administrative decisions.

---

## Acknowledgments & Citation

- **Event Host**: Organized under the **ML Bubble 2026** challenge by the **Army Institute of Technology (AIT), Pune**.
- **Dataset Citation**: 
  > P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance*. In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, April, 2008. EUROSIS, ISBN 978-9077381-39-7. [UCI ML Repository](https://archive.ics.uci.edu/dataset/320/student+performance).
- **License**: Released under the [MIT License](LICENSE).
