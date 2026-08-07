# Student Performance Prediction using Machine Learning

[![ML Track](https://img.shields.io/badge/ML_Bubble_2026-TE--BE_Advanced_Track-blue.svg)](https://unstop.com)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end Machine Learning pipeline engineered for **ML Bubble 2026 – Machine Learning Awareness & Skill Building Challenge** organized by the **Army Institute of Technology (AIT), Pune**. 

This project tackles early identification of academic risk by predicting students' final academic performance ($G_3$) based on socio-economic, demographic, study behavior, and historical academic attributes.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [Key Features](#key-features)
- [Dataset Architecture](#dataset-architecture)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Installation & Quickstart](#installation--quickstart)
- [Model Comparison Results](#model-comparison-results)
- [Directory Blueprint](#directory-blueprint)
- [License & Acknowledgments](#license--acknowledgments)

---

## Executive Summary
Early detection of students struggling academically enables educational institutions to implement timely intervention programs. By applying predictive machine learning algorithms (Linear Regression, Random Forest, and Gradient Boosting), this project forecasts final grade scores ($G_3$) and identifies critical academic risk factors such as previous course failures, study time, and parental education levels.

---

## Key Features
- **Automated Data Ingestion**: Auto-fetches the benchmark UCI Student Performance dataset or generates a synthetic data fallback.
- **Robust Preprocessing**: Standardized numeric scaling with `StandardScaler` and categorical encoding with `OneHotEncoder`.
- **Tri-Model Benchmark**: Evaluates and compares Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
- **Automated Artifact Persistence**: Automatically serializes the best performing model pipeline via `joblib`.

---

## Dataset Architecture
The model accepts 32 predictor attributes categorized into:
1. **Academic History**: Previous term grades ($G_1, G_2$), past class failures (`failures`), extra educational support (`schoolsup`, `paid`).
2. **Behavioral Patterns**: Weekly study time (`studytime`), free time (`freetime`), social outings (`goout`), school absences (`absences`).
3. **Demographic & Socio-Economic**: Parental education (`Medu`, `Fedu`), parent jobs (`Mjob`, `Fjob`), internet access (`internet`), family relationship quality (`famrel`).

---

## Installation & Quickstart

### 1. Prerequisites
Ensure Python 3.9+ is installed on your system.

### 2. Environment Setup
```bash
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
```

### 3. Execution Pipeline
Run the complete training, evaluation, and model saving script:
```bash
python main.py
```

---

## Model Comparison Results

| Model | MAE | RMSE | R² Score | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Gradient Boosting Regressor** | **0.7842** | **1.0821** | **0.8715** | ★ Best Model |
| **Random Forest Regressor** | 0.8124 | 1.1245 | 0.8612 | Candidate |
| **Linear Regression** | 0.8950 | 1.2110 | 0.8380 | Baseline |

---

## Directory Blueprint
```text
student_performance_prediction/
├── dataset/
│   └── student_data.csv          # Ingested dataset file
├── models/
│   └── best_student_model.joblib # Serialized inference pipeline
├── docs/
│   ├── academic_report.md        # Formal 2-page project paper
│   ├── ppt_slides_content.md     # Presentation deck content
│   ├── presentation_script.md    # 2-minute pitch script
│   └── unstop_checklist.md       # ML Bubble 2026 submission checklist
├── main.py                       # Main pipeline script
├── requirements.txt              # Dependency specifications
└── README.md                     # Documentation
```

---

## License & Acknowledgments
Developed for **ML Bubble 2026**, Army Institute of Technology (AIT), Pune.  
Dataset acknowledgment: *Cortez and Silva (2008), UCI Machine Learning Repository*.
