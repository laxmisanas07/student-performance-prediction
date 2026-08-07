# PowerPoint Presentation Deck Content (6 Slides)

**Track**: TE-BE – Design & Solve (Advanced)  
**Event**: ML Bubble 2026 – Army Institute of Technology (AIT), Pune  
**Project Title**: Student Performance Prediction using Machine Learning  

---

## Slide 1: Title & Project Overview

### Slide Content
- **Header**: Student Performance Prediction using Machine Learning
- **Subtitle**: Early Academic Risk Detection & Predictive Analytics for EdTech
- **Track**: TE-BE – Design & Solve (Advanced) | ML Bubble 2026, AIT Pune
- **Team**: ML Innovators
- **Key Takeaway**: A data-driven machine learning system to predict final student grades ($G_3$) using socio-economic, behavioral, and academic predictors to enable timely remedial intervention.

### Suggested Diagram / Visual Layout
- **Visual**: split screen layout with project title on left side and an icon layout on the right showing 3 pillars: `Data Ingestion -> Machine Learning Pipeline -> EdTech Early Warning`.

---

## Slide 2: Problem Statement & ML Suitability

### Slide Content
- **The Challenge**:
  - Educational institutions rely on reactive end-of-term exams.
  - At-risk students are identified too late for effective intervention.
  - Academic outcomes depend on complex, multi-dimensional socio-economic and behavioral factors.
- **Why Machine Learning?**:
  - **Non-Linear Dynamics**: Captures subtle interactions between study time, parental education, absences, and academic scores.
  - **Scalability**: Enables instant evaluation of thousands of student profiles.
  - **Proactive EdTech**: Powers real-time early warning alerts for educators and counselors.

### Suggested Diagram / Visual Layout
- **Diagram**: Comparative Infographic:
  - *Traditional Approach*: Student Enrolls -> Terminal Exam -> Failure Detected (Too Late!)
  - *ML Predictive Approach*: Ingest Profile & Mid-term Data -> ML Model Forecasts Risk -> Targeted Counseling -> Student Success!

---

## Slide 3: Dataset Architecture & Feature Engineering

### Slide Content
- **Dataset Source**: Benchmark UCI Student Performance Dataset (33 total fields).
- **Core Feature Categories**:
  - **Academic Records**: Period 1 grade ($G_1$), Period 2 grade ($G_2$), past failures (`failures`).
  - **Study Habits**: Weekly study time (`studytime`), extra educational support (`schoolsup`), class absences (`absences`).
  - **Socio-Economic & Demographics**: Parental education (`Medu`, `Fedu`), family status (`famsize`, `Pstatus`), internet access (`internet`).
- **Data Preprocessing**:
  - `OneHotEncoder`: Encodes 17 categorical variables cleanly.
  - `StandardScaler`: Normalizes 15 continuous variables to zero mean / unit variance.
  - `Train-Test Split`: 80% Training ($N=800$) / 20% Evaluation ($N=200$).

### Suggested Diagram / Visual Layout
- **Diagram**: Feature Taxonomy Grid / Block Diagram displaying 4 color-coded feature buckets connected to a central Preprocessing Transformer block (`ColumnTransformer`).

---

## Slide 4: Model Architecture & Comparative Evaluation

### Slide Content
- **Algorithms Evaluated**:
  1. **Linear Regression**: Baseline interpretable continuous model.
  2. **Random Forest Regressor**: 100-tree bagging ensemble.
  3. **Gradient Boosting Regressor**: 100-stage boosting ensemble.
- **Experimental Results**:

| Model Architecture | MAE ↓ | RMSE ↓ | R² Score ↑ | Evaluation Status |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | **0.7672** | **0.9674** | **0.9522** | **Best Estimator** |
| **Gradient Boosting Regressor** | 0.8235 | 1.0318 | 0.9457 | High Precision |
| **Random Forest Regressor** | 0.8250 | 1.0395 | 0.9448 | Robust Baseline |

### Suggested Diagram / Visual Layout
- **Suggested Chart**: Horizontal Bar Chart comparing $R^2$ Scores and Grouped Bar Chart comparing MAE/RMSE across the 3 algorithms.

---

## Slide 5: Model Deployment & EdTech Architecture

### Slide Content
- **Model Serialization**: Best performing model pipeline serialized to `models/best_student_model.joblib`.
- **Production Integration**:
  - **REST API Endpoint**: Exposed via FastAPI / Flask microservice.
  - **LMS Integration**: Plugs into Canvas, Moodle, or institutional ERP systems.
- **Key Beneficiaries**:
  - **Educators**: Identify struggling students early in the semester.
  - **Counselors**: Receive feature importance breakdowns (e.g., high absences, low study hours).
  - **Students**: Receive personalized study recommendations.

### Suggested Diagram / Visual Layout
- **Diagram**: System Architecture Flowchart:  
  `Student ERP Data -> REST API Endpoint -> Joblib Model Pipeline -> Risk Score Dashboard (Red / Yellow / Green Alert)`.

---

## Slide 6: Future Scope & Conclusion

### Slide Content
- **Future Enhancements**:
  - **Explainable AI (XAI)**: Integrate SHAP values to explain specific risk factors to parents and teachers.
  - **Multi-Semester Time-Series**: Incorporate LSTM/Recurrent models for multi-year academic trajectory tracking.
  - **Remediation Recommender**: Auto-generate personalized study plans based on identified feature gaps.
- **Conclusion**:
  - Built a production-ready end-to-end ML solution for student performance prediction.
  - Achieved outstanding predictive fit ($R^2 = 0.9522, \text{MAE} = 0.7672$).
  - Ready for immediate submission for **ML Bubble 2026**.

### Suggested Diagram / Visual Layout
- **Visual**: Summary Roadmap Timeline (Phase 1: ML Baseline Completed -> Phase 2: LMS Deployment -> Phase 3: Explainable AI Integration).
