# ML Bubble 2026 - Unstop Final Submission Checklist

**Track**: TE-BE – Design & Solve (Advanced)  
**Event**: ML Bubble 2026 – Army Institute of Technology (AIT), Pune  
**Project Title**: Student Performance Prediction using Machine Learning  

---

## Submission Tasks Checklist

### 1. Codebase & Executable Verification
- [x] **`main.py` Execution Check**: Verified clean execution of `python main.py` with zero errors.
- [x] **Dataset Auto-Fetch / Fallback**: Tested local caching, remote dataset retrieval, and synthetic fallback generation.
- [x] **Preprocessing Pipeline**: Confirmed `ColumnTransformer`, `OneHotEncoder`, and `StandardScaler` operate cleanly.
- [x] **Multi-Model Benchmark**: Evaluated Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor.
- [x] **Metric Calculations**: Outputted MAE, RMSE, and $R^2$ scores for all candidate models.
- [x] **Model Serialization**: Exported serialized model artifact `models/best_student_model.joblib` via `joblib`.
- [x] **Dependencies File**: Verified `requirements.txt` contains required libraries (`pandas`, `numpy`, `scikit-learn`, `joblib`).

### 2. Documentation & Report Verification
- [x] **GitHub `README.md`**: Includes title, badge icons, executive summary, feature catalog, installation steps, and model results table.
- [x] **Academic Report (`academic_report.md`)**: Formal 2-page academic paper featuring Abstract, Problem Statement, Methodology, Mathematical Formulations, Experimental Results Table, Deployment Discussion, and References.
- [x] **Technical & Conceptual Sections (`project_overview_sections.md`)**: Detailed documentation covering all 14 required sections in full depth.
- [x] **Repository Blueprint (`repository_structure.txt`)**: Clean tree representation of all project files and directories.

### 3. Presentation & Defense Preparation
- [x] **PowerPoint Slides Content (`ppt_slides_content.md`)**: Complete 6-slide deck structure including titles, bullet points, and visual diagram layout descriptions.
- [x] **Presentation Script (`presentation_script.md`)**: Timed 2-minute (120-second) pitch script for the online model explanation round.

---

## Final Unstop Upload Checklist
1. **GitHub Repository Link**: Ensure your repository is public and contains `main.py`, `requirements.txt`, `README.md`, `dataset/`, `models/`, and `docs/`.
2. **Project Report File**: Export `docs/academic_report.md` as PDF or upload the Markdown document as required by the Unstop portal.
3. **Presentation Deck**: Convert `docs/ppt_slides_content.md` into your PowerPoint `.pptx` or `.pdf` slides for submission.
4. **Project Title & Abstract**: Copy title ("Student Performance Prediction using Machine Learning") and Abstract from `docs/academic_report.md` into the Unstop submission form text fields.
