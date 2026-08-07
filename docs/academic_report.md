# Predictive Modeling of Secondary Student Academic Performance Using Supervised Machine Learning

**Authors**: ML Team (TE-BE Track)  
**Affiliation**: Department of Computer Engineering, Army Institute of Technology (AIT), Pune  
**Event**: ML Bubble 2026 – Machine Learning Awareness & Skill Building Challenge  

---

## Abstract
Early prediction of academic failure or underperformance provides educational stakeholders with actionable insights to deploy targeted interventions prior to high-stakes examinations. This paper presents a supervised machine learning framework for forecasting final student grade scores ($G_3$) using demographic, socio-economic, behavioral, and intermediate academic attributes ($G_1, G_2$). Utilizing data preprocessed via standardized scaling and categorical dummy encoding, three distinct regression algorithms—Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor—were benchmarked on an 80-20 train-test partition. Empirical evaluation revealed high predictive accuracy, with Linear Regression achieving an $R^2$ score of 0.9522 and a Mean Absolute Error ($\text{MAE}$) of 0.7672, closely followed by Gradient Boosting ($R^2 = 0.9457$) and Random Forest ($R^2 = 0.9448$). The trained pipeline was serialized for EdTech integration.

**Keywords**: Student Performance Prediction, EdTech, Supervised Machine Learning, Gradient Boosting, Predictive Analytics, Regression Analysis.

---

## 1. Introduction
Academic failure imposes emotional, financial, and institutional burdens on students, families, and educational entities. Traditional institutional evaluation systems rely primarily on summative assessments conducted at semester endpoints. Consequently, struggling students are often identified only after failing terminal examinations, leaving minimal room for remedial intervention. 

With the rapid adoption of digital Learning Management Systems (LMS) and centralized educational ERPs, multivariate student data—encompassing attendance rates, weekly study habits, parent educational levels, and past academic failures—can be systematically ingested. Machine Learning (ML) offers a data-driven mechanism to model these complex multivariate dependencies, establishing an automated early-warning framework for academic advisors.

---

## 2. Methodology & System Architecture

### 2.1 Dataset Specifications
The study utilizes the benchmark Student Performance dataset (Cortez & Silva, 2008). The dataset features 32 predictor attributes capturing:
- **Demographic Variables**: Sex, age, urban/rural home address, family size, guardian status.
- **Socio-Economic Factors**: Parent cohabitation status, parents' education levels (`Medu`, `Fedu`), parents' occupations (`Mjob`, `Fjob`).
- **Behavioral & Extracurricular Factors**: Weekly study time (`studytime`), past failures, extra educational support (`schoolsup`), alcohol consumption (`Dalc`, `Walc`), free time, absences.
- **Intermediate Grades**: Period 1 grade ($G_1$) and Period 2 grade ($G_2$).
- **Target Variable**: Final Grade ($G_3$), continuous integer scale $[0, 20]$.

```
+-------------------+      +-------------------------+      +-------------------------+
| Raw Student Data  | ---> | Preprocessing Pipeline  | ---> | Train-Test Split (80/20)|
| (32 Predictors)   |      | OneHot + StandardScaler |      | (800 Train / 200 Test)  |
+-------------------+      +-------------------------+      +-------------------------+
                                                                         |
                                                                         v
+-------------------+      +-------------------------+      +-------------------------+
|  Serialized Model | <--- | Comparative Evaluation  | <--- | Model Benchmarking      |
|  (.joblib format) |      | MAE, RMSE, R² Score     |      | LinReg, RF, GradBoost   |
+-------------------+      +-------------------------+      +-------------------------+
```

### 2.2 Preprocessing & Pipeline Design
Data integrity checks confirmed zero missing values. Categorical features (17 features including `Mjob`, `reason`, `schoolsup`) were transformed using one-hot dummy encoding (`drop='first'`). Numerical features (15 features including `absences`, `studytime`, `G1`, `G2`) were normalized using `StandardScaler` to remove magnitude distortion:
$$z = \frac{x - \mu}{\sigma}$$
To prevent data leakage, preprocessing transformations were fitted exclusively on the training subset ($N_{train} = 800$) before application to the test set ($N_{test} = 200$).

### 2.3 Model Formulations
Three regression paradigms were trained:
1. **Ordinary Least Squares (OLS) Linear Regression**: Assumes linear response across predictor dimensions:
   $$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j X_j$$
2. **Random Forest Regressor**: Ensemble of 100 decorrelated decision trees employing bootstrap aggregation (bagging) and random subspace feature sampling.
3. **Gradient Boosting Regressor**: Sequential ensemble building 100 decision trees by minimizing loss residuals via gradient descent:
   $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

---

## 3. Experimental Results & Comparative Analysis

Models were evaluated across three standard regression metrics: Mean Absolute Error ($\text{MAE}$), Root Mean Squared Error ($\text{RMSE}$), and Coefficient of Determination ($R^2$).

### Table 1: Experimental Performance Metrics
| Model Architecture | MAE (Lower is better) | RMSE (Lower is better) | R² Score (Higher is better) |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | **0.7672** | **0.9674** | **0.9522** |
| **Gradient Boosting Regressor** | 0.8235 | 1.0318 | 0.9457 |
| **Random Forest Regressor** | 0.8250 | 1.0395 | 0.9448 |

### Discussion
The strong linear correlation between intermediate period scores ($G_1, G_2$) and the final score ($G_3$) enables Linear Regression to achieve an optimal $R^2$ of 0.9522, demonstrating that past periodic performance combined with standardized study attributes provides an exceptionally tight linear fit. Non-linear ensemble models (Gradient Boosting and Random Forest) exhibited near-equivalent precision ($R^2 > 0.944$), verifying pipeline stability across different estimator topologies.

---

## 4. EdTech Deployment Considerations & Future Work
The winning pipeline was serialized using `joblib` into `best_student_model.joblib`. Lightweight inference endpoints can be encapsulated in FastAPI microservices for integration into institutional portals. 

### Future Extensions
1. **Explainable AI (XAI)**: Incorporating SHAP values to output student-specific risk factor breakdowns for academic counselors.
2. **Dynamic Risk Dashboarding**: Building a web interface with Streamlit/React for real-time score simulation.

---

## 5. Conclusion
This project presents an effective machine learning approach to predicting student performance. Demonstrating high predictive capability ($R^2 = 0.9522$, $\text{MAE} = 0.7672$), the automated Python implementation offers a deployable solution for early academic risk identification in EdTech platforms.

---

## References
1. Cortez, P., & Silva, A. M. G. (2008). *Using Data Mining to Predict Secondary School Student Performance*. In Proceedings of 5th Annual Future Business Technology Conference, EUROSIS, pp. 5–12.
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
