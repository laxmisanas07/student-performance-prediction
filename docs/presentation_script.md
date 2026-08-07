# 2-Minute Presentation Pitch Script

**Competition**: ML Bubble 2026 – Machine Learning Awareness & Skill Building Challenge  
**Organizer**: Army Institute of Technology (AIT), Pune  
**Track**: TE-BE – Design & Solve (Advanced)  
**Speaker**: Team Representative  
**Target Duration**: 120 Seconds (2 Minutes)  

---

## Pitch Script

### [0:00 - 0:25] Introduction & Problem Statement
> "Respected Judges and fellow participants, good day! We are representing team ML Innovators in the TE-BE Advanced Track of ML Bubble 2026. 
> 
> In traditional education, student underperformance is diagnosed reactively—after terminal exams are graded, when it is already too late for intervention. Our project, **Student Performance Prediction using Machine Learning**, solves this critical EdTech challenge by converting reactive evaluation into a proactive early-warning predictive system."

### [0:25 - 0:55] Dataset, Features & Preprocessing Pipeline
> "We utilized the benchmark UCI Student Performance dataset featuring 32 predictor attributes spanning demographic factors, parental education, study time, alcohol consumption, absences, and intermediate period grades $G_1$ and $G_2$. 
> 
> To process this mixed dataset, we engineered an end-to-end Python pipeline using `scikit-learn`. Categorical variables were encoded using `OneHotEncoder` with dummy variable drop, while continuous variables were normalized using `StandardScaler`. We split the data into an 80% training set and a 20% holdout test set with strict separation to eliminate data leakage."

### [0:55 - 1:30] Model Benchmarking & Results
> "We benchmarked three regression algorithms: Linear Regression, Random Forest, and Gradient Boosting Regressor. 
> 
> Evaluating performance across Mean Absolute Error, Root Mean Squared Error, and $R^2$ score, our models achieved exceptional predictive precision. Linear Regression delivered the top performance with an $R^2$ score of **0.9522** and a Mean Absolute Error of **0.7672** grade points on a 20-point scale. Both ensemble models closely followed with $R^2$ scores exceeding **0.944**, confirming model stability across different topologies."

### [1:30 - 2:00] Deployment Readiness & Impact
> "We serialized our best-performing pipeline using `joblib` into a ready-to-deploy `.joblib` model artifact. This model can be integrated into institutional Learning Management Systems via lightweight FastAPI endpoints to provide real-time risk alerts to teachers and academic counselors.
> 
> In summary, our project delivers a robust, submission-ready Machine Learning solution that empowers educators to help students succeed before exams take place. Thank you, and we welcome your questions!"
