import os
import io
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_student_model.joblib")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student-mat.csv"


def generate_synthetic_dataset(n_samples=1000, random_state=42):
    """Generates realistic synthetic data matching UCI Student Performance dataset schema."""
    np.random.seed(random_state)

    schools = np.random.choice(['GP', 'MS'], size=n_samples, p=[0.88, 0.12])
    sexes = np.random.choice(['F', 'M'], size=n_samples, p=[0.53, 0.47])
    ages = np.random.randint(15, 22, size=n_samples)
    addresses = np.random.choice(['U', 'R'], size=n_samples, p=[0.78, 0.22])
    famsizes = np.random.choice(['LE3', 'GT3'], size=n_samples, p=[0.29, 0.71])
    pstatuses = np.random.choice(['T', 'A'], size=n_samples, p=[0.89, 0.11])

    medus = np.random.randint(0, 5, size=n_samples)
    fedus = np.random.randint(0, 5, size=n_samples)
    mjobs = np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples)
    fjobs = np.random.choice(['at_home', 'health', 'other', 'services', 'teacher'], size=n_samples)
    reasons = np.random.choice(['course', 'other', 'home', 'reputation'], size=n_samples)
    guardians = np.random.choice(['mother', 'father', 'other'], size=n_samples, p=[0.70, 0.23, 0.07])

    traveltimes = np.random.randint(1, 5, size=n_samples)
    studytimes = np.random.randint(1, 5, size=n_samples)
    failures = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.80, 0.12, 0.05, 0.03])

    schoolsups = np.random.choice(['yes', 'no'], size=n_samples, p=[0.13, 0.87])
    famsups = np.random.choice(['yes', 'no'], size=n_samples, p=[0.61, 0.39])
    paids = np.random.choice(['yes', 'no'], size=n_samples, p=[0.46, 0.54])
    activities = np.random.choice(['yes', 'no'], size=n_samples, p=[0.51, 0.49])
    nurserys = np.random.choice(['yes', 'no'], size=n_samples, p=[0.79, 0.21])
    highers = np.random.choice(['yes', 'no'], size=n_samples, p=[0.95, 0.05])
    internets = np.random.choice(['yes', 'no'], size=n_samples, p=[0.83, 0.17])
    romantics = np.random.choice(['yes', 'no'], size=n_samples, p=[0.33, 0.67])

    famrels = np.random.randint(1, 6, size=n_samples)
    freetimes = np.random.randint(1, 6, size=n_samples)
    goouts = np.random.randint(1, 6, size=n_samples)
    dalcs = np.random.randint(1, 6, size=n_samples)
    walcs = np.random.randint(1, 6, size=n_samples)
    healths = np.random.randint(1, 6, size=n_samples)
    absences = np.random.poisson(lam=5, size=n_samples)

    # Realistic relationship for G1, G2, G3
    base_ability = 3.0 * studytimes + 1.2 * (medus + fedus) - 1.5 * failures - 0.2 * absences + np.random.normal(5, 2, n_samples)
    g1 = np.clip(np.round(base_ability + np.random.normal(0, 1.5, n_samples)), 0, 20)
    g2 = np.clip(np.round(0.7 * g1 + 0.3 * base_ability + np.random.normal(0, 1.2, n_samples)), 0, 20)
    g3 = np.clip(np.round(0.55 * g2 + 0.35 * g1 + 0.1 * base_ability + np.random.normal(0, 1.0, n_samples)), 0, 20)

    df = pd.DataFrame({
        'school': schools, 'sex': sexes, 'age': ages, 'address': addresses,
        'famsize': famsizes, 'Pstatus': pstatuses, 'Medu': medus, 'Fedu': fedus,
        'Mjob': mjobs, 'Fjob': fjobs, 'reason': reasons, 'guardian': guardians,
        'traveltime': traveltimes, 'studytime': studytimes, 'failures': failures,
        'schoolsup': schoolsups, 'famsup': famsups, 'paid': paids,
        'activities': activities, 'nursery': nurserys, 'higher': highers,
        'internet': internets, 'romantic': romantics, 'famrel': famrels,
        'freetime': freetimes, 'goout': goouts, 'Dalc': dalcs, 'Walc': walcs,
        'health': healths, 'absences': absences, 'G1': g1, 'G2': g2, 'G3': g3
    })
    return df


def load_dataset():
    """Loads student performance dataset from local storage, UCI URL, or synthetic fallback."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"[+] Loading dataset from local disk: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        print("[*] Attempting to download UCI Student Performance dataset...")
        try:
            req = urllib.request.Request(UCI_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('utf-8')
                df = pd.read_csv(io.StringIO(content), sep=';')
                df.to_csv(DATA_PATH, index=False)
                print(f"[+] Dataset successfully downloaded and cached at: {DATA_PATH}")
        except Exception as e:
            print(f"[!] Remote download failed ({e}). Generating realistic synthetic dataset...")
            df = generate_synthetic_dataset(n_samples=1000)
            df.to_csv(DATA_PATH, index=False)
            print(f"[+] Synthetic dataset cached at: {DATA_PATH}")

    return df


def main():
    print("==========================================================================")
    print("   ML BUBBLE 2026 - TE-BE TRACK: STUDENT PERFORMANCE PREDICTION PIPELINE   ")
    print("==========================================================================")

    # 1. Load Data
    df = load_dataset()
    print(f"[*] Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features.")
    
    # 2. Define Features & Target
    target_col = 'G3'
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[*] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"[*] Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # 3. Data Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # 4. Train-Test Split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[*] Train-Test Split Completed: Train ({len(X_train)} samples), Test ({len(X_test)} samples).")

    # 5. Define Candidate Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    results = []
    trained_pipelines = {}

    print("\n--------------------------------------------------------------------------")
    print("                   MODEL TRAINING & EVALUATION BENCHMARK                 ")
    print("--------------------------------------------------------------------------")

    # 6. Model Benchmark Loop
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Model": name,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        })
        trained_pipelines[name] = (pipeline, r2)

        print(f" -> [{name}] Trained. MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # 7. Comparative Performance Table
    results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
    
    print("\n==========================================================================")
    print("                        COMPARATIVE PERFORMANCE SUMMARY                   ")
    print("==========================================================================")
    print(results_df.to_string(index=False))

    # 8. Save Best Performing Model
    best_model_name = results_df.iloc[0]["Model"]
    best_pipeline = trained_pipelines[best_model_name][0]
    best_r2 = results_df.iloc[0]["R2 Score"]

    joblib.dump(best_pipeline, MODEL_PATH)
    print("==========================================================================")
    print(f"[BEST] BEST MODEL SELECTED: '{best_model_name}' (R2 = {best_r2:.4f})")
    print(f"[+] Serialized model pipeline saved to: {MODEL_PATH}")
    print("==========================================================================\n")

if __name__ == "__main__":
    main()
