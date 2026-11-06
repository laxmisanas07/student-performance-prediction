"""
Student Performance Prediction ML Pipeline
ML Bubble 2026 - TE-BE Advanced Track | Army Institute of Technology (AIT), Pune

Trains and benchmarks multiple regression architectures on the UCI Student Performance Dataset:
- Linear Regression (Baseline)
- Random Forest Regressor
- Gradient Boosting Regressor (Selected Best Model)

Evaluates MAE, RMSE, and R² Score, saves the best serialized pipeline,
and exports verified benchmark metrics to models/model_metrics.json.
"""

import os
import io
import json
import urllib.request
import zipfile
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(DATA_DIR, "student_data.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "best_student_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def generate_synthetic_dataset(n_samples=1000, random_state=42):
    """
    Generates realistic synthetic student performance data as a fallback only.
    Clearly marks data provenance as synthetic.
    """
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
    """
    Loads student performance dataset.
    Prioritizes authentic UCI benchmark data (395 samples for mathematics course).
    Falls back gracefully if network is unavailable, recording data provenance clearly.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    data_source = "Local Cached UCI Dataset"

    if os.path.exists(DATA_PATH):
        print(f"[+] Loading dataset from local disk: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        if len(df) == 395:
            data_source = "Authentic UCI Student Performance Dataset (Mathematics Course)"
        else:
            data_source = f"Local Dataset ({len(df)} samples)"
    else:
        print("[*] Local dataset not found. Downloading authentic UCI benchmark dataset...")
        try:
            req = urllib.request.Request(UCI_ZIP_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=12) as response:
                z_outer = zipfile.ZipFile(io.BytesIO(response.read()))
                inner_bytes = z_outer.read('student.zip')
                z_inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
                csv_bytes = z_inner.read('student-mat.csv')
                df = pd.read_csv(io.BytesIO(csv_bytes), sep=';')
                df.to_csv(DATA_PATH, index=False)
                data_source = "Authentic UCI Student Performance Dataset (Mathematics Course)"
                print(f"[+] UCI Dataset successfully downloaded and cached at: {DATA_PATH}")
        except Exception as e:
            print(f"[!] Remote download failed ({e}). Generating realistic synthetic fallback dataset...")
            df = generate_synthetic_dataset(n_samples=1000)
            df.to_csv(DATA_PATH, index=False)
            data_source = "Synthetic Demonstration Fallback (Generated locally due to network failure)"
            print(f"[+] Synthetic fallback dataset cached at: {DATA_PATH}")

    return df, data_source


def build_pipeline(regressor, numerical_cols, categorical_cols):
    """Constructs a reproducible scikit-learn preprocessing and prediction pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])


def train_and_evaluate():
    """Executes the full model training, cross-model comparison, and artifact serialization."""
    print("==========================================================================")
    print("   ML BUBBLE 2026 - TE-BE TRACK: STUDENT PERFORMANCE PREDICTION PIPELINE   ")
    print("   Army Institute of Technology (AIT), Pune                              ")
    print("==========================================================================")

    # 1. Load Data
    df, data_source = load_dataset()
    print(f"[*] Data Provenance: {data_source}")
    print(f"[*] Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features.")

    # 2. Define Features & Target
    target_col = 'G3'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"[*] Categorical features ({len(categorical_cols)}): {categorical_cols}")
    print(f"[*] Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # 3. Train-Test Split (80% Train, 20% Test, random_state=42 for exact reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[*] Train-Test Split Completed: Train ({len(X_train)} samples), Test ({len(X_test)} samples).")

    # 4. Candidate Model Configurations
    models = {
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=6, random_state=42
        ),
        "Linear Regression": LinearRegression()
    }

    results = []
    trained_pipelines = {}

    print("\n--------------------------------------------------------------------------")
    print("                   MODEL TRAINING & EVALUATION BENCHMARK                 ")
    print("--------------------------------------------------------------------------")

    # 5. Benchmark Models
    for name, model_instance in models.items():
        pipeline = build_pipeline(model_instance, numerical_cols, categorical_cols)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))

        results.append({
            "model_name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4)
        })
        trained_pipelines[name] = pipeline

        print(f" -> [{name}] MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # 6. Evaluation Criterion: Lowest RMSE (highest predictive accuracy with minimal variance penalty)
    results_sorted = sorted(results, key=lambda x: (x["rmse"], -x["r2_score"]))
    best_candidate = results_sorted[0]
    best_model_name = best_candidate["model_name"]
    best_pipeline = trained_pipelines[best_model_name]

    print("\n==========================================================================")
    print("                        COMPARATIVE PERFORMANCE SUMMARY                   ")
    print("==========================================================================")
    summary_df = pd.DataFrame(results_sorted)
    print(summary_df.to_string(index=False))

    # 7. Serialize Best Model Pipeline
    joblib.dump(best_pipeline, MODEL_PATH)
    print("\n==========================================================================")
    print(f"[BEST] SELECTED MODEL: '{best_model_name}' (RMSE = {best_candidate['rmse']:.4f}, R² = {best_candidate['r2_score']:.4f})")
    print(f"[+] Serialized model pipeline saved to: {MODEL_PATH}")

    # 8. Persist Actual Metrics to JSON for Streamlit App Consumption
    metrics_payload = {
        "selected_model": best_model_name,
        "selection_criterion": "Lowest Root Mean Squared Error (RMSE) on 20% holdout test set",
        "data_provenance": data_source,
        "sample_size": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_variable": "G3 (Final Mathematics Grade, 0-20 scale)",
        "features_count": len(X.columns),
        "categorical_features": categorical_cols,
        "numerical_features": numerical_cols,
        "best_metrics": {
            "mae": best_candidate["mae"],
            "rmse": best_candidate["rmse"],
            "r2_score": best_candidate["r2_score"]
        },
        "all_benchmarks": results_sorted
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"[+] Actual model benchmark metrics persisted to: {METRICS_PATH}")
    print("==========================================================================\n")
    return best_model_name, best_candidate, metrics_payload


def main():
    train_and_evaluate()


if __name__ == "__main__":
    main()
