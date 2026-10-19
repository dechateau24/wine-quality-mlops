import json
import os

import joblib

# Necessaire pour utiliser le dossier ./mlruns avec les versions recentes de MLflow
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPERIMENT_NAME = "wine_quality"
MODEL_NAME = "WineQualityModel"
MODEL_ALIAS = "production"


def build_model(name, params):
    """Cree le modele a partir de son nom et de ses hyperparametres."""
    if name == "ridge":
        # Le modele lineaire a besoin de variables a la meme echelle
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "random_forest":
        return RandomForestRegressor(**params)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Modele inconnu : {name}")


MODEL_NAMES = ["ridge", "random_forest", "gradient_boosting"]


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze("columns")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze("columns")
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one(name, params, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        model = build_model(name, params)
        model.fit(X_train, y_train)

        metrics = evaluate(y_test, model.predict(X_test))

        mlflow.log_param("model_type", name)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        info = mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            input_example=X_train.iloc[:3],
        )
    return metrics, info.model_uri, model


def register_best_model(model_uri):
    result = mlflow.register_model(model_uri, MODEL_NAME)
    mlflow.MlflowClient().set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, result.version)
    return result.version


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = load_data()

    results = {}
    for name in MODEL_NAMES:
        metrics, uri, model = train_one(name, params[name], X_train, X_test, y_train, y_test)
        results[name] = (metrics, uri, model)
        print(f"[train] {name:<18} RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}")

    # Le meilleur modele est celui dont la RMSE est la plus basse
    best_name = min(results, key=lambda n: results[n][0]["rmse"])
    best_metrics, best_uri, best_model = results[best_name]

    version = register_best_model(best_uri)
    print(f"[train] Meilleur modele : {best_name} -> {MODEL_NAME} v{version} (alias @{MODEL_ALIAS})")

    # Export du meilleur modele pour l'API (fichier autonome, sans dependance a mlruns)
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model.joblib")
    with open("models/features.json", "w", encoding="utf-8") as f:
        json.dump(list(X_train.columns), f)

    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, **best_metrics}, f, indent=2)


if __name__ == "__main__":
    main()