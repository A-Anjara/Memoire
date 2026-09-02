"""Module 1 — Estimation de la densité de Tilapia (régression), section 5.1."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor

from data_gen import generate_module1_data, TEXTURE_ENC, COULEUR_ENC, IRRIGATION_ENC, ENGRAIS_ENC, TEMP_ENC

FEATURES = ["Texture_sol", "Couleur_sol", "Type_irrigation", "Hauteur_Eau", "Niveau_Engrais", "Temperature_Actuelle"]
TARGET = "Densite_Tilapia_Are"


def encode(df):
    out = df.copy()
    out["Texture_sol"] = out["Texture_sol"].map(TEXTURE_ENC)
    out["Couleur_sol"] = out["Couleur_sol"].map(COULEUR_ENC)
    out["Type_irrigation"] = out["Type_irrigation"].map(IRRIGATION_ENC)
    out["Niveau_Engrais"] = out["Niveau_Engrais"].map(ENGRAIS_ENC)
    out["Temperature_Actuelle"] = out["Temperature_Actuelle"].map(TEMP_ENC)
    return out


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def train_all_models():
    raw = generate_module1_data()
    df = encode(raw).drop_duplicates()

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    grids = {
        "Régression Linéaire": (LinearRegression(), {"fit_intercept": [True, False]}, True),
        "Arbre de Décision": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]},
            False,
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10], "max_features": ["sqrt", "log2"]},
            False,
        ),
        "SVM (SVR)": (SVR(), {"C": [1, 10], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}, True),
        "XGBoost": (
            XGBRegressor(random_state=42, verbosity=0),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]},
            False,
        ),
    }

    results, models = {}, {}
    for name, (estimator, grid, needs_scaling) in grids.items():
        Xtr, Xte = (X_train_s, X_test_s) if needs_scaling else (X_train, X_test)
        gs = GridSearchCV(estimator, grid, cv=3, scoring="r2", n_jobs=-1)
        gs.fit(Xtr, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(Xte)

        results[name] = {
            "best_params": gs.best_params_,
            "cv_r2": round(gs.best_score_, 4),
            "RMSE": round(rmse(y_test, y_pred), 3),
            "MAE": round(mean_absolute_error(y_test, y_pred), 3),
            "MAPE": round(mean_absolute_percentage_error(y_test, y_pred), 4),
            "R2": round(r2_score(y_test, y_pred), 4),
        }
        models[name] = {"model": best, "needs_scaling": needs_scaling}

    return results, models, scaler


def predict_density(models, scaler, model_name, texture, couleur, irrigation, hauteur_eau, engrais, temperature):
    row = pd.DataFrame(
        [
            {
                "Texture_sol": TEXTURE_ENC[texture],
                "Couleur_sol": COULEUR_ENC[couleur],
                "Type_irrigation": IRRIGATION_ENC[irrigation],
                "Hauteur_Eau": hauteur_eau,
                "Niveau_Engrais": ENGRAIS_ENC[engrais],
                "Temperature_Actuelle": TEMP_ENC[temperature],
            }
        ]
    )
    info = models[model_name]
    X = scaler.transform(row) if info["needs_scaling"] else row
    return max(0.0, float(info["model"].predict(X)[0]))
