"""Module 2 — Classification de l'état actuel de l'eau, section 5.2."""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from data_gen import generate_water_quality_dataset

FEATURES = ["PH", "TEMP", "DO", "TURBIDITY"]


def balance_classes(df, seed=42):
    n_min = df["label"].value_counts().min()
    parts = [df[df.label == c].sample(n_min, random_state=seed) for c in df.label.unique()]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def train_all_models():
    df = generate_water_quality_dataset()
    df_bal = balance_classes(df)

    X = df_bal[FEATURES]
    y = df_bal["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    grids = {
        "Régression Logistique": (
            LogisticRegression(max_iter=1000, solver="saga", random_state=42),
            {"C": [0.1, 1]},
            True,
        ),
        "Arbre de Décision": (
            DecisionTreeClassifier(random_state=42),
            {"criterion": ["gini", "entropy"], "max_depth": [None, 10, 20]},
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [50, 100], "max_depth": [None, 10, 20]},
            False,
        ),
        "SVM (SVC)": (SVC(probability=True, random_state=42), {"C": [1], "kernel": ["rbf", "linear"]}, True),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
            {"n_estimators": [50, 100], "learning_rate": [0.1, 0.2], "max_depth": [3, 5]},
            False,
        ),
    }

    results, models = {}, {}
    for name, (estimator, grid, needs_scaling) in grids.items():
        Xtr, Xte = (X_train_s, X_test_s) if needs_scaling else (X_train, X_test)
        gs = GridSearchCV(estimator, grid, cv=3, scoring="recall", n_jobs=-1)
        gs.fit(Xtr, y_train)
        best = gs.best_estimator_
        y_pred = best.predict(Xte)
        y_proba = best.predict_proba(Xte)[:, 1]

        results[name] = {
            "best_params": gs.best_params_,
            "cv_recall": round(gs.best_score_, 4),
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4),
        }
        models[name] = {"model": best, "needs_scaling": needs_scaling}

    return results, models, scaler


def predict_quality(models, scaler, model_name, ph, temp, do, turbidity):
    row = pd.DataFrame([{"PH": ph, "TEMP": temp, "DO": do, "TURBIDITY": turbidity}])
    info = models[model_name]
    X = scaler.transform(row) if info["needs_scaling"] else row
    pred = int(info["model"].predict(X)[0])
    proba = float(info["model"].predict_proba(X)[0][1])
    return pred, proba
