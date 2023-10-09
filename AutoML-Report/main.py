import numpy as np
import pandas as pd
import optuna
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import os
import datetime

# Create a 'result' directory if it doesn't exist
if not os.path.exists("result"):
    os.mkdir("result")


# Define a function to load and split the California Housing dataset for regression
def load_and_split_regression_data():
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Define a function to load and split the dataset for classification
def load_and_split_classification_data():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Define objective function for Optuna optimization
def optimize_model(trial, X_train, y_train, X_test, y_test, task, model_func):
    model = model_func(trial)
    model.fit(X_train, y_train)

    if task == "regression":
        loss_fn = mean_squared_error
    elif task == "classification":
        loss_fn = accuracy_score

    y_pred = model.predict(X_test)
    score = loss_fn(y_test, y_pred)

    return model, score


# Define a function to perform AutoML for XGBoost
def xgboost_model(trial):
    params = {
        "n_estimators": trial.suggest_int("xgboost_n_estimators", 50, 100),
        "max_depth": trial.suggest_int("xgboost_max_depth", 3, 20),
        "learning_rate": trial.suggest_float("xgboost_learning_rate", 0.001, 1.0),
        "subsample": trial.suggest_float("xgboost_subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("xgboost_colsample_bytree", 0.1, 1.0),
        "min_child_weight": trial.suggest_int("xgboost_min_child_weight", 1, 10),
        "gamma": trial.suggest_float("xgboost_gamma", 0.0, 2.0),
        "lambda": trial.suggest_float("xgboost_lambda", 0.0, 2.0),
        "alpha": trial.suggest_float("xgboost_alpha", 0.0, 2.0),
        "eval_metric": "rmse" if task == "regression" else "logloss"
    }
    model = XGBRegressor(**params, random_state=42) if task == "regression" else XGBClassifier(**params,
                                                                                               random_state=42)
    return model


# Define a function to perform AutoML for CatBoost
def catboost_model(trial):
    params = {
        "n_estimators": trial.suggest_int("catboost_n_estimators", 50, 200),
        "max_depth": trial.suggest_int("catboost_max_depth", 3, 20),
        "learning_rate": trial.suggest_float("catboost_learning_rate", 0.001, 1.0),
        "subsample": trial.suggest_float("catboost_subsample", 0.1, 1.0),
        "colsample_bylevel": trial.suggest_float("catboost_colsample_bylevel", 0.1, 1.0),
        "min_child_samples": trial.suggest_int("catboost_min_child_samples", 1, 20),
        "l2_leaf_reg": trial.suggest_float("catboost_l2_leaf_reg", 0.0, 10.0),
        "border_count": trial.suggest_int("catboost_border_count", 1, 255),
        "eval_metric": "RMSE" if task == "regression" else "Logloss"
    }
    model = CatBoostRegressor(**params, random_state=42) if task == "regression" else CatBoostClassifier(**params,
                                                                                                         random_state=42)
    return model


# Define a function to perform AutoML for RandomForest
def randomforest_model(trial):
    params = {
        "n_estimators": trial.suggest_int("randomforest_n_estimators", 50, 200),
        "max_depth": trial.suggest_int("randomforest_max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("randomforest_min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("randomforest_min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("randomforest_max_features", 0.1, 1.0),
        "min_impurity_decrease": trial.suggest_float("randomforest_min_impurity_decrease", 0.0, 0.2)
    }
    model = RandomForestRegressor(**params, random_state=42) if task == "regression" else RandomForestClassifier(
        **params, random_state=42)
    return model


# Define a function to perform AutoML for LightGBM
def lightgbm_model(trial):
    params = {
        "n_estimators": trial.suggest_int("lightgbm_n_estimators", 50, 200),
        "max_depth": trial.suggest_int("lightgbm_max_depth", 3, 20),
        "learning_rate": trial.suggest_float("lightgbm_learning_rate", 0.001, 1.0),
        "subsample": trial.suggest_float("lightgbm_subsample", 0.1, 1.0),
        "colsample_bytree": trial.suggest_float("lightgbm_colsample_bytree", 0.1, 1.0),
        "min_child_samples": trial.suggest_int("lightgbm_min_child_samples", 1, 20),
        "reg_lambda": trial.suggest_float("lightgbm_reg_lambda", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("lightgbm_reg_alpha", 0.0, 2.0),
        "min_split_gain": trial.suggest_float("lightgbm_min_split_gain", 0.0, 1.0)
    }
    model = LGBMRegressor(**params, random_state=42) if task == "regression" else LGBMClassifier(**params,
                                                                                                 random_state=42)
    return model


# Define a function to perform AutoML for a specific model
def perform_automl(task, model_name, model_func):
    if task == "regression":
        X_train, X_test, y_train, y_test = load_and_split_regression_data()
    elif task == "classification":
        X_train, X_test, y_train, y_test = load_and_split_classification_data()

    study = optuna.create_study(direction="minimize")
    best_model, best_score = optimize_model(study, X_train, y_train, X_test, y_test, task, model_func)

    # Get the current date as a string
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    # Save the best parameters as a JSON file with the current date
    with open(f"result/{model_name}_best_params_{current_date}.json", "w") as json_file:
        json.dump(best_model.get_params(), json_file)

    # Save the best model as a PKL file with the current date
    with open(f"result/{model_name}_best_model_{current_date}.pkl", "wb") as pkl_file:
        pickle.dump(best_model, pkl_file)

    # Get feature importance for the best model
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
        sorted_indices = np.argsort(importance)[::-1]
        top_features = X_train.columns[sorted_indices][:20]

        # Create and save a bar chart for feature importance as an image with the current date
        plt.figure(figsize=(12, 6))
        sns.barplot(x=importance[sorted_indices][:20], y=top_features)
        plt.title(f"Top 20 Feature Importance - {model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.savefig(f"result/{model_name}_feature_importance_{current_date}.png")

    print(f"{model_name} - Best Score:", best_score)


# Get user input for task (regression or classification)
task = input("Enter 'regression' or 'classification': ")

# Perform AutoML for each model
models_to_run = {
    "XGBoost": xgboost_model,
    "CatBoost": catboost_model,
    "RandomForest": randomforest_model,
    "LightGBM": lightgbm_model,
}

for model_name, model_func in models_to_run.items():
    perform_automl(task, model_name, model_func)
