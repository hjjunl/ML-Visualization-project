import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb
import catboost as ctb
import lightgbm as lgb

# Define the objective function
def objective(trial):

    # Define the parameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
    }

    # Create the model
    model = choose_model(trial.suggest_categorical('model', ['random_forest', 'xgboost', 'catboost', 'lgbm']))
    model.set_params(**params)

    # Train the model
    model.fit(X_train, y_train)

    # Predict the test data
    y_pred = model.predict(X_test)

    # Calculate the evaluation metric
    if trial.suggest_categorical('task', ['regression', 'classification']) == 'regression':
        metric = mean_squared_error(y_test, y_pred)
    else:
        metric = accuracy_score(y_test, y_pred)

    # Return the evaluation metric
    return metric

# Choose the model based on the trial's suggestion
def choose_model(model_name):
    if model_name == 'random_forest':
        return RandomForestRegressor()
    elif model_name == 'xgboost':
        return xgb.XGBRegressor()
    elif model_name == 'catboost':
        return ctb.CatBoostRegressor()
    elif model_name == 'lgbm':
        return lgb.LGBMRegressor()
    else:
        raise ValueError(f'Invalid model name: {model_name}')

