import logging

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from .config import ModelNameConfig

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        X_train: pd.DataFrame
        X_test: pd.DataFrame
        y_train: pd.DataFrame
        y_test: pd.DataFrame
    """
    try:
        model = None
        if config.model_name == "LinearRegression":

            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)
        # if config.model_name == "RandomForestRegressor":
            # model = RandomForestRegressionModel
            # trained_model = model.train(X_train, y_train)
        else: 
            raise ValueError(f"Model {config.model_name} not supported")
        
        return trained_model
    except Exception as e:
        logging.error(f"Error in training model {e}")
        raise e
