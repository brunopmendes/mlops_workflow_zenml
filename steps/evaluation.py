import logging

import pandas as pd
import mlflow
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from typing import Tuple, Annotated
from src.evaluation import MSE, R2, RMSE
import numpy as np

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "mse"],
    ]:
    
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse, mse
    except Exception as e:
        logging.error(f"Error in evaluating model {e}")
        raise e