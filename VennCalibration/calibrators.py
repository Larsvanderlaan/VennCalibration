import numpy as np
import xgboost as xgb
from VennCalibration.utils import *


def mean_calibrator_isotonic(f: np.ndarray, y: np.ndarray, max_depth=20, min_child_weight=10):
    """
    Creates a 1D calibration function based on isotonic regression using XGBoost. This function
    fits an XGBoost model to predict `y` from `f` ensuring a monotonic relationship.

    Args:
        f (np.ndarray): Array of uncalibrated predictions (features).
        y (np.ndarray): Array of actual outcomes (labels).
        max_depth (int, optional): Maximum depth of each tree used in the XGBoost model. Defaults to 20.
        min_child_weight (int, optional): Minimum sum of instance weight needed in a child node. Defaults to 20.

    Returns:
        function: A function that takes an array of model predictions and returns calibrated predictions.
    """
    data = xgb.DMatrix(data=np.array(f).reshape(-1, 1), label=y)
    iso_fit = xgb.train(params={
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'monotone_constraints': '(1)',
        'eta': 1, 'gamma': 0,
        'lambda': 0
    }, dtrain=data, num_boost_round=1)

    def transform(x):
        data_pred = xgb.DMatrix(data=np.array(x).reshape(-1, 1))
        pred = iso_fit.predict(data_pred)
        return pred

    return transform

 

def quantile_calibrator_isotonic(f: np.ndarray, y: np.ndarray, alpha: float, max_depth=20, min_child_weight=10, num_boost_round=20):
    """
    Creates a 1D calibration function based on XGBoost using quantile loss (pinball loss).
    This function fits an XGBoost model to predict `y` from `f` ensuring a monotonic relationship.

    Args:
        f (np.ndarray): Array of uncalibrated predictions (features).
        y (np.ndarray): Array of actual outcomes (labels).
        alpha (float): The quantile level for the pinball loss (between 0 and 1).
        max_depth (int, optional): Maximum depth of each tree used in the XGBoost model. Defaults to 20.
        min_child_weight (int, optional): Minimum sum of instance weight needed in a child node. Defaults to 20.

    Returns:
        function: A function that takes an array of model predictions and returns calibrated predictions.
    """
    data = xgb.DMatrix(data=np.array(f).reshape(-1, 1), label=y)
    params = {
        'max_depth': max_depth,
        'min_child_weight': min_child_weight,
        'eta': 1,
        'objective': 'reg:quantileerror',
        'monotone_constraints': '(1)',
        'quantile_alpha': alpha,
        'lambda': 1,
        'gamma': 0,
        'verbosity': 0
    }

    booster = xgb.train(params=params, dtrain=data, num_boost_round=num_boost_round)

    # monotonize since xgboost may not be totally monotone and may have too many bins due to boosting.
    #data_pred = xgb.DMatrix(data=np.array(f).reshape(-1, 1))
    #pred = booster.predict(data_pred)
    #isotonic_projector = regression_calibrator_isotonic(f = f, y = pred, max_depth=max_depth, min_child_weight=min_child_weight)
    isotonic_projector = lambda x: x
    
    def transform(x):
        data_pred = xgb.DMatrix(data=np.array(x).reshape(-1, 1))
        pred = isotonic_projector(booster.predict(data_pred))
        return pred

    return transform


def quantile_calibrator_histogram(f: np.ndarray, y: np.ndarray, alpha: float, num_bin=10, binning_method="quantile"):
    """
    Creates a calibration function based on histogram binning. It divides the prediction space into
    bins and assigns the mean of actual outcomes within each bin as the calibrated prediction.

    Args:
        f (np.ndarray): Array of uncalibrated predictions.
        y (np.ndarray): Array of actual outcomes.
        num_bin (int, optional): Number of bins for histogram binning. Defaults to 10.
        binning_method (str, optional): Method for creating bins ('quantile' or 'fixed'). Defaults to "quantile".

    Returns:
        function: A function that maps original predictions to calibrated predictions based on the bin averages.
    """
    grid = make_grid(f, num_bin, binning_method=binning_method)
    bin_ids = match_grid_value(f, grid, return_index=True, all_inside=True)
    bin_preds = [np.quantile(y[bin_ids == bin_id], alpha) for bin_id in sorted(set(bin_ids))]

    def transform(x):
        bin_ids = match_grid_value(x, grid, return_index=True, all_inside=True)
        values = [bin_preds[bin_id] for bin_id in bin_ids]
        return np.array(values)

    return transform
