 

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
# Package imports
from VennCalibration.calibrators import *
from VennCalibration.utils import *
from VennCalibration.quantile_regression import *
from VennCalibration.venn_quantile_calibration import *


def conformal_venn_prediction(X_train, Y_train, X_cal, Y_cal, alpha = 0.1, quantile_calibrator = None, calibrator_params = None, return_model = False):
    """
    Conformal Venn Prediction for calibrated conformal prediction intervals derived from Venn quantile calibration of conformity scores.

    Parameters:
    X_train : array-like
        Feature matrix for training data.
    Y_train : array-like
        Target values for training data.
    X_cal : array-like
        Feature matrix for calibration data.
    Y_cal : array-like
        Target values for calibration data.
    alpha : float, optional, default=0.1
        Significance level for prediction intervals (1-alpha is the confidence level).

    Returns:
    conformal_predictor : function
        A function that takes feature matrix `X` and returns the median prediction and calibrated prediction intervals.
    """
    # Train predictor of median outcome
    median_predictor = quantile_regression(X_train, Y_train, alpha=0.5)
    
    # Construct conformity scores
    S_train = np.abs(Y_train - median_predictor(X_train))
    S_cal = np.abs(Y_cal - median_predictor(X_cal))

    def inversion_map(lower, upper, X):
        """
        Map lower and upper bounds of the conformity score quantile to prediction intervals for Y.

        Parameters:
        lower : float
            Lower bound for conformity score.
        upper : float
            Upper bound for conformity score.
        X : array-like
            Feature matrix for which prediction intervals are generated.

        Returns:
        y_interval : ndarray
            Prediction intervals for target variable Y.
        """
        y_preds = median_predictor(X)
        y_interval = np.array([y_preds - upper, y_preds + upper]).T
        return y_interval
    
    # Train predictor of (1-alpha) quantile of conformity scores
    quantile_predictor = quantile_regression(X_train, S_train, alpha=1 - alpha)

    # Venn calibrate conformal quantile model
    VQC = VennQuantileCalibrator(quantile_predictor=quantile_predictor, alpha=alpha, quantile_calibrator = quantile_calibrator, calibrator_params = calibrator_params)
    VQC.calibrate(X_cal, S_cal)

    def conformal_predictor(X):
        """
        Generate calibrated prediction intervals and median predictions for a given feature matrix X.

        Parameters:
        X : array-like
            Feature matrix for which calibrated prediction intervals and median predictions are generated.

        Returns:
        median : ndarray
            Median predictions for the target variable Y.
        interval : ndarray
            Calibrated prediction intervals for the target variable Y.
        """
        median = median_predictor(X)
        interval = VQC.predict_interval(X, inversion_map=inversion_map)
        quantile = VQC.predict_point(X, calibrate = True)
        return median, interval, quantile

    if return_model:
        return conformal_predictor, VQC 
    return conformal_predictor