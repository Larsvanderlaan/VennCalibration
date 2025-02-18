 

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
# Package imports
from VennCalibration.calibrators import *
from VennCalibration.utils import *
from VennCalibration.quantile_regression import *

 


import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class VennQuantileCalibrator:
    def __init__(self, quantile_predictor: callable, alpha=0.1,
                 quantile_calibrator=None,  # Default set later if None
                 calibrator_params=None,  # Default set later if None
                 algo_params=None):  # Default set later if None
        """
        Initializes the VennConformalQuantile class.

        Parameters:
        quantile_predictor (callable): Function for making quantile predictions.
        alpha (float): Significance level for prediction intervals (default: 0.1).
        quantile_calibrator (callable): Calibration function for predictor outputs.
        calibrator_params (dict): Parameters for the calibration function.
        algo_params (dict): Algorithm parameters for binning and calibration.
        """
        if quantile_calibrator is None:
            quantile_calibrator = quantile_calibrator_isotonic
            calibrator_params = {'max_depth': 10, 'min_child_weight': 10, 'num_boost_round': 20}
        if calibrator_params is None:
            calibrator_params = {}
        if algo_params is None:
            algo_params = {'num_bin_predictor': 100, 'num_bin_Y': 3, 'binning_method': "quantile"}

        self.alpha = alpha
        self.quantile_predictor = quantile_predictor
        self.quantile_calibrator = quantile_calibrator
        self.quantile_calibrator_params = calibrator_params
        self.num_bin_predictor = algo_params['num_bin_predictor']
        self.num_bin_Y = algo_params['num_bin_Y']
        self.binning_method = algo_params['binning_method']

    def calibrate(self, X: np.ndarray, Y: np.ndarray, Y_grid=None):
        """
        Calibrates the predictor using training data.

        Parameters:
        X (np.ndarray): Feature/covariate data for calibration.
        Y (np.ndarray): Outcome variable data for calibration.
        Y_grid (tuple, optional): Range of Y values for binning. If None, calculated from Y.
        """
        alpha = self.alpha

        # Define default Y_grid if not provided
        if Y_grid is None:
            Y_range = [min(Y), max(Y)]
            Y_grid = make_grid(Y, self.num_bin_Y, Y_range, binning_method=self.binning_method)

        # Extended range for interpolation
        Y_range = [min(Y_grid), max(Y_grid)]
        Y_interp = make_grid(Y, 1000, Y_range, binning_method="quantile")

        # Predictions and bins
        preds = np.array(self.quantile_predictor(X))
        preds_grid = make_grid(preds, self.num_bin_predictor, binning_method=self.binning_method)

        # Initialize prediction sets and intervals
        prediction_sets = pd.Series([[] for _ in preds_grid])
        predictions_intervals = pd.Series([[] for _ in preds_grid])

        # Augmented datasets for calibration
        list_preds_augmented = [np.hstack([preds, pred]) for pred in preds_grid]
        list_Y_augmented = [np.hstack([Y, Y_val]) for Y_val in Y_grid]

        # Calibrate predictions
        quantile_calibrator = self.quantile_calibrator(f=preds, y=Y, alpha=1 - alpha, **self.quantile_calibrator_params)
        predsibrated_grid = quantile_calibrator(preds_grid)

        def VennCalibrate(index_pred, pred):
            """
            Performs Venn calibration for a specific prediction.
            """
            preds_augmented = list_preds_augmented[index_pred]
            multipred_venn_abers = np.zeros(len(Y_grid))
            thresholds = np.zeros(len(Y_grid))
            test_Ys = np.zeros(len(Y_grid))

            for index_Y, Y_val in enumerate(Y_grid):
                Y_augmented = list_Y_augmented[index_Y]
                quantile_calibrator = self.quantile_calibrator(
                    f=preds_augmented, y=Y_augmented, alpha=1 - alpha, **self.quantile_calibrator_params)
                preds_augmented_calibrated = quantile_calibrator(preds_augmented)
                pred_test_calibrated = preds_augmented_calibrated[-1]
                test_Y = Y_augmented[-1]

                test_Ys[index_Y] = test_Y
                thresholds[index_Y] = pred_test_calibrated

            # Interpolate scores and thresholds
            test_Ys_interp = np.interp(Y_interp, Y_grid, test_Ys)
            thresholds_interp = np.interp(Y_interp, Y_grid, thresholds)

            # Construct prediction set
            prediction_set = [
                y for y, threshold in zip(test_Ys_interp, thresholds_interp)
                if y <= threshold
            ]
            prediction_sets[index_pred] = prediction_set
            predictions_intervals[index_pred] = [min(prediction_set), max(prediction_set)]

        # Apply Venn calibration to all predictions
        for index_pred, pred in enumerate(preds_grid):
            VennCalibrate(index_pred, pred)

        # Store fit information
        self.fit_info = pd.DataFrame({
            "prediction_uncal": preds_grid,
            "prediction_cal": predsibrated_grid,
            "prediction_set": prediction_sets,
            "prediction_interval": predictions_intervals
        })

    def predict_point(self, X: np.ndarray, calibrate=True):
        """
        Generates point predictions for given features.

        Parameters:
        X (np.ndarray): Input features.
        calibrate (bool): If True, applies calibration to predictions.

        Returns:
        np.ndarray: Predicted quantile values.
        """
        f = np.array(self.quantile_predictor(X))
        if calibrate:
            return self._extrapolate(self.fit_info['prediction_uncal'], self.fit_info['prediction_cal'], f)
        return f

    def predict_interval(self, X: np.ndarray, inversion_map: callable = None):
        """
        Outputs range of Venn prediction set for input features.

        Parameters:
        X (np.ndarray): Input features.
        inversion_map (callable, optional): Function to apply inverse mapping to intervals.

        Returns:
        np.ndarray: Prediction intervals (lower and upper bounds).
        """
        f = np.array(self.quantile_predictor(X))
        f_grid = self.fit_info['prediction_uncal']
        bounds = [(row[0], row[1]) for row in self.fit_info['prediction_interval']]
        lower = self._extrapolate(f_grid, [b[0] for b in bounds], f)
        upper = self._extrapolate(f_grid, [b[1] for b in bounds], f)

        if inversion_map is not None:
            return inversion_map(lower, upper, X)
        return np.array([lower, upper]).T

    def _extrapolate(self, x_grid, Y_grid, x_new):
        """
        Performs extrapolation or smoothing for given x values.

        Parameters:
        x_grid (array-like): Known x values.
        Y_grid (array-like): Known y values.
        x_new (array-like): New x values to predict.

        Returns:
        np.ndarray: Predicted y values.
        """
        interp = interp1d(x_grid, Y_grid, kind='nearest', bounds_error=False, fill_value="extrapolate")
        return interp(x_new)

    