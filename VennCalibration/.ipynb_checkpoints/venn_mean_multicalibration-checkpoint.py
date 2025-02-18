import numpy as np
from data_analysis.data_analysis_utils import *
from VennCalibration.quantile_regression import *
from data_analysis.datasets.datasets import GetDataset
from VennCalibration.basis_transform import *
 

 
import pandas as pd

def venn_multicalibration(
    predictor, X_cal, Y_cal, X_test, basis_transform=None, lambda_reg=1e-3, y_grid=None, y_test=None
):
    """
    Perform Venn regression multicalibration by generating prediction sets for a test set.

    Parameters:
        predictor (callable): Function to predict outcomes based on input features.
        X_cal (ndarray): Calibration features (n_cal x d), where n_cal is the number of calibration examples.
        Y_cal (ndarray): Calibration labels (n_cal x 1), corresponding to X_cal.
        X_test (ndarray): Test features (n_test x d), where n_test is the number of test examples.
        basis_transform (callable): Function to transform features into basis functions.
        lambda_reg (float): Ridge regularization parameter (default is 1e-3).
        y_test (ndarray, optional): True labels for test set. If provided, compute oracle predictions.
        quantile_grid (ndarray, optional): Grid of quantiles for Y_cal. If None, defaults to np.linspace(0.01, 0.99, 10).

    Returns:
        DataFrame: A Pandas DataFrame with columns:
                   - 'pred_original': Baseline prediction
                   - 'pred_calibrated': Calibrated prediction
                   - 'pred_lower': Lower bound of the prediction interval
                   - 'pred_upper': Upper bound of the prediction interval
                   - 'pred_oracle': Oracle prediction (if y_test is provided)
    """
    if basis_transform is None:
        basis_transform = generate_spline_basis_transform(X_cal, n_splines=10)
    if y_grid is None:
        quantile_grid = np.linspace(0, 1, 10)
        y_grid = np.quantile(Y_cal, quantile_grid)

    # Step 1: Compute predictions and basis transformations
    mu_cal = predictor(X_cal)  # Predictions on calibration set
    mu_test = predictor(X_test)  # Predictions on test set
    X_basis_cal = basis_transform(X_cal)  # Basis transformation for calibration features
    X_basis_test = basis_transform(X_test)  # Basis transformation for test features

    # Adjust calibration labels with mu_cal as offset
    Y_offset_cal = Y_cal - mu_cal
    Y_offset_cal = Y_offset_cal.reshape(-1, 1)  # Ensure Y_cal is a column vector

    # Step 2: Prepare ridge regression components
    XTX_basis = X_basis_cal.T @ X_basis_cal + lambda_reg * np.eye(X_basis_cal.shape[1])  # Regularized X^T X
    XTy_basis = X_basis_cal.T @ Y_offset_cal  # X^T Y
    XTX_inv = np.linalg.inv(XTX_basis)  # Inverse of X^T X

    beta_calibrated = XTX_inv @ XTy_basis  # Calibrated regression coefficients
    pred_calibrated = mu_test.reshape(-1) + (X_basis_test @ beta_calibrated).reshape(-1)  # Calibrated predictions

    # Step 3: Define prediction intervals for each test point
    predictions = np.zeros((X_test.shape[0], 2))  # Initialize prediction intervals
    predictions_oracle = np.zeros((X_test.shape[0], 1))

    for idx in range(X_test.shape[0]):
        x_new = X_basis_test[idx].reshape(1, -1)  # Single test point basis
        mu_new = mu_test[idx]  # Baseline prediction

        # Compute predictions for the quantile grid
        preds = []
        y_union = np.unique(np.append(y_grid, y_test[idx] if y_test is not None else []))
        for y_q in y_union:
            beta_updated = compute_updated_solution(
                XTX_inv, XTy_basis, x_new, y_q - mu_new, lambda_reg=lambda_reg
            )
            pred = mu_new + x_new @ beta_updated
            preds.append(pred)

        # Store the range (min, max) of predictions for the quantile grid
        predictions[idx] = [np.min(preds), np.max(preds)]

        # Compute oracle prediction if y_test is provided
        if y_test is not None:
            y_new = y_test[idx]  
            beta_oracle = compute_updated_solution(
                XTX_inv, XTy_basis, x_new, y_new - mu_new, lambda_reg=lambda_reg
            )
            pred_oracle = mu_new + x_new @ beta_oracle
            predictions_oracle[idx] = pred_oracle

    # Combine results into a DataFrame
    output = pd.DataFrame({
        "pred_original": mu_test.flatten(),
        "pred_calibrated": pred_calibrated.flatten(),
        "pred_lower": predictions[:, 0],
        "pred_upper": predictions[:, 1],
        "pred_oracle": predictions_oracle.flatten() if y_test is not None else np.nan
    })

    if y_test is None:
        output = output.drop(columns="pred_oracle")

    return output



def compute_updated_solution(XTX_inv, XTy, x_new, y_new, lambda_reg=1e-3):
    """
    Update the ridge regression solution with a new observation using the Sherman-Morrison formula.

    Parameters:
        XTX_inv (ndarray): Current inverse of (X^T X + lambda_reg * I), a (p x p) matrix.
        XTy (ndarray): Current X^T y, a (p x 1) vector.
        x_new (ndarray): New observation, a (1 x p) vector.
        y_new (float): New response value.
        lambda_reg (float): Ridge regularization parameter.
        verify (bool): If True, verify the updated solution by recomputing the regression from scratch (default is False).

    Returns:
        ndarray: Updated regression coefficients (p x 1 vector).
    """
    # Reshape x_new and y_new for consistent dimensions
    x_new = np.array(x_new).reshape(1, -1)  # Ensure x_new is a row vector
    if isinstance(y_new, (np.ndarray, list)):
        y_new = np.array(y_new).flatten()  # Flatten to 1D
        if y_new.size != 1:
            raise ValueError("y_new must be a scalar or a single-element array.")
        y_new = float(y_new[0])
    else:
        y_new = float(y_new)
        
    v = x_new.T  # Transpose for matrix operations
    u = x_new.T  # Duplicate for clarity

    # Update XTX inverse using Sherman-Morrison formula
    XTX_new_inv = XTX_inv - (XTX_inv @ u @ v.T @ XTX_inv) / (1 + v.T @ XTX_inv @ u)

    # Update XTy with the new contribution
    XTy_new = XTy + v * y_new

    # Compute the updated beta coefficients
    beta_updated = XTX_new_inv @ XTy_new
    return beta_updated


