import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd

def generate_spline_basis_transform(X, n_splines=10):
    """
    Generate spline basis functions for each column in input data X using patsy,
    combine them, and remove collinear basis functions. Creates a transform
    function that can be applied to new data.

    If a column has fewer unique values than or equal to `n_splines`, it is one-hot encoded.

    Parameters:
        X (ndarray): Input data (2D array).
        n_splines (int): Number of splines to use for the basis functions.

    Returns:
        function: A transformation function that applies the same processing
                  to new data as performed on the original X.
    """
    # Ensure X is a 2D array
    if X.ndim == 1:
        X = X[:, np.newaxis]

    # Compute the number of unique values in each column of X
    unique_values_per_column = [len(np.unique(X[:, i])) for i in range(X.shape[1])]

    # Store transformation rules
    transformation_rules = []
    basis_list = []
    category_map = {}  # Map of column indices to categories for one-hot encoding

    for i in range(X.shape[1]):
        if unique_values_per_column[i] <= n_splines:
            # One-hot encode if unique values are <= n_splines
            one_hot_df = pd.get_dummies(X[:, i], prefix=f"col{i}")
            basis_list.append(one_hot_df.values)
            category_map[i] = one_hot_df.columns  # Store the one-hot categories
            transformation_rules.append({"type": "one-hot", "columns": X[:, i]})
        else:
            # Generate spline basis otherwise
            basis = dmatrix(f"bs(x, df={n_splines}, degree=3, include_intercept=False)", 
                            {"x": X[:, i]}, return_type='dataframe')
            basis_list.append(basis.values)
            transformation_rules.append({"type": "spline", "formula": f"bs(x, df={n_splines}, degree=3, include_intercept=False)"})

    # Concatenate all basis functions horizontally
    combined_basis = np.hstack(basis_list)

    # Remove basis functions with only a single unique value
    mask = np.apply_along_axis(lambda col: len(np.unique(col)) > 1, axis=0, arr=combined_basis)
    filtered_basis = combined_basis[:, mask]

    # Add an intercept column
    intercept = np.ones((filtered_basis.shape[0], 1))
    final_basis = np.hstack([intercept, filtered_basis])

    # Standardize to avoid scaling issues
    scaler = StandardScaler()
    final_basis = scaler.fit_transform(final_basis)

    # Remove collinear columns using SVD
    u, s, vh = svd(final_basis, full_matrices=False)
    tol = 1e-3  # Stricter tolerance
    non_collinear_columns = s > tol
    reduced_basis = final_basis[:, non_collinear_columns]

    # Define transformation function
    def basis_transform(new_X):
        """
        Apply the same transformations to new data as were applied to the original X.
        """
        if new_X.ndim == 1:
            new_X = new_X[:, np.newaxis]
        
        transformed_list = []
        for i, rule in enumerate(transformation_rules):
            if rule["type"] == "one-hot":
                # Apply one-hot encoding with the same categories as the original
                one_hot_df = pd.get_dummies(new_X[:, i], prefix=f"col{i}")
                one_hot_df = one_hot_df.reindex(columns=category_map[i], fill_value=0)
                transformed_list.append(one_hot_df.values)
            elif rule["type"] == "spline":
                # Apply spline transformation
                basis = dmatrix(rule["formula"], {"x": new_X[:, i]}, return_type='dataframe')
                transformed_list.append(basis.values)

        combined_transformed = np.hstack(transformed_list)
        combined_transformed = combined_transformed[:, mask]  # Apply the same mask
        combined_transformed = np.hstack([np.ones((combined_transformed.shape[0], 1)), combined_transformed])  # Add intercept
        combined_transformed = scaler.transform(combined_transformed)  # Standardize
        return combined_transformed[:, non_collinear_columns]  # Remove collinear columns

    return basis_transform
