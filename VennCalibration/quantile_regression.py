import xgboost as xgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

def quantile_regression(X: np.ndarray, y: np.ndarray, alpha: float, params_updated={}, test_size = 0.2):
    """
    Creates a 1D quantile regression model using XGBoost with quantile loss (pinball loss).
    This function includes early stopping to determine the optimal number of boosting rounds.

    Args:
        X (np.ndarray): Array of features.
        y (np.ndarray): Array of actual outcomes (labels).
        alpha (float): The quantile level for the pinball loss (between 0 and 1).
        params_updated (dict, optional): Dictionary of updated XGBoost parameters.

    Returns:
        function: A function that takes an array of features and returns quantile predictions.
    """
    X = np.atleast_2d(X)
    
    # Split data into training and validation sets for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dval = xgb.DMatrix(data=X_val, label=y_val)
    
    params = {
        'max_depth': 5,
        'min_child_weight': 20,
        'eta': 0.1,
        'objective': 'reg:quantileerror',
        'quantile_alpha': alpha,
        'lambda': 1,
        'gamma': 0,
        'verbosity': 0
    }
    params.update(params_updated)
    
    evals = [(dtrain, 'train'), (dval, 'eval')]
    
    # Train model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,            # Set a high upper limit
        evals=evals,
        early_stopping_rounds=20,        # Stop if no improvement in 20 rounds
        verbose_eval=False
    )
    
    def predictor(x):
        data_pred = xgb.DMatrix(data=np.atleast_2d(x))
        pred = model.predict(data_pred)
        return pred

    return predictor


 
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split

class QuantileRegression(BaseEstimator, RegressorMixin):
    def __init__(self, alpha: float = None, params_updated: dict = None, test_size: float = 0.2, **kwargs):
        """
        Initializes the QuantileRegression model with the specified quantile level and parameters.

        Args:
            alpha (float): The quantile level for the pinball loss (between 0 and 1).
            params_updated (dict, optional): Dictionary of updated XGBoost parameters.
            test_size (float, optional): Fraction of the data to use for validation during training.
        """
         
        self.alpha = alpha
        self.params_updated = params_updated if params_updated is not None else {}
        self.test_size = test_size
        self.model = None
        self.loss_name = "quantile"  # Required by MapieQuantileRegressor
        self.alpha_name = "alpha_name"  # Required by MapieQuantileRegressor

 

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the quantile regression model using the provided features and target.

        Args:
            X (np.ndarray): Array of features.
            y (np.ndarray): Array of actual outcomes (labels).
        """
        X = np.atleast_2d(X)
        
        # Split data into training and validation sets for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.test_size, random_state=42)
        
        dtrain = xgb.DMatrix(data=X_train, label=y_train)
        dval = xgb.DMatrix(data=X_val, label=y_val)
         
        params = {
        'max_depth': 5,
        'min_child_weight': 5,
        'eta': 0.1,
        'objective': 'reg:quantileerror',
        'quantile_alpha': self.alpha,
        'lambda': 1,
        'gamma': 0,
        'verbosity': 0
    }
        params.update(self.params_updated)
        
        evals = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train model with early stopping
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,            # Set a high upper limit
            evals=evals,
            early_stopping_rounds=20,        # Stop if no improvement in 20 rounds
            verbose_eval=False
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts quantile outcomes for the given input features.

        Args:
            X (np.ndarray): Array of features to predict on.

        Returns:
            np.ndarray: Predicted quantile outcomes.
        """
        if self.model is None:
            raise ValueError("The model has not been trained. Please call 'fit' first.")
        
        data_pred = xgb.DMatrix(data=np.atleast_2d(X))
        return self.model.predict(data_pred)

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): Whether to return parameters of nested objects.

        Returns:
            dict: Parameters of the model.
        """
        return {
            'alpha': self.alpha,
            'params_updated': self.params_updated,
            'test_size': self.test_size,
            'loss_name' : self.loss_name,  # Required by MapieQuantileRegressor
            'alpha_name' : self.alpha_name,
            'quantile': 'quantile'
        }

    def set_params(self, **params):
        """
        Set parameters for this estimator.

        Args:
            **params: Parameters to set.

        Returns:
            self: Returns the instance itself.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
