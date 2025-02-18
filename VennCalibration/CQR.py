import lightgbm as lgb
from mapie.regression import MapieQuantileRegressor
from VennCalibration.quantile_regression import *
import numpy as np


def CQR(X_train, Y_train, X_cal, Y_cal, alpha=0.1):
    # Define the LightGBM model for quantile regression
    params_lgb = {
        "objective": "quantile",
        "alpha": 0.5,  # Default median, we'll adjust for lower and upper quantiles
        "n_estimators": 100,
        "min_data_in_leaf": 30,
        "max_depth": 5,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "boosting_type": "gbdt",
        "random_state": 42,
        'verbose': -1
    }

    # Set up the MapieQuantileRegressor
    MapieQuantileRegressor.quantile_estimator_params["QuantileRegression"] = {"loss_name": "quantile",
            "alpha_name": "alpha"
    }
    mapie_regressor = MapieQuantileRegressor(
        estimator= QuantileRegression(), #lgb.LGBMRegressor(**params_lgb),
        alpha=alpha
    )
    loss_name, alpha_name = mapie_regressor.quantile_estimator_params[
                        "QuantileRegression"
                    ].values()
    params = QuantileRegression().get_params()
 

    
 
   

    # Fit the model on the training data
    mapie_regressor.fit(X_train, Y_train, X_calib=X_cal,
    y_calib=Y_cal)

        
    def conformal_predictor(X):
        median, interval = mapie_regressor.predict(X)
        return median, interval.reshape(-1,2)
     

    return conformal_predictor 
