"""
configure the file path and global parameters
"""
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

"""
drop the useless feature and specify the target
"""
UND_LIST = ["DOM", "url", "id", "Cid"]
TARGET = ["totalPrice"]

"""
specify the file out put name
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FILE = {"file_1": os.path.join(BASE_DIR, "dataset", "housePrice.csv")}
OUT_FILE = {"clean_file": os.path.join(BASE_DIR, "cleanData", "clean.csv"),
            "model_file": os.path.join(BASE_DIR, "model", "model.joblib"),
            "x_train_file":os.path.join(BASE_DIR,"dataset","x_train.csv"),
            "x_test_file":os.path.join(BASE_DIR,"dataset","x_test.csv"),
            "y_train_file":os.path.join(BASE_DIR,"dataset","y_train.csv"),
            "y_test_file":os.path.join(BASE_DIR,"dataset","y_test.csv")}


"""
model list where manage the scikit learn model configuration
"""
MODEL = {"linear": LinearRegression(), "lasso": Lasso(), "xgboost": XGBRegressor(),
         "randomForest": RandomForestRegressor()}

"""
hyper parameter
"""
PARAMETER = {"ridge": [], "lasso": [], "xgboost": [{"max_depth": [3, 5, 6, 7, 9],
                                                    "min_child_weight": [1, 3, 5, 7],
                                                    "gamma": [0.1, 0.7, 1],
                                                    "subsample": [0.5, 0.8, 1]}]}
