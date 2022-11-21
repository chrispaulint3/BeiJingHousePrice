"""
configure the file path and global parameters
"""
import os
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

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
OUT_FILE = {"clean_file": os.path.join(BASE_DIR, "cleanData", "clean.csv")}

"""
model list where manage the scikit learn model configration
"""
MODEL = {"linear": {LinearRegression()}, "xgboost": XGBRegressor()}

"""
hyper parameter
"""
PARAMETER = {"ridge": [], "lasso": [], "xgboost": [{"max_depth": [3, 5, 6, 7, 9],
                                                    "min_child_weight": [1, 3, 5, 7],
                                                    "gamma": [0.1, 0.7, 1],
                                                    "subsample": [0.5, 0.8, 1]}]}
