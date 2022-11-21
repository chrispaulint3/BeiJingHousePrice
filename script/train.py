"""
train AI model
"""
import sys

import pandas as pd
import settings
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import time
from script.commonUtil import performance
from sklearn.pipeline import Pipeline


def _dataset_process():
    """
    split the dataset and one hot encoding
    23 features, 7 features one hot encode
    :return:
    """
    df = pd.read_csv(settings.OUT_FILE["clean_file"])
    df = df.drop(["index"], axis=1)

    # 1. encode the the features with one hot encoder
    # one hot encode the features
    encoded_feature = ["elevator", "buildingType", "subway", "district", "floor_type", "renovationCondition",
                       "buildingStructure", "fiveYearsProperty"]
    df = pd.get_dummies(df, columns=encoded_feature)

    # # 2. features min_max_scaler
    df["tradeTime"] = pd.to_datetime(df["tradeTime"]).dt.year
    scale_features = ["Lng", "Lat", "followers", "price", "square", "livingRoom", "drawingRoom", "kitchen",
                      "bathRoom", "constructionTime", "ladderRatio", "communityAverage", "numeric_floor", "center_r"]
    min_max_scaler = preprocessing.MinMaxScaler()
    df[scale_features] = min_max_scaler.fit_transform(df[scale_features])
    df["tradeTime"] = min_max_scaler.fit_transform(df["tradeTime"].to_numpy().reshape(-1, 1))
    return df


def linear_regression(args):
    """
    train the regression model
    :param args: model types,string
    :return:
    """
    # preprocess the data,split the dataset
    df = _dataset_process()
    target = df[settings.TARGET]
    feature = df.drop(settings.TARGET, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)

    if args.model in settings.MODEL:
        start = time.time()
        model = settings.MODEL[args.model]
        model.fit(x_train, y_train)
        time_consuming = time.time() - start
        performance(model, x_test, y_test, time_consuming)
    else:
        print("do not support the model type. please retry linear/lasso/ridge/xgboost")

    if args.save:
        dump(model, "../model/model.joblib")


def hyper_parameters(args):
    df = _dataset_process()
    target = df[settings.TARGET]
    feature = df.drop(settings.TARGET, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)

    gs = GridSearchCV(settings.MODEL[args.model],param_grid=settings.PARAMETER[args.model])
    gs.fit(x_train,y_train)
    model = gs.best_estimator_
    print(model.score(x_test,y_test))


