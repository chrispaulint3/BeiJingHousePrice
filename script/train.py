"""
train AI model
"""
import pandas as pd
import settings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBRegressor as XGBR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np


def dataset_process():
    """
    split the dataset and one hot encoding
    23 features, 7 features one hot encode

    :return:
    """
    df = pd.read_csv(settings.OUT_FILE["clean_file"])
    df = df.drop(["index"], axis=1)


    # 1. encode the the features with one hot encoder
    # one hot encode the features
    encoded_feature = ["elevator", "buildingType", "subway", "district", "floor_type","renovationCondition",
                       "buildingStructure","fiveYearsProperty"]
    df = pd.get_dummies(df, columns=encoded_feature)

    # # 2. features min_max_scaler
    df["tradeTime"] = pd.to_datetime(df["tradeTime"]).dt.year
    print(df["tradeTime"])
    scale_features = ["Lng", "Lat", "followers", "price", "square", "livingRoom", "drawingRoom", "kitchen",
                      "bathRoom", "constructionTime", "ladderRatio", "communityAverage", "numeric_floor","center_r"]
    min_max_scaler = preprocessing.MinMaxScaler()
    df[scale_features] = min_max_scaler.fit_transform(df[scale_features])
    df["tradeTime"] = min_max_scaler.fit_transform(df["tradeTime"].to_numpy().reshape(-1, 1))

    print(df.info())
    target = df[settings.TARGET]
    feature = df.drop(settings.TARGET, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=1)
    print(x_train)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    print(mse)
    print(mae)
    print(lr.score(x_test,y_test))


def train():
    x_train = np.array([[1, 2],
                        [2, 4],
                        [3, 6]])
    x_test = np.array([[4, 8],
                       [6, 10]])
    y_train = np.array([3, 6, 9])
    y_test = np.array([12, 16])
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    print(mse)
    print(y_predict)


if __name__ == "__main__":
    dataset_process()
