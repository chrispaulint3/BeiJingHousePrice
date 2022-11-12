"""
train AI model
"""
import pandas as pd
import settings
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np

path = "../cleanData/clean.csv"


def dataset_process():
    """
    split the dataset and one hot encoding
    :return:
    """
    df = pd.read_csv(settings.OUT_FILE["clean_file"])
    df.rename(columns={"Unnamed:0":"ID"},inplace=True)
    print(df.info())
    df["tradeTime"] = pd.to_datetime(df["tradeTime"])
    target = df["price"]
    feature = df.drop(["price"], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.4, random_state=1)

    # 2. encode the the features with one hot encoder
    # one hot encode the features
    encoded_feature = ["elevator", "buildingType", "subway", "district", "floor_type"]
    encode_result = pd.get_dummies(df, columns=encoded_feature)
    df = pd.concat([df,encode_result])
    df = df.drop(encoded_feature,axis=1)



    # min_max_scaler = preprocessing.MinMaxScaler()
    # df[] = min_max_scaler.fit_transform(df["tradeTime"].to_numpy().reshape(-1,1))
    # print(a)
    # print(x_train.shape)
    # print(x_test.shape)
    pass


if __name__ == "__main__":
    dataset_process()
