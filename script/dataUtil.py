import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

import settings
from sklearn import preprocessing


def corr_analyse():
    """
    correlation analyse
    :return:
    """
    df = pd.read_csv(settings.ORIGIN_FILE["file_1"])
    df.drop(settings.UND_LIST, axis=1, inplace=True)

    # draw the correlation series extract from correlation matrix
    corr_data = df.corr()[settings.TARGET].sort_values(by=settings.TARGET, ascending=False)
    print(corr_data)
    plt.figure(figsize=(5, 10))
    sns.heatmap(corr_data, cmap="rainbow", annot=True)
    plt.show()


def _clean_data():
    """
    drop the useless feature
    clean the data,handle the missing value
    :return:
    """
    df = pd.read_csv(settings.ORIGIN_FILE["file_1"], low_memory=False)

    # 1. drop the column in conf file
    df = df.drop(settings.UND_LIST, axis=1)

    # 2. drop the NA value
    df = df.dropna()
    df["tradeTime"] = pd.to_datetime(df["tradeTime"])

    # 3. drop the transaction before the year 2013
    df = df[df["tradeTime"].dt.year >= 2013]
    return df


def clean_feature():
    """
    process the feature to fit the AI model
    :return: processed data
    """
    df = _clean_data()

    # 1. convert the object to int to reduce memory use
    df["livingRoom"] = df["livingRoom"].astype(int)
    df["drawingRoom"] = df["drawingRoom"].astype(int)
    df["kitchen"] = df["kitchen"].astype(int)

    # 2. split the feature floor into two features
    split_array = df["floor"].str.split(" ", expand=True)
    split_array.columns = ["floor_type", "numeric_floor"]
    # map the content to number
    map_dic = {"高": 4, "中": 3, "低": 2, "底": 1, "顶": 5, "未知": 0}
    split_array["floor_type"] = split_array["floor_type"].map(map_dic)
    # replace the floor feature in dataset
    df = df.drop(["floor"], axis=1)
    df["floor_type"] = split_array["floor_type"]
    df["numeric_floor"] = split_array["numeric_floor"].astype(int)

    # 4. drop the feature construction time with missing value
    # convert features to int
    df = df[df["constructionTime"] != '未知']
    df["bathRoom"] = df["bathRoom"].astype(int)
    df["constructionTime"] = df["constructionTime"].astype(int)

    # 5. The longitude and the latitude should combine to make prediction
    # take the center position (116.402544,39.915599)
    df["center_r"] = ((df["Lng"] - 116.402544) ** 2 + (df["Lat"] - 39.915599) ** 2) ** 0.5
    df.to_csv(settings.OUT_FILE["clean_file"], encoding="utf-8", index_label="index")
    return df


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


# split train test data
def split_data(df, save=True):
    """
    :param df: the preprocessed data to split
    :param save: save the train set and test set in dataset dir
    :return:
    """
    target = df[settings.TARGET]
    feature = df.drop(settings.TARGET, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)
    if save:
        x_train.to_csv(settings.OUT_FILE["x_train_file"], index=False)
        x_test.to_csv(settings.OUT_FILE["x_test_file"], index=False)
        y_train.to_csv(settings.OUT_FILE["y_train_file"], index=False)
        y_test.to_csv(settings.OUT_FILE["y_test_file"], index=False)
    return x_train, x_test, y_train, y_test




def show_map():
    """
    draw the house position on the map
    :return:
    """
    df = clean_feature()
    sns.scatterplot(x="Lng", y="Lat", hue="price", palette="viridis_r", data=df)
    plt.scatter(x=[116.402544], y=[39.915599], sizes=[120], color="red")
    plt.show()

if __name__ == "__main__":
    df = dataset_process()
    split_data(df)
