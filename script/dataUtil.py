import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import settings

path = "../dataset/housePrice.csv"
__all__ = ["clean_feature"]

def corr_analyse():
    """
    correlation analyse
    :return:
    """
    df = pd.read_csv(path)
    df.drop(settings.UND_LIST, axis=1, inplace=True)

    # draw the correlation series extract from correlation matrix
    corr_data = df.corr()[settings.TARGET].sort_values(by=settings.TARGET, ascending=False)
    print(corr_data)
    plt.figure(figsize=(5, 10))
    sns.heatmap(corr_data, cmap="rainbow", annot=True)
    plt.show()


def clean_data():
    """
    drop the useless feature
    clean the data,handle the missing value
    :return:
    """
    df = pd.read_csv(path)

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
    df = clean_data()

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
    df["center_r"] = ((df["Lng"]-116.402544)**2+(df["Lat"]-39.915599)**2)**0.5
    df.to_csv(settings.OUT_FILE,encoding="utf-8",index_label="index")
    return df


def show_map():
    """
    draw the house position on the map
    :return:
    """
    df = clean_feature()
    sns.scatterplot(x="Lng", y="Lat", hue="price",palette="viridis_r", data=df)
    plt.scatter(x=[116.402544],y=[39.915599],sizes=[120],color="red")
    plt.show()



