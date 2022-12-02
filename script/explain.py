import pandas as pd
from joblib import load
from dataUtil import dataset_process, split_data
from interpret import show
import settings
import time
from lime.lime_tabular import LimeTabularExplainer
import lime


def explain():
    # load the data from csv file
    x_train = pd.read_csv(settings.OUT_FILE["x_train_file"])
    x_test = pd.read_csv(settings.OUT_FILE["x_test_file"])
    y_train = pd.read_csv(settings.OUT_FILE["y_train_file"])
    y_test = pd.read_csv(settings.OUT_FILE["y_test_file"])
    x_train_lime = pd.concat([x_train, y_train], axis=1)
    # load the model from file
    ai_model = load(settings.OUT_FILE["model_file"])
    # if args.model == "lime":
    explainer = LimeTabularExplainer(x_train, feature_names=x_train_lime.columns, verbose=True,
                                     class_names=["totalPrice"], mode="regression")
    exp = explainer.explain_instance(x_test[0], ai_model.predict, num_features=5)

    time.sleep(10000)


if __name__ == "__main__":
    explain()
