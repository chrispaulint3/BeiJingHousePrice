"""
train AI model
"""
import settings
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.model_selection import GridSearchCV
from script.commonUtil import performance
from dataUtil import dataset_process,split_data
import time






def linear_regression(args):
    """
    train the regression model
    :param args: model types,string
    :return:
    """
    # preprocess the data,split the dataset


    if args.model in settings.MODEL:
        df = dataset_process()
        x_train,x_test,y_train,y_test = split_data(df)
        start = time.time()
        model = settings.MODEL[args.model]
        model.fit(x_train, y_train)
        time_consuming = time.time() - start
        performance(model, args.model, x_test, y_test, time_consuming)
    else:
        print("do not support the model type. please retry linear/lasso/ridge/xgboost")

    if args.save:
        dump(model, settings.OUT_FILE["model_file"])
        print("the model is saved, check the model in model directory")


def hyper_parameters(args):
    df = dataset_process()
    x_train, x_test, y_train, y_test = split_data(df)

    gs = GridSearchCV(settings.MODEL[args.model], param_grid=settings.PARAMETER[args.model])
    gs.fit(x_train, y_train)
    model = gs.best_estimator_
    performance(model, "XGBoost", x_test, y_test, 10000)
