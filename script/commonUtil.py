from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def performance(model, x_test, y_test,time_consuming,verbose=True):
    """
    compute the performance of the regression model and redirect the performance into log file
    :param model: model trained
    :param x_test: test_set features
    :param y_test: test_set target
    :param time_consuming: the time the script
    :return:
    """
    y_predict = model.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    mae = mean_absolute_error(y_test, y_predict)
    r2 = r2_score(y_test, y_predict)

    # redirect the performance to the log file
    with open("../log/log.txt","a") as f:
        print("task complete at {date_time}".format(date_time=datetime.now()),file=f)
        print("time-consuming: {time_consuming}".format(time_consuming=time_consuming),file=f)
        print("mse:{mse} mae:{mae} r2:{r2}".format(mse=mse, mae=mae, r2=r2),file=f)
        print("-----------------------------------------",file=f)

    if verbose:
        print("task complete at {date_time}".format(date_time=datetime.now()))
        print("time consuming: {time_consuming}".format(time_consuming=time_consuming))
        print("mse:{mse} mae:{mae} r2:{r2}".format(mse=mse, mae=mae, r2=r2))
        print("-----------------------------------------")
