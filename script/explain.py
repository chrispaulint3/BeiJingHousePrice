from interpret.blackbox import LimeTabular
from joblib import load
from dataUtil import dataset_process,split_data
import settings


def explain(args):
    # load the model from file
    df = dataset_process()
    x_train,x_test,y_train,y_test = split_data(df)
    ai_model = load(settings.OUT_FILE["model_file"])
    if args.model == "lime":
        lime = LimeTabular(predict_fn=ai_model.predict_proba,data=x_train)
