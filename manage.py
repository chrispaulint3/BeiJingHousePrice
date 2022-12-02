"""
cli commands are defined here
"""
import argparse
import os
import sys
from script.dataUtil import clean_feature
from script.train import linear_regression
from script.train import hyper_parameters

version = 1.0


def show_version():
    print(version)


# create the project tree in the current dir
def directory_tree():
    # store the dataset
    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
        print("create dataset dir successfully")
    # store the intermediate result
    if not os.path.exists("./cleanData"):
        os.mkdir("./cleanData")
        print("create cleanData dir successfully")
    # store the ai train script
    if not os.path.exists("./script"):
        os.mkdir("./script")
        with open("./script/__init__.py", "w") as f:
            pass
        print("create script dir successfully")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI training tool")

    # AI management tool use sub commands
    sub_parser = parser.add_subparsers(help="manage the AI training")

    # p1 command show the version of the tool
    p1 = sub_parser.add_parser("version")
    p1.set_defaults(func=show_version)

    # p2 command start the project
    p2 = sub_parser.add_parser("createProject")
    p2.set_defaults(func=directory_tree)

    # p3 clean the dataset
    p3 = sub_parser.add_parser("clean")
    p3.set_defaults(func=clean_feature)

    # p4 train the model
    p4 = sub_parser.add_parser("train")
    p4.add_argument("-m", "--model", help="train model linear,lasso,ridge,XGboost")
    p4.add_argument("-s", "--save", action="store_true",help="train the model in")
    p4.set_defaults(func=linear_regression)

    # p5 search for the best parameters
    p5 = sub_parser.add_parser("search")
    p5.add_argument("-m", "--model", help=" model parameter linear/lasso/ridge/XGboost")
    p5.set_defaults(func=hyper_parameters)

    parser.parse_args()
    args = parser.parse_args()
    args.func(args)
