"""
cli commands are defined here
"""


import argparse
import os
from script.dataUtil import clean_feature

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
        print("create script dir successfully")
    #
    if not os.path.exists("./script/__init__.py"):
        with open("./script/__init__.py","w") as f:
            pass



def show_profile():
    pass


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

parser.parse_args()
args = parser.parse_args()
args.func()
