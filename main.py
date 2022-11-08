import argparse
import configparser
import os
import sys

version = 1.0


def show_version():
    print(version)


def directory_tree():
    os.mkdir("./dataset")
    os.mkdir("./cleanData")
    os.mkdir("./script")

def show_profile():
    


parser = argparse.ArgumentParser(description="AI training tool")

# AI management tool use sub commands
sub_parser = parser.add_subparsers(help="manage the AI training")

# p1 command show the version of the tool
p1 = sub_parser.add_parser("version")
p1.set_defaults(func=show_version)

# p2 command start the project
p2 = sub_parser.add_parser("createProject")
p2.set_defaults(func=directory_tree)

# p3 show the data in origin dataset
p3 = sub_parser.add_parser("show")
p3.add_argument("-l", "--length", help="length of the data", default=5,type=int)

parser.parse_args()
args = parser.parse_args()
args.func()
