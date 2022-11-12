"""
configure the file path and global parameters
"""
import os

"""
drop the useless feature and specify the target
"""
UND_LIST = ["DOM", "url", "id", "totalPrice"]
TARGET = ["price"]


"""
specify the file out put name
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = {"clean_file":os.path.join(BASE_DIR,"cleanData","clean.csv")}
