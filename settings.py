"""
configure the file path and global parameters
"""
import os

"""
drop the useless feature and specify the target
"""
UND_LIST = ["DOM", "url", "id","Cid"]
TARGET = ["totalPrice"]


"""
specify the file out put name
"""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGIN_FILE = {"file_1":os.path.join(BASE_DIR,"dataset","housePrice.csv")}
OUT_FILE = {"clean_file":os.path.join(BASE_DIR,"cleanData","clean.csv")}
