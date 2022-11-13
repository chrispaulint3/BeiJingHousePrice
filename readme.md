# AI data analyse
predicting house price in Beijing. 
* Dataset: [Housing price in Beijing](https://www.kaggle.com/datasets/ruiqurm/lianjia)
* AI model: linear regression, random forest...
## suggestions
Thanks to the open source python AI community like pytorch, scikit-learn, pandas etc.,
the artificial intelligence application development is much easier and faster.
But the AI development workflow and user interface are diverse, the readability
and maintainability are tricky problems. I have seen some ingenious programs on
git hub, but some are hard to learn and duplicate and need months of work to 
rebuild the AI system. So I have some suggestions below:
* global settings or hyper-parameter put in one directory,for better management.
* store the intermediate in one directory, for researchers check the result the
data preprocessing and training the model directly.
* script about dataset io, data preprocessing, model training put in script.

## project structure
```txt
ai:.
│  .gitignore
│  main.py // model train command in this file
│  readme.md
│  
│          
├─cleanData   // ai training use clean data in this directory
│      clean.csv
│      
├─conf
│  │  settings.py // configuration for ai training
│  │  
│  └─__pycache__
│          settings.cpython-39.pyc
│          
├─dataset // store orignal dataset store
│      housePrice.csv
│      
└─script  // ai data clean and model training script 
        dataUtil.py
        train.py
```
you can change the workflow of ai training in script dir.
## command reference
```shell
# create project tree in current directory
python manage.py createProject 
# clean the dataset
python manage.py clean 
# train the models,-s means save the model
python manage.py train -s 
# train the model and predict on the test set
# -l means load the model in model dir
python manage.py predict -l 
```
## notes
## settings configuration
```python
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
```
