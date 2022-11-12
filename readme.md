# ai data analyse
* dataset
* data preprocessor
* AI model
* command line reference

## porject structure
```buildoutcfg
ai:.
│  .gitignore
│  main.py // model train command in this file
│  readme.md
│  tree.txt
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
