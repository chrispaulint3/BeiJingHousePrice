import pandas as pd

path = "../dataset/HR Employee Attrition.csv"
pd.set_option("display.max_columns", None)
df = pd.read_csv(path)
print(df.columns)
print(df.describe())
