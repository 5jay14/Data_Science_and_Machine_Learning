import pandas as pd

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\04"
                  r"-Pandas-Exercises\Salaries.csv")
print(df.head())

# LOC is lable based, we have to specify rows and columns based on their index/row and column lables.
# ILOC is integer position based, it starts with 0. We have to specify rows and columns based on the integer position values
# Index label is the name we have given, index postion is the integer position behind the lable
print(df.head())
