import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Ames_NO_Missing_Data.csv")
print(df.info())

#with open(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Ames_Housing_Feature_Description.txt") as description:
    #print(description.read())

# There would be feature represented in numbers but they are categorical. There will be no clear ordinal relationship
# between the numbers. Example MSSubClass

# converting the integer into string/object first
df['MS SubClass'] = df['MS SubClass'].apply(str)
string_objects = df.select_dtypes(include='object') # this will pull just the selected objects, essentially a string
numeric_objects = df.select_dtypes(exclude='object') # selecting everything but object dtypes

# creating dummy variables for objects
dummy_object_variables = pd.get_dummies(string_objects, drop_first=True)

# concatenating the object dummy variables with numeric features
final_df = pd.concat([dummy_object_variables,numeric_objects],axis=1)

# we also need to keep in mind the human aspect that came with the dataset ex overall condition is correlated with
# sale price. The human judges the overall quality and we are not sure how it is derived

print(final_df.corr()['SalePrice'].sort_values())

