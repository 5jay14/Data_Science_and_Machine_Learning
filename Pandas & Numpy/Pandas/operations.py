import pandas as pd
import numpy as np

df = pd.DataFrame({'col1': [1, 2, 3, 4],
                   'col2': [444, 555, 666, 444],
                   'col3': ['abc', 'def', 'ghi', 'xyx']})

# Finding unique values in DF

print(df['col2'].unique())  # finding unique values in a given column, lists all the unique values
print(df['col2'].nunique())  # finding the total number of unique values in a given column
print(df['col2'].value_counts())  # This shows how many time each unique element occurred
print(df[df['col1'] > 2])


# applying custom functions to Pandas DF
def times2(x):
    return x * 2


print(df['col1'].apply(times2))  # this applies the custom function to each element in the data frame
print(df['col1'].apply(lambda x: x*2))  # lambda function
print(df['col3'].apply(len))  # applying a built-in function
print(df.columns) #this will throw an index object with the list of column names
print(df.index) #details about index


#Sorting
a = df.sort_values(by= 'col2')
print(a) #note how the index stays with element

#finding null values
df.isnull()  # this returns a df with bool values