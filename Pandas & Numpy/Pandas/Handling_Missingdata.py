import numpy as np
import pandas as pd

# Dealing with missing datas in padas: if we have a missing values, pandas will return NaN or null value

# We can use functions to fill or drop the values

a = {'A': [0, 1, 2], 'B': [4, np.nan, 5], 'C': [7, np.nan, np.nan]}
df = pd.DataFrame(a)
print(df)
print('')

# using .dropna method we can drop any row or column with one or more null values. axis 0 drops rows
# Specify how many NaN value are required to get dropped

print(df.dropna(axis=0))
print(df.dropna(axis=1, thresh=2))

# fill values
print(df.fillna(value='example'))
print(df['B'].fillna(value=df['B'].mean()))
