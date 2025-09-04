# Each series sharing a common index. From the below example.
# Column A B and C is a pandas series sharing common index x y and z
import random
from random import randint

import numpy as np
import pandas as pd
from numpy.random import randn

#               A   B   C
#          X    1   2   3
#          Y    4   5   6
#          Z    7   8   9

np.random.seed(101)  # this will be same values everywhere
df = pd.DataFrame(np.arange(1,10).reshape(3,3), ['x', 'y', 'z'], ['A', 'B', 'C'])
print(df)
print('')


# knowing the names of the index
print(df.index.names)


# adding the names to the index
df.index.names = ['C']

# grabbing the series/column data
print(df['A'])
print(df[['A', 'B']])  # Notice additional square bracket for multiple series/columns

# Selecting rows. Notice there is no bracket, it is square for this inbuilt method.
# notice the rows are passed as series as well

# first way, We need to pass the row label as index
print(df.loc['x']),

# 2nd Way, numerical index based location. x=0,y=1, z=2
print(df.iloc[0])

# subset, extracting single or multiple specific values
print(df.loc['x', 'B'])  # 1st is index, 2nd is column
print(df.loc[['x', 'y'], ['A', 'B']])  # a bit different , first argument takes the rows, second takes the columns

matrix = np.arange(0, 25).reshape(5, 5)
print(matrix)

pdDf = pd.DataFrame(matrix, ['R1', 'R2', 'R3', 'R4', 'R5'], ['C1', 'C2', 'C3', 'C4', 'C5'])
print(pdDf)
print(pdDf.loc['R5','C4'])

# adding/removing columns or rows

# adding
pdDf['C6'] = pdDf['C4']+pdDf['C5']

# removing, this will not make changes to the original dataframe, we need to specify by inplace to true
# This is done to not lose the data accidentally
pdDf.drop('C6', axis=1,inplace=False)
print(pdDf)

