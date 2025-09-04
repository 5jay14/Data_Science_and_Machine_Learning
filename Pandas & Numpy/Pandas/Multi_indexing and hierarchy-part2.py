import numpy as np
import pandas as pd
import random

# Multi level indexing is like merging the first column and creating DF for each merged column
# index levels

outside = 'G1 G1 G1 G2 G2 G2'.split()
inside = [1, 2, 3, 1, 2, 3]
hierarchy_index = list(zip(outside, inside))
hierarchy_index = pd.MultiIndex.from_tuples(hierarchy_index)

arr = np.arange(12).reshape(6, 2)
DF = pd.DataFrame(arr, hierarchy_index, ['C1', 'C2'])

# adding a name to outside and inside index
DF.index.names = ['Out', 'In']
print(DF)

# Extracting the inside index from outside index
print(DF.loc['G1'].loc[1])  # this will be a single series
print(DF.loc['G2'].loc[1].loc['C1'])  # Calling a specific value
print(DF.loc['G1'].loc[2].loc['C1'])  # ex2: Calling a specific value

# cross-section = This returns a cross-section data. we can extract the data from inside index without putting filter
# on the external index

DF_xs = DF.xs(2, level='In')
print(DF_xs)