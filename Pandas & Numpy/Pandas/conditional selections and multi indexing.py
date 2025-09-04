import numpy as np
import pandas as pd

# Just like numpy, we can select data by supplying conditions

matrix = np.arange(0, 25).reshape(5, 5)
DF = pd.DataFrame(matrix, ['R1', 'R2', 'R3', 'R4', 'R5'], ['C1', 'C2', 'C3', 'C4', 'C5'])

print(DF > 5)  # This returns a true or false list based on the condition supplied

bool_df = DF > 5
print(DF[bool_df])

print(DF['C1'] > 2)  # this returns true and false
print(DF[DF['C1'] > 2])  # This will only return the rows and columns where the condition is met

# i can further filter out using additional conditions
print(DF[DF['C1'] > 2][['C2', 'C3']])

# multiple conditions. We would need to use & | instead of 'and' 'or'. The reason is 'and' 'or' can work on df if
# the resultant is single boolean value

# print(DF[(DF['C1'] > 2) and (DF['C1'] > 2)]) # this will throw error
a1 = (DF[(DF['C1'] > 2) & (DF['C2'] > 10)])
print(f'this is double filtered {a1}')
print(DF[(DF['C1'] > 5) | (DF['C2'] < 5)])

# Replacing the index to integer values from the custom ones. We have to explicitly mention to make changes to original
# DF. Original index will not be added as a new columns
print(DF.reset_index())

# adding new columns/series
newInd = 'KA TN AP TS KL'.split()
DF['States'] = newInd
print(DF)

# making a column as index, need to explicitly mention to save. Cannot reset once committed
print(DF.set_index('States'))
