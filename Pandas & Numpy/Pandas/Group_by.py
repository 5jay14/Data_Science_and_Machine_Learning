import numpy as np
import pandas as pd

# Group by to perform some aggregate functions

dicta = {'Company': ['FB', 'FB', 'Goog', 'Goog', 'MSFT', 'MSFT'],  # 'Person': ['JJ', 'VJ', 'AJ', 'DJ', 'MJ', 'CJ'],
         'Sales': [100, 200, 300, 400, 500, 800]}

DF = pd.DataFrame(dicta)
print(DF)

by_comp = DF.groupby('Company')
# all in one line
by_comp = DF.groupby('Company')
print(by_comp.mean())
print(by_comp.sum())
print(by_comp.count())
print(by_comp.max())
print(by_comp.min())
# This creates an object and displays where it is stored.
# Perform aggregate function to see the data
# Pandas will automatically ignore if its feels a non-numeric columns
# Other useful methods

print(by_comp.describe())  # This gives most of the information about a dataset
print(by_comp.describe().transpose()['FB'])
