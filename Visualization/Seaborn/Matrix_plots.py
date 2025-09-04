'''
Heat maps = primary way of showing matrix plots
IMP --- It is necessary to have the data in matrix form to work matrix plots
index and column name should match to that the value of the cell that indicates relevant to these two
tips data set does not have actual row variable. it is given automatically
we can have the same column and row name by many method
1.use corr() to have same row and column names. getting error in converting the data types
2. use pivot_table
'''

import seaborn as sns
import matplotlib.pyplot as plt

# Heat Maps
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
print(flights.head())
fl = flights.pivot_table(index='month', columns='year', values='passengers')
sns.heatmap(fl, cmap='coolwarm')  # cmap is color map
# sns.heatmap(fl,cmap='coolwarm', linecolor='white',linewidths=1) #cmap is color map


'''
Cluster map - it tries to cluster columns and rows together based off their similarity
Observe, x and y are not in order
'''

sns.clustermap(fl, cmap='coolwarm')
# can also pass the scale, scale =1
plt.show()