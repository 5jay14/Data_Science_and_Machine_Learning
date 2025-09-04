import matplotlib.pyplot as plt
import seaborn as sns

'''
Grids= Automate subplots based of off features from the data
instead of pairplot use PairGrid
pp is the simplified version of pairgrid
pairgrid gives more control

'''

# load a dataset
iris = sns.load_dataset('iris')
print(iris.head())

# create a pair grid and it assing it a variable
a = sns.PairGrid(iris)  # this will just give the empty pair grids

# map the plot types
a.map(plt.scatter)

# if we need to specify the plot types for placeholder like diagonal, upper or lower half
# a.map_lower(sns.distplot)


# FacetGrid grid

tips = sns.load_dataset('tips')
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.heatmap(tips))
plt.show()
