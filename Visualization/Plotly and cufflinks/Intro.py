# Plotly is a interactive open source visualization library
# Cufflink connects plotly with Pandas, is the library used to call plots
# install both,

import matplotlib.pyplot as plt
import matplotlib_inline

import pandas as pd
import numpy as np
import cufflinks as cf
from plotly.offline import init_notebook_mode, iplot, plot, download_plotlyjs

# Set up the offline mode for Plotly and connect it to Cufflinks
cf.go_offline()
init_notebook_mode(connected=True)

# Create a DF
df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
df1 = pd.DataFrame(np.random.randn(100, 4), columns=['A', 'B', 'C', 'D'])
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [7, 8, 15]})

# Plot the DataFrame using iplot()
df.iplot()
df2.iplot(kind='scatter', x='Category', y='Values', mode='markers', xTitle='Category', yTitle='Values')  # size = 10
df2.iplot(kind='bar', x='Category', y='Values')

# calling barplots on non categorical dataframes : use some kind of aggregate or group by functions
df1.count().iplot(kind='bar')

# box plots: aggregation done automatically, we can select the dataset that we want in the graph itself
df.iplot(kind='box')

# 3d surface plot
df3 = pd.DataFrame({'x': [1, 2, 3, 4, 5], 'y': [10, 20, 30, 40, 50], 'z': [100, 200, 300, 400, 500]})
df3.iplot(kind='surface')

# histogram
df['A'].iplot(kind='hist', bins=30)

# overlapping histogram, gives the option to view all the column data and we can switch which to view
df.iplot(kind='hist')

# spread, used in stock marketing. we get one line plot and spread subplot
# line and spread plot = it shows the spread against each other

df[['A', 'B']].iplot(kind='spread')
# Bubble = similar to scatter, except it will change the size of the points based off another variable
df.iplot(kind = 'bubble',x='A',y='B', size='C')


# scatter matrix similar to seaborn pairplots. All the columns needs to be numerical for this to work
df.scatter_matrix()