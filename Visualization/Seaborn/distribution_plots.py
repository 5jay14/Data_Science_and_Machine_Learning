import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Seaborn comes with built in DF which can be used for examples
tips = sns.load_dataset('tips')
pd.option_context('mode.use_inf_as_na', True)


''' 
Dist/distribution plot = This type of plot allows us to show just 1 variable / univariable
This gives a histogram with KDE, kernel density estimation which is the line that goes through the plots
We can also remove by passing additional arguments as KDE = False
KDE = Line through the histogram, its the sum of all the normal point along the rug plot
Bins = it will show much more data, kinda of like zooming out to view more data. This depends on the size of the
Dataframe
'''
sns.distplot(tips['total_bill'], kde=False, bins=100)  # Ignore the error, this has to do with another package
plt.show()

'''
Joint Plot = We can combine two different distribution plots with variants(two variables)
We need to pass x, y and dataset
It takes additional argument called kind, where we can pass  'hex', 'kde','reg', the default is scatter
'''

sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')
plt.show()

'''
Pair plot = This allows to draw comparison/ relation between all the numerical variables of the dataset
As additional arguments, we can pass
hue('') = this categories the dataset, meaning if we look at the column 'sex', there are men and women.this color 
codes the gender and gives the legends
palette('')

when compared against the own column, it shows histogram
'''
sns.pairplot(tips,hue='sex')
plt.show()

'''
rugplot = it draws dash mark for every point along the distribution plot
it has relation to distribution plot, 
'''

sns.rugplot(tips['total_bill'], kde=False)
plt.show()

#kdeplots
'''
This allows just the KDE not the histograms
'''

sns.kdeplot(tips['total_bill'])
plt.show()