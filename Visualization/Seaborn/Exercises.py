import seaborn as sns
import matplotlib.pyplot as plt


titanic = sns.load_dataset('titanic')
print(titanic.head())

sns.jointplot(titanic,x='fare', y='age')
plt.show()

sns.distplot(titanic['fare'],bins = 30, kde=False,color='red')
plt.show()

sns.set_style('ticks')
sns.boxplot(titanic,x='class', y ='age')
plt.show()

sns.countplot(titanic,x='sex')
plt.show()

sns.swarmplot(titanic,x='class', y='age')
plt.show()

#sns.heatmap(titanic.corr(), cmap='coolwarm')
#plt.title('titanic')
#plt.show()

g = sns.FacetGrid(titanic, col='sex')
h = sns.FacetGrid(titanic, col='sex')
g.map(sns.histplot,'age')
h.map(sns.distplot,'age')


plt.show()