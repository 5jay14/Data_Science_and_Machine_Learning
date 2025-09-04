import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\wholesome_customers_data.csv")
# print(df.info())
# print(len(df.columns))

# print(df.head())

sns.scatterplot(df, x='Fresh', y='Frozen', hue='Region')
plt.show()

# Use seaborn to create a histogram of MILK spending, colored by Channel. Can you figure out how to use seaborn to
# "stack" the channels, instead of have them overlap?

sns.histplot(df, x='Milk', hue='Channel', multiple='stack', palette='Set1')
plt.show()

sns.clustermap(df.drop(['Region', 'Channel'], axis=1).corr(), annot=True, row_cluster=False)
plt.show()

# df[['Fresh','Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen', 'Channel','Region']]
# sns.pairplot(data=df, hue='Region', palette='Set1')
# plt.show()

# DBSCAN

# Scaling is appropriate here, although the features are in different magnitude but they are in the same metric(dollars)
# there are two categorical columns but the variables are less than 3, Scale them too

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_x = scaler.fit_transform(df)

# Use DBSCAN and a for loop to create a variety of models testing different epsilon values. Set min_samples equal
# to 2 times the number of features. During the loop, keep track of and log the percentage of points that are outliers.
# For reference the solutions notebooks uses the following range of epsilon values for testing: np.linspace(0.001,3,50)

from sklearn.cluster import DBSCAN

Outliers_percentage = []
Outliers_count = []

for e in np.linspace(0.001, 3, 50):
    model = DBSCAN(eps=e, min_samples=scaled_x.shape[1] * 2)
    model_categories = model.fit(scaled_x)

    outliers = np.sum(model.labels_ == -1)
    Outliers_count.append(outliers)
    Outliers_percentage.append((outliers / len(model.labels_) * 100))

# Create a line plot of the percentage of outlier points versus the epsilon value choice

sns.lineplot(data=scaled_x, y=Outliers_percentage, x=np.linspace(0.001, 3, 50))
plt.ylabel('Outlier %')
plt.xlabel('Eps')
plt.show()

# EPS = 2 seems to be showing good result
model2 = DBSCAN(eps=2, min_samples=scaled_x.shape[1] * 2)
model2.fit(scaled_x)

df['Labels'] = model2.labels_  # since the train test split is used, everything is set same on the index (not shuffled)
print(df.head())

# Now we can visualize the predicted categroy on the original data, like scatter plot
# Create a scatterplot of Milk vs Grocery, colored by the discovered labels of the DBSCAN model.

sns.scatterplot(df, x='Milk', y='Grocery', hue='Labels', palette='Set1')
plt.show()

# Compare the statistical mean of the clusters and outliers for the spending amounts on the categories

cat_var = df.drop(['Channel', 'Region'], axis=1)
cat_means = cat_var.groupby('Labels').mean()
print(cat_means)

sns.heatmap(cat_means, annot=True)
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_cat_data = scaler.fit_transform(cat_means)

scaled_df = pd.DataFrame(scaled_cat_data, cat_means.index, cat_means.columns)
sns.heatmap(scaled_df)
plt.show()

# Create another heatmap similar to the one above, but with the outliers removed

sns.heatmap(scaled_df.loc[[0, 1]], annot=True)
plt.show()
