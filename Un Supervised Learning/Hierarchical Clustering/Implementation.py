import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\mpg.csv")
# df['horsepower'] = df['horsepower'].astype(int)
# df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# print(df.head(), df.describe())


df_dummies = pd.get_dummies(df.drop('name', axis=1))
# Name is unique which is not useful in the unsupervised cluserting

# Using MinMax Scaler instead of standard sclaer, which works based on the euclidean metric which scales between 0 and 1

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df_dummies)  # all the data distance between 0 and 1, they are in numpy array
scled_df = pd.DataFrame(scaled_data, columns=df_dummies.columns)  # Converting back to dataframe

sns.heatmap(scled_df)  # 1 row for each car
plt.show()

sns.clustermap(scled_df, col_cluster=False)
plt.show()
# we are trying to cluster based on the car(rows) not on the features(columns)
# sns.clustermap(scled_df,row_cluster=False) # to evaluate the the relation between the features
# sns.heatmap(scled_df.corr()) # Another way


# Dummy features are going to be last in ine to get clustered as they are most Dissimilar, if one feature is 1 the
# other will be 0 Seaborn cannot cluster map for large data set

from sklearn.cluster import AgglomerativeClustering

# Defining the n_clusters
model = AgglomerativeClustering(n_clusters=4)
model_labels = model.fit_predict(scled_df)
print(model_labels)

# Defining the clusters numbers based on the distance
model1 = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
model1_labels = model1.fit_predict(scled_df)
print(model1_labels)

# Using scipy, we can visulzise the dendogram if we give the model linkage matrix

from scipy.cluster.hierarchy import linkage, dendrogram

linkage_matrix = linkage(model1.children_)
# Summary of the matrix , we will see 4  columns, Column and rows
# Example [67,161,1.41421536, 2.],[10,45,141421356,2]] ...
# at column index 0, there is cluster 67
# at column index 1, there is cluster 161, these two clusters will be combined, 2 is min requirement
# and at column index 2, 1.414421536, it is the distance between two clusters
# index 3 shows how many points in the cluster, this increase as we go up

print(linkage_matrix)

dendro = dendrogram(linkage_matrix, )
plt.show()
# index 0 and 1 will be x axis which will be nique, index 2 will be y axis and
# index 3 = how many points are being connected
# we can limit the cluster or cut of the hierrachy as seeing the entire dendogram does not give meaningful detail


dendro1 = dendrogram(linkage_matrix, truncate_mode='lastp', p=10)
plt.show()

dendro2 = dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.show()
# x axis is not easliy intepretable and dont go by the number
# we can use this method to draw a line that can help to decide number of cluster
