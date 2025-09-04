import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

moons = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cluster_moons.csv")
blobs = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cluster_blobs.csv")
circles = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cluster_circles.csv")

# visualising the points arrangement using scatter plot
# this is unlabelled data

sns.scatterplot(moons, x='X1', y='X2')
plt.show()

sns.scatterplot(blobs, x='X1', y='X2')
plt.show()

sns.scatterplot(circles, x='X1', y='X2')
plt.show()


def category_labels(model, data):
    labels = model.fit_predict(data)
    sns.scatterplot(data=data, x= 'X1', y='X2', hue=labels)
    plt.show()

from sklearn.cluster import KMeans, DBSCAN

kmeans_blobs = KMeans(3)
category_labels(kmeans_blobs,blobs)

kmeans_circles = KMeans(2)
category_labels(kmeans_circles,circles)

kmeans_moons = KMeans(2)
category_labels(kmeans_moons,moons)

# the data is being sliced as Kmeans works based on the distance metric. I have adjusted N cluster accordingly to easily
# Visualise what Kmeans is doing

from sklearn.cluster import DBSCAN

dbscan_blobs = DBSCAN() # default hyperparameters
category_labels(dbscan_blobs,blobs) # it also identified the outliers

dbscan_moons = DBSCAN(eps=0.15)
category_labels(dbscan_moons, moons)

dbscan_circles = DBSCAN(eps=0.15)
category_labels(dbscan_circles,circles)