import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

two_blobs = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cluster_two_blobs.csv")
two_blobs_outliers = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cluster_two_blobs_outliers.csv")


def display_categories(model, data):
    label = model.fit_predict(data)
    sns.scatterplot(data=data, x='X1', y='X2', hue=label, palette='Set1')
    plt.show()


from sklearn.cluster import DBSCAN

model = DBSCAN(eps=0.5)  # lower the eps, lower the circle radius, thus more outliers default is 0.5

display_categories(model, two_blobs)

# checkig the count of outliers, there are three categories for the above model, 0,1 and -1 for outliers
# use the attribute(.labels_)

total_outliers = np.sum(model.labels_ == -1)
total_points = len(model.labels_)
cluster_counts = np.unique(model.labels_)
outliers_percent = (total_outliers / total_points) * 100

# Elbow method

outliers_count = []
outliers_per = []
cluster_count = []
# Outliers count or percent with eps/min_samples can be visually seen with scatter plot

# Evaluating the outliers with Eps
for n in np.linspace(0.001,10,100):
    model = DBSCAN(eps=n)
    model.fit(two_blobs)
    labels = model.labels_

    outliers_count.append(np.sum(labels == -1))
    outliers_per.append(100 * (np.sum(labels == -1)) / len(labels))

    # we can also keep track of outliers, cluster counts
    cluster_count.append(len(np.unique(labels)))

sns.lineplot(x=np.linspace(0.001,10,100), y=outliers_count)
plt.ylabel('Outliers')
plt.xlabel('Eps')
# plt.xlim(0.0,2.0)
plt.show()

sns.lineplot(x=np.linspace(0.001,10,100), y=outliers_per)
plt.ylabel('Outliers % ')
plt.xlabel('Eps')
plt.hlines(y=3,xmin = 0, xmax=10, colors='red')
plt.show()


# Evaluating the outliers Min number of samples. Highers the min_points, higher the outliers


for n in np.arange(1,100):
    model = DBSCAN(min_samples=n)
    model.fit(two_blobs)
    labels = model.labels_

    outliers_count.append(np.sum(labels == -1))
    outliers_per.append(100 * (np.sum(labels == -1)) / len(labels))

    # we can also keep track of outliers, cluster counts
    cluster_count.append(len(np.unique(labels)))


sns.lineplot(x=np.arange(1,100), y=outliers_count)
plt.show()

sns.lineplot(x=np.arange(1,100), y=outliers_per)
plt.show()