import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\bank-full.csv")


X = pd.get_dummies(df, drop_first=True)
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

model = KMeans(n_clusters=2, random_state=101)
model.fit(scaled_X)
cluster_label = model.predict(scaled_X)
X['cluster'] = cluster_label
X_corr = X.corr()['cluster'].iloc[:-1].sort_values().plot(kind = 'bar')


plt.show()
print(X_corr)