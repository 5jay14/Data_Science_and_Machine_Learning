import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import geopandas as gpd

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\CIA_Country_Facts.csv")
missing_agri = df[df['Agriculture'].isnull()]['Country']
# we can see the countries with no value for agri is either small islands or desert  or small population countries,
# so we are going to fill those countries with 0 with a the above assumption
df[df['Agriculture'].isnull()] = df[df['Agriculture'].isnull()].fillna(0)

# for missing climate and literacy values, we can simply take the region mean value and fill the null values
# 1st find the region average
# fill the mean value of the region to those null countires


df['Climate'] = df['Climate'].fillna(df.groupby('Region')['Climate'].transform('mean'))
df['Literacy (%)'] = df['Literacy (%)'].fillna(df.groupby('Region')['Literacy (%)'].transform('mean'))

X = df.iloc[:, 1:]
X = X.dropna()
X = pd.get_dummies(X)
print(X.info())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ssd = []

for k in range(2, 30):
    model = KMeans(n_clusters=k)
    model.fit(X_scaled)
    ssd.append(model.inertia_)

plt.plot(range(2, 30), ssd, 'o--')
plt.show()
print(ssd)

pd.Series(ssd).diff().plot(kind='bar')
plt.show()

model1 = KMeans(n_clusters=14)
model1.fit(X_scaled)
print(model1.labels_)
X['K=3,cluster_labels'] = model1.labels_
corr = X.corr()['K=3,cluster_labels'].iloc[:-1].sort_values()
corr.plot(kind='bar')
plt.show()


print(X.head(1))
# iso_codes = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\country_iso_codes.csv")
#
# fig = px.choropleth(X, locations="iso_alpha",
#                     color="lifeExp", # lifeExp is a column of gapminder
#                     hover_name="country", # column to add to hover information
#                     color_continuous_scale=px.colors.sequential.Plasma)