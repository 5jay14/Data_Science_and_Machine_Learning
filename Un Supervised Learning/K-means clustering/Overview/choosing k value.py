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
ssd = []

for k in range(2, 10):
    model = KMeans(n_clusters=k)
    model.fit(scaled_X)
    ssd.append(model.inertia_)  # ssd_point --> cluster center

plt.plot(range(2, 10), ssd, 'o--')
plt.show()
print(ssd)

# calculating differences

ser = pd.Series(ssd)
print(ser.diff())
# it is represneted with series index not actual k value, K valuea are linked with automatic index