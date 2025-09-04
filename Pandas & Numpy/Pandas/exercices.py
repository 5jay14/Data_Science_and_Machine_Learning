import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df3 = pd.read_csv(r"C:\Users\vijay\Desktop\df3.csv")
print(df3.head(),df3.info())

df3.plot.scatter(x='a',y='b')
fig= plt.figure(figsize=(8, 3), dpi=100)

plt.show()