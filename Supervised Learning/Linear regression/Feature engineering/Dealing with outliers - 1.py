import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Ames_Housing_Data.csv")


def create_ages(mu=50, sigma=13, num_samples=100, seed=42):
    # mu is mean and sigma is standard deviation
    # We set seed to 42 (42 is an arbitrary choice)
    np.random.seed(seed)

    sample_ages = np.random.normal(loc=mu, scale=sigma, size=num_samples)
    sample_ages = np.round(sample_ages, decimals=0)

    return sample_ages


sample = create_ages()
sns.displot(sample, bins=20)
# This shows the normal distribution, we cannot tell an outlier most of the data seems to be around 70
plt.show()

sns.boxplot(sample)  # this shows the outlier by using the IQR inter quartile range
plt.show()

ser = pd.Series(sample)

print(ser.describe())
# another method using numpy, below is giving me 75 and 25 percentile
q75, q25 = np.percentile(sample, [75, 25])

print(df.describe())

print(df.info())

# finding correlation

# data_num = df.select_dtypes(include=['int64','float']).corr()# this shows the correlation between all
data_num = df.select_dtypes(include=['int64','float']).corr()['SalePrice'].sort_values()
# relation of other features with this column, look for positive
# overall quality feature has the highest positive correlation
print(data_num)
