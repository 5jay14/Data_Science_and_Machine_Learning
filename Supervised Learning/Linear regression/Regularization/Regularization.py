import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")

# assigning the Features and labels
X = df.drop('sales', axis=1)
y = df['sales']

# Creating polynomial features
from sklearn.preprocessing import PolynomialFeatures

polynomial_converter = PolynomialFeatures(degree=3, include_bias=False)
polynomial_features = polynomial_converter.fit_transform(X)

# splitting the train test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(polynomial_features, y, test_size=0.33, random_state=42)

# Scaling the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # While performing standardization, we need to take the mean and standard deviation from the
# training set in order to avoid data leakage

Scaled_X_train = scaler.transform(X_train)
Scaled_X_test = scaler.transform(X_test)














