# There are models that has inbuilt CV. Example: RidgeCV
# SCIKIT learn offers general tools for utilizing CV for other models
# Most basic CV is train test split and the other common one is k-fold CV

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Advertising.csv")
print(df.head())

# Train | Test split procedure
"""
0. Clean and adjust data for X and Y
1. Split data in train and test for both X and Y
2. Fit/Train scaler on training X data
3. Scale X data
4. Create Model
5. Fit/Train model on X Train data
6. Evaluate Model on X Test data(By creating predictions and comparing to Y_test)
7. Adjust the parameters as necessary and repeat the steps 5 and 6
"""
X = df.drop('sales', axis=1)
y = df['sales']

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Scaling and transforming data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)  # Fitting the train data to avoid leakage

Scaled_X_train = scaler.transform(X_train)
Scaled_X_test = scaler.transform(X_test)

from sklearn.linear_model import Ridge

model = Ridge(alpha=100)
model.fit(Scaled_X_train, y_train)
y_prediction = model.predict(Scaled_X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

MSE = mean_squared_error(y_test,y_prediction)
print(MSE) # 7.34177578903413

model_two = Ridge(alpha=1)
model_two.fit(Scaled_X_train, y_train)
y_prediction_two = model_two.predict(Scaled_X_test)

MSE = mean_squared_error(y_test, y_prediction_two)
print(MSE)

"""
Advantage: Easy to understand and implement
Disadvantage: Getting the optimal alpha parameter is time consuming. We can run a loop but it takes time
"""

