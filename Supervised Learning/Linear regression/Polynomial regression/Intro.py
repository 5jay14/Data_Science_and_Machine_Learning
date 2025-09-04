import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")
X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.preprocessing import PolynomialFeatures

pl = PolynomialFeatures(degree=2, include_bias=False)  # features squared
pl.fit(X)  # this analyses the relation
poly_features = pl.transform(X)  # this transforms the data

# difference between original vs new data set
# print(X.head())
# print(new_data)  # now has 6 additional features

# both fit and transform can be done in one line
# pl.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(poly_features,y,test_size=0.33, random_state=42)
# pass the dataset with polynomial features

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

test_model = model.predict(X_test)

# Evaluating performance metrics

from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y_test,test_model)
MSE = mean_squared_error(y_test,test_model)
RMSE = np.sqrt(MSE)


print(MAE,MSE,RMSE)

# Various polynomial degree higher degree polynomial
# Chossing right plolynomial degree without overfitting. We can do it with a loop and main steps are following

'''
1. Create the different order poly
2. split poly train/test
3. Fit on train
4. Store/save the RMSe for both the train and test
5. Plot the results(Error vs poly order)
'''

train_rmse_error = [] # storing rmse for train set
test_rmse_error =[] #storing rmse for test set

for d in range(1,10):
    poly_converter = PolynomialFeatures(degree=d, include_bias=False)
    poly_features1 = poly_converter.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(poly_features1, y, test_size=0.33, random_state=42)

    model1 = LinearRegression()
    model1.fit(X_train,y_train)

    train_pred = model1.predict(X_train)
    test_pred = model1.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred)) # RMSe for both the train and test
    test_rmse =  np.sqrt(mean_squared_error(y_test, test_pred)) # RMSe for both the train and test

    train_rmse_error.append(train_rmse)
    test_rmse_error.append(test_rmse)

print(train_rmse_error)
print(test_rmse_error)