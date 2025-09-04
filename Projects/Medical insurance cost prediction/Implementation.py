import numpy as np
import pandas as pd


load_data = pd.read_csv(r"C:\Users\vijay\Data Science and Machine Learning\DATA\insurance.csv")

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

poly = PolynomialFeatures(degree=3)

X = load_data.drop('charges', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = load_data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_validation, X_holdout, y_validation, y_holdout = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

scaler = StandardScaler()
scaled_X_Train = scaler.fit_transform(X_train)
Scaled_X_validation = scaler.transform(X_validation)
Scaled_X_holdout = scaler.transform(X_holdout)


def func_model(model, x_train, Y_train,x_val,Y_eval):
    model.fit(x_train, Y_train)
    y_pred = model.predict(x_val)

    MAE = mean_absolute_error(Y_eval,y_pred)
    print(f'MAE score is :{MAE}')

    MSE = mean_squared_error(Y_eval,y_pred)
    print(f'MSE score is :{MSE}')

    RMSE = np.sqrt(mean_squared_error(Y_eval,y_pred))
    print(f'RMSE score is :{RMSE}')

linear_model = LinearRegression()
func_model(linear_model,scaled_X_Train,y_train,Scaled_X_validation,y_validation)

from sklearn.tree import DecisionTreeClassifier


