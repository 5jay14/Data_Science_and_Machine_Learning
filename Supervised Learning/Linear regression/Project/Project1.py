import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\AMES_Final_DF.csv")
print(df.head())
print(df.info())

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
Scaled_X_train = scaler.transform(X_train)
Scaled_X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNet

elastic_model = ElasticNet(max_iter=100)
param_grid = {'alpha': [0.1, 1, 5, 50, 10, 100], 'l1_ratio': [0.1, 0.5, 0.7, 0.95, 99,1]}


from sklearn.model_selection import GridSearchCV

grid_model = GridSearchCV(estimator=elastic_model,
                          param_grid=param_grid,
                          cv=5, verbose=1,
                          scoring='neg_mean_squared_error')

grid_model.fit(Scaled_X_train, y_train)
print(grid_model.best_params_)

y_pred = grid_model.predict(Scaled_X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y_test,y_pred)
print(MAE)

MSE = np.sqrt(mean_squared_error(y_test,y_pred))
print(MSE)

print(np.mean(df['SalePrice']))