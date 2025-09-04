import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creating X and Y
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Advertising.csv")
X = df.drop('sales', axis=1)
y = df['sales']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import ElasticNet
elastic_model = ElasticNet()
param_grid = {'alpha': [0.1,1,5,10,100], 'l1_ratio': [0.1,0.5,0.7,0.95,0.99,1]}

from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(estimator=elastic_model,
                          param_grid=param_grid,
                          scoring= 'neg_mean_squared_error',
                          cv=5,# K fold
                          verbose=1) # how much information is outputted

grid_model.fit(X_train,y_train)
print(grid_model.best_params_)