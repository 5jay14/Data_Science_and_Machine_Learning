# Using SVM to predict continuous label, the idea is same as classification
# So here, we dont care which side the points falls on. Example if we are using X feature and Y label
# Y plotted vertically and X horizontally, we need to check the Y value for X data

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR,LinearSVR

# For regression using SVM, there are multiple models to choose from
# SVR - Generic version. Can be used with any kernel
# LinearSVR - Provides a faster implementation but the caveat is it only considers linear kernel.
# Cannot use other kernel such as RBF, poly

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\cement_slump.csv")
print(df.columns)
X = df.drop("Compressive Strength (28-day)(Mpa)",axis=1)
y = df["Compressive Strength (28-day)(Mpa)"]
sns.heatmap(df.corr(),annot=True)
plt.show()


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=101)


# For SVM or for any model that considers the feature space(Dimensions), Scaling the data is recommended
# When not sure, scale the data anyway. There is no negative impact from scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Parameters passed : C, Gamma, Kernel, degree and epsilon
# epsilon: it denotes how much error we are willing to allow per training data instance
# if 0, then maximum error allowed is 0 per training data instance is 0 which could lead to overfitting.
# Do the Grid serach CV for best C, epsilon and kernel combo

base_model = SVR()
base_model.fit(scaled_X_train,y_train)
base_pred = base_model.predict(scaled_X_test)

mean_absolute_error(y_test,base_pred) # 5.236902091259179
np.sqrt(mean_squared_error(y_test,base_pred)) # 6.695914838327134
y_test.mean() # 36.26870967741935

param_grid = {'C':[0.001,0.01,0.5,1],
              'kernel':['rbf','linear','poly'],
              'degree':[2,3,4],
              'epsilon':[0,0.01,0.10,1,2,3,1.5,0.5]}

tuned_SVR = SVR()
grid = GridSearchCV(tuned_SVR,param_grid)
grid.fit(scaled_X_train,y_train)
print(grid.best_params_) # {'C': 1, 'degree': 2, 'epsilon': 2, 'kernel': 'linear'}
grid_pred = grid.predict(scaled_X_test)
mean_absolute_error(y_test,grid_pred) # 2.5128012210762005
np.sqrt(mean_squared_error(y_test,grid_pred)) # 3.178210305119844
