        # Issue with Train test split is that it is not 100% fair evaluation even though the newer model as never seen the
# X_test data but we adjust the alpha parameter on the newer model based on the performance of the previous model
# So what we can do is, we can hold of some data that the model is adjusted on or fitted to so we can achieve  fair evaluation
# If we want a truly fair and final set of performance metrics, we should get these metrics from the final set set
# of data that we do not allow ourseleves to adjust on

"""
1.Decide the data distribution for test, validation and test. 70,20,10
2.set aside the test set/hold out set
3.Fit the model on the Training set
4.Evaluate model performance on evlatuation set
5.Adjust the hyperparameters on validation set
6.Repeat the 4 and 5 process
7.Perform Final evaluation on hold out test set. we cannot adjust the hyperparamets here. This should only be used to
get the model metric performance
"""

# Python Implementation
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Advertising.csv")
print(df.head())

X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split
X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.3, random_state=101)  # Training data gets
# 70% data, but the remaining 30% is not assigned so to speak so assign it to other

# test size of 0.5 => 50% of 30% other is 15% of all data
X_eval, X_test, y_eval, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=101)

# Confirming the distribution
print(len(df), len(X_train), len(X_eval), len(X_test))

# Scaling, we need to scale all(test, eval and test) as the model will perform validation on scaled data
from sklearn.preprocessing import  StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)


from sklearn.linear_model import Ridge
model_1 = Ridge(alpha=100)
model_1.fit(X_train,y_train)
y_eval_prediction = model_1.predict(X_eval)


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_eval,y_eval_prediction)
print(MSE) # 7.320101458823871

model_2 = Ridge(alpha=1)
model_2.fit(X_train,y_train)
y_eval_prediction_2 = model_2.predict(X_eval)
MSE = mean_squared_error(y_eval,y_eval_prediction_2)
print(MSE) # 2.383783075056986

# Final performance
y_final_prediction = model_2.predict(X_test)
MSE = mean_squared_error(y_test,y_final_prediction)
print(MSE)