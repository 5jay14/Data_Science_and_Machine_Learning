# Cross val score of scikit learn helps to apply cross validation for the models that does not come with inbuilt with
# CV version


"""
# K Fold CV
1. Split the Training(Larger) and Test(Smaller) data set
2. Remove the test data set for final evaluation. This becomes the hold out data set
3. Chose the K fold value to split the Training data. Larger the K more computation
    K = Number of rows
    Largest split possible = K-1(Leave one out policy)
4. After choosing the K fold value lets say K =5, train on K-1 folds. 4 folds
5. Validate the results on remaining 1 fold and obtain the first error
6 Repeat the steps for other combination
7. Obtain the error for all the combination and take the mean of the errors
8. Adjust the Hyperparameters till the the result is satisfactory(Avg error)
9. Get the final metrics from the final test set
10 Cross val score does this all for us easily
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Advertising.csv")
X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import Ridge

model = Ridge(alpha=100)  # Creating a model with random alpha

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# Parameters that cross val score accepts
# estimator = model, X_Train, Y_Train, CV = K folds, 'Scoring error metric' get this from scoring metrics lesson
# observe, any model can be passed here
print(scores)  # returns the array of error for five combination
# [ -9.32552967  -4.9449624  -11.39665242  -7.0242106   -8.38562723]. Some folds is performing better which is a good
# indicator of getting true evaluation of the model

print(abs(scores.mean()))
# 8.215396464543607, this is the average of errors which is not that good compared to the normal train test split so
# adjust the hyperparamters


model_1 = Ridge(alpha=1)
scores = cross_val_score(model_1, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
# model is not really fit inside, we have it fit it again for training data set
print(abs(scores.mean()))

# now we need to fit the model all the training data
model_1.fit(X_train,y_train)
y_pred = model_1.predict(X_test)

from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,y_pred)
print(MSE)