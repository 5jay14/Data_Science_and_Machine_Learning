import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")

# assigning the Features and labels
X = df.drop('sales', axis=1)
y = df['sales']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

''' 
Ridge Regression, Lambda(tunable parameter) from the formula is referred as Alpha in scikit learn Cross validation is
used to determine the alpha For cross validation metrics, sklearn uses a scorer object. THe convention, higher the
return values are better than the lower return values which is a caveat.
Ridge CV comes with built in Cross validation
Example: 
Higher accuracy is better but higher RMSE is actually worse. So scikit fixes this by making RMSE negative as its scorer 
metric. This allows uniformity across all scorer metrics
'''

from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=10) # sample
ridge_model.fit(X_train,y_train)
test_prediction = ridge_model.predict(X_test)


# performance metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
MAE = mean_absolute_error(y_test,test_prediction)
rMSE = np.sqrt(mean_squared_error(y_test,test_prediction))
MSE = mean_squared_error(y_test,test_prediction)

# Scikit learn offers another version of ridge regression which helps in cross validating to find the alpha
# and it returns the best alpha, which is the average error value

from sklearn.linear_model import RidgeCV
ridge_cv_model = RidgeCV(alphas=(0.1,1.0,10.0), scoring='neg_mean_absolute_error')
ridge_cv_model.fit(X_train, y_train)

# finding which alpha that performed the best
print(ridge_cv_model.alpha_)

# checking scoring metrics, all the scoring metrics are transformed to show higher the better
# from sklearn.metrics import SCORERS

'''
Classification Metrics
Accuracy: accuracy
Balanced Accuracy: balanced_accuracy
ROC AUC: roc_auc
Log Loss: neg_log_loss
F1 Score: f1
Precision: precision
Recall: recall

Regression Metrics
Mean Absolute Error: neg_mean_absolute_error
Mean Squared Error: neg_mean_squared_error
Root Mean Squared Error: neg_root_mean_squared_error
R^2 Score: r2
'''

# checking performance based out of alpha parameter

test_prediction = ridge_cv_model.predict(X_test)
MAE1 = mean_absolute_error(y_test, test_prediction)
RMSE1 = np.sqrt(mean_squared_error(y_test,test_prediction))
print(MAE1,RMSE1)