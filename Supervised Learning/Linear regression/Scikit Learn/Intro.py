import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")
print(df.describe()) # shows the mean, standard deaviation, count, min and max

# Validating if there is a linear relation
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
axes[0].plot(df['TV'], df['sales'], 'o')
axes[0].set_ylabel('Sales')
axes[0].set_title('Tv Spend')

axes[1].plot(df['radio'], df['sales'], 'o')
axes[1].set_ylabel('Sales')
axes[1].set_title('Radio Spend')

axes[2].plot(df['newspaper'], df['sales'], 'o')
axes[2].set_ylabel('Sales')
axes[2].set_title('Newspaper Spend')
plt.tight_layout()
plt.show()

# Assigning labels and features
X = df.drop('sales', axis=1)  # drops sales column and returns other columns
y = df['sales']

'''
train_test_split shuffles the data to avoid using the the ordered date(if it is ordered)
Meaning if the data is ordered based on the sales column, we dont want to consider the x percentage from beginning 
for training.
'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# hover over train_test_split and copy the tuples to unpack
# test_size = decide what percentage of the data we want to train

print(X_train)  # index position of the original DF will be retained
print(y_train)  # index position of the original DF will be retained

# Creating an estimator or building a model
# Choose the family of models and import the model

from sklearn.linear_model import LinearRegression

# hyperparameters are the parameters that we can edit to adjust the model performance
# print(help(LinearRegression))

model = LinearRegression()  # Create the instance of the model
model.fit(X_train, y_train)  # pass the train data
test_predictions = model.predict(X_test)  # pass the test data to predict the corresponding test label

# Evaluating the performance of the model
# comparing the X_test with y_test
'''
1. MAE
2. MSE
3. RMSE - just take the square root of MSE
'''

from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1, take the average value of the sales from the original data frame
df['sales'].mean()
# Average sales value is 14.0225

sns.histplot(data=df, x='sales', bins=50)  # shows the general distribution of sales values
plt.show()

MAE = print(mean_absolute_error(y_test, test_predictions))
# 1.4937750024728966

'''
so the average sales value is 14.0225
MAE is 1.49378
on the data the model has never seen before, the prediction is going to be off by 1.49378
when compared to average sales value = 1.49378/14.0225 = so the error is <10%
is it acceptable? again it is the context
'''

MSE = mean_squared_error(y_test, test_predictions)
print(MSE)  # 3.7279283306815105

RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))
print(RMSE)  # 1.9307843822347204

# Residual plots: they help to identify the pattern. We can make use of RP to identify if linear regression is the right
# fit for the given data set

test_residuals = y_test - test_predictions
print(test_residuals)
sns.scatterplot(x=y_test, y=test_residuals)
plt.axhline(y=0, color='red', ls='--')  # y = 0 because it shows the perfect fit.
plt.show()  # the graph shows the data points are scattered and normally distributed. This shows no clear sign of any
# clear curve or pattern, which is a good indication that it is good fit for linear regression

# Saving model and coefficient interpretation

final_model = LinearRegression()
final_model.fit(X,y) # note, we are fitting with the full data set not only test data set
print(final_model.coef_)
# This shows how much sales changes for 1 unit change in that feature
# [ 0.04576465  0.18853002 -0.00103749] # news paper spend has no relation to sales, actuallly it is negative


#Saving and loading

from joblib import dump,load
dump(final_model,'final_sales_model.joblib')

#loading the saved model which was trained earlier
loaded_model = load('final_sales_model.joblib')

#using the model on a new dataset
X.shape # this shows the shape of the data frame. 200 rows into 3 columns(features). This is 2d rows x columns

new_campaign = [[149,22,15]]   
# tv, radio, newspape
print(loaded_model.predict(new_campaign))
# [13.88991952]