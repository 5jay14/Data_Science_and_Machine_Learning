import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")

# assigning the Features and labels
X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LassoCV

lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
lasso_cv_model.fit(X_train,y_train)
print(lasso_cv_model.alpha_)
print(lasso_cv_model.coef_)
# alpha can obtained in two ways,
# epsolon(eps) = alpha min/alpha max
# arrays of alpha from 0 to positive infinity
