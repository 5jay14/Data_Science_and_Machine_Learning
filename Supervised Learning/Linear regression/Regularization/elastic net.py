# we know that lasso can shrink some coeff to 0, elatic net makes it clear, which works by combining l1(lasso) and l2(ridge)
# regression


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, ElasticNetCV
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")

# assigning the Features and labels
X = df.drop('sales', axis=1)
y = df['sales']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
elastic_cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], eps=0.001,n_alphas=100, max_iter=10000000)
# l1_ratio here is the alpha from the equation(ratio btween l1 and l2)
# n_alphas is the lambda


elastic_cv_model.fit(X_train,y_train)
print(elastic_cv_model.l1_ratio_) # an attribute of the model that shows best performing l1 ratio
