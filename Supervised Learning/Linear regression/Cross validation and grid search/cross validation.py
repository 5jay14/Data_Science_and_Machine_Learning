# Cross validation functions allows us to view multiple performance metrics from CV on a model and exlore how much time
# fitting and testing took

import pandas as pd
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Advertising.csv")

X = df.drop('sales',axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train) # CV will be done on this test set and hyperparamters adjustment as well
X_test = scaler.transform(X_test) # hold out test set

from sklearn.model_selection import cross_validate
from sklearn.linear_model import  Ridge
model = Ridge(alpha=100)
scores = cross_validate(model, X_train, y_train, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],cv=10)
print(scores) # returns a dictionary which is hard to interpret

scores = pd.DataFrame(scores)
print(scores)
print(scores.mean())

# improving the model
model_1 = Ridge(alpha=1)
scores = cross_validate(model_1, X_train, y_train, scoring=['neg_mean_squared_error','neg_mean_absolute_error'],cv=10)
scores = pd.DataFrame(scores)
print(scores)
print(scores.mean())

# final data
model_1.fit(X_train,y_train)
final_y_pred = model_1.predict(X_test)


from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test,final_y_pred)
print(MSE)
