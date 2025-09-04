import numpy as np
import pandas as pd
import joblib

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")
X = df.drop('sales', axis=1)
y = df['sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

X_validation, X_holdout, y_validation, y_holdout = train_test_split(X_test, y_test, test_size=0.5, random_state=101)
# does not matter which is holdout or validation as long as it is consistent
# Note: X_test, y_test will not be used anywhere

print(len(X_train), len(X_test), len(X_holdout), len(X_validation))

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=10, random_state=101)
model.fit(X_train, y_train)

validation_pred = model.predict(X_validation)

from sklearn.metrics import mean_squared_error, mean_absolute_error

MSE = np.sqrt(mean_squared_error(y_validation, validation_pred))
print(MSE)

MEA = mean_absolute_error(y_validation, validation_pred)
print(MEA)

print(df.describe()['sales'])

holdout_predictions = model.predict(X_holdout)

MEA = mean_absolute_error(y_holdout, holdout_predictions)
print(MEA)

MSE = np.sqrt(mean_squared_error(y_holdout, holdout_predictions))
print(MSE)

Final_model = RandomForestRegressor(n_estimators=10, random_state=101)
Final_model.fit(X, y)

joblib.dump(Final_model, r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Final_model.pkl')

joblib.dump(list(X.columns), r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Col_names.pkl')

# Test the pkl files
new_columns = joblib.load(r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Col_names.pkl')
print(new_columns)
new_model = joblib.load(r'C:\Users\vijay\Data Science and Machine Learning\Model Deplloyment\Final_model.pkl')
print(new_model.predict([[230.1, 37.8, 69.2]]))
