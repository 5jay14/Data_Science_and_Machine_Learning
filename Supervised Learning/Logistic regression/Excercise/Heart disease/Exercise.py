# In this exercise, we will predict if the person has a heart disease based on the features give

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\heart.csv")
X = df.drop('target', axis=1)
y = df['target']

print(df.info())  # dont see any missing data
print(df.isnull().sum())  # another way finding missing data
print(df.describe())  # take into account of age(min, max, mean and SD)

'''
features = ['age','trestbps', 'chol','thalach','target']
df_subset = df[features]
sns.pairplot(df_subset,hue='target')
plt.show()


sns.heatmap(df.corr(),annot=True)
plt.show()

sns.countplot(df,x='target') # 140=0 ,160=1. Balanced
plt.show()

sns.scatterplot(data=df, x ='cp', y = 'age', hue='target')
plt.show()

sns.boxplot(data=df,x='target',y='age')
plt.show()

'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)

scaler = StandardScaler()

scaled_X_Train = scaler.fit_transform(X_train)
scaled_x_Test = scaler.transform(X_test)

# If we want to use Logistic regression CV, instead of combining grid search and standard logistic regression)
# C value will be default to 10, meaning 10 values will be evaluated between logarithmic scale of 1e-4, 1e4

logCV_model = LogisticRegressionCV()
logCV_model.fit(scaled_X_Train, y_train)
print(logCV_model.Cs_)  # prints what C values the model evaluated on
print(logCV_model.C_)  # prints what C value it chose
print(logCV_model.coef_)
coeffs = pd.Series(index=X.columns, data=logCV_model.coef_[0]) # chose the first index because model.coeff gives two brackets
coeffs = coeffs.sort_values()

sns.barplot(x=coeffs.index,y = coeffs.values)
plt.show()

model = LogisticRegression()
param_rid = {'penalty': ['l1', 'l2', 'elasticnet'],
             'C': np.linspace(0, 1, 20),
             'l1_ratio': np.linspace(0, 1, 20)}
grid_model = GridSearchCV(model, param_grid=param_rid)
grid_model.fit(scaled_X_Train, y_train)
best_param = grid_model.best_params_
best_score = grid_model.best_score_
best_model = grid_model.best_estimator_

print(best_param,best_score,best_model)

y_pred = grid_model.predict(scaled_x_Test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
CM = confusion_matrix(y_test, y_pred)
print(accuracy, recall, report, CM)

cm = confusion_matrix(y_test, y_pred, labels=grid_model.classes_)  # ,normalize='all', this will show it in %
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_model.classes_)
disp.plot()
plt.show()

RocCurveDisplay.from_estimator(estimator=grid_model, X=scaled_x_Test, y=y_test)
plt.show()

PrecisionRecallDisplay.from_estimator(estimator=grid_model, X=scaled_x_Test, y=y_test)
plt.show()

patient = [[54., 1., 0., 122., 286., 0., 0., 116., 1., 3.2, 1., 2., 2.]]
scaled_patient = scaler.transform(patient)

print(grid_model.predict(patient))
print(grid_model.predict_proba(patient))

coefficients_df = pd.DataFrame({
    'Feature1': [9.99999967e-01],
    'Feature2': [3.33120242e-08]
})

# Display coefficients in a readable format
pd.set_option('display.float_format', lambda x: '{:.10f}'.format(x))
print(coefficients_df)