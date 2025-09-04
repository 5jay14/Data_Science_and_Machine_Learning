import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR, LinearSVR, SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, precision_score,
                             recall_score, ConfusionMatrixDisplay)

# Wine fraud relates to the commercial aspects of wine. The most prevalent type of fraud is one where wines are
# adulterated, usually with the addition of cheaper products (e.g. juices) and sometimes with harmful chemicals and
# sweeteners (compensating for color or flavor).

# Counterfeiting and the relabelling of inferior and cheaper wines to more expensive brands is another common type of
# wine fraud. A distribution company that was recently a victim of fraud has completed an audit of various samples of
# wine through the use of chemical analysis on samples. The distribution company specializes in exporting extremely
# high quality, expensive wines, but was defrauded by a supplier who was attempting to pass off cheap, low quality
# wine as higher grade wine. The distribution company has hired you to attempt to create a machine learning model
# that can help detect low quality (a.k.a "fraud") wine samples. They want to know if it is even possible to detect
# such a difference.

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\wine_fraud.csv")
print(df['quality'].value_counts())
sns.countplot(data=df, x='quality')
plt.show()  # The data is not balanced

sns.countplot(data=df, x='type', hue='quality')
plt.show()

# Fraud % for each type
reds = df[df['type'] == 'red']
whites = df[df['type'] == 'white']
print((len(reds[reds['quality'] == 'Fraud']) / len(reds)) * 100)
print((len(whites[whites['quality'] == 'Fraud']) / len(whites)) * 100)

# String handling
# Converting the string variables in the label column to binary values for finding the correlation
df['Fraud'] = df['quality'].map({'Legit': 0, 'Fraud': 1})

# Creating dummies for the string variables manually
# df['type'] = pd.get_dummies(df['type'],drop_first=True) # This is returning booleans
df['red_wine'] = df['type'].map({'red': 1, 'white': 0})
df['white_wine'] = df['type'].map({'white': 1, 'red': 0})
df = df.drop(['type', 'quality'], axis=1)
print(df.corr()['Fraud'].sort_values())

df.corr()['Fraud'].sort_values().plot(kind='bar')
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

X = df.drop('Fraud', axis=1)
y = df['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
scaler = StandardScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

model = SVC(class_weight='balanced')

param_grid = {'C': [0.01, 0.10, 0.5, 1],
              'kernel': ['rbf', 'linear'],
              'gamma': ['scale', 'auto']
              }

grid = GridSearchCV(model, param_grid)
grid.fit(scaled_X_train, y_train)
print(grid.best_params_)

y_pred_class = grid.predict(scaled_X_test)

accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
print(accuracy, precision, recall)

cr = classification_report(y_test, y_pred_class)
print(cr)

CM = confusion_matrix(y_test, y_pred_class)  # shows for the test numbers only
print(CM)

CMD = ConfusionMatrixDisplay(confusion_matrix=CM,display_labels=grid.classes_ )
CMD.plot()
plt.show()
