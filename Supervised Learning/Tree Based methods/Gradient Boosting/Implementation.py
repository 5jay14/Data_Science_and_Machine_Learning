import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\mushrooms.csv")

X = df.drop('class', axis=1)
X = pd.get_dummies(X, drop_first=True)
X = X.astype(int)
y = df['class']

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# Reducing the test set size since we are using Grid Search

model = GradientBoostingClassifier()
param_grid = {'learning_rate': [0.1, 0.05, 0.2],
              'n_estimators': [50, 100],
              'max_depth': [3, 4, 5]}
grid = GridSearchCV(estimator=model,param_grid=param_grid)

grid.fit(X_test, y_test)
grid_pred = grid.predict(X_test)
fea_imp = pd.DataFrame(data= grid.best_estimator_.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance')
# These are not co-efficients, these are based on Gini Impurity value(not the GI)
fea_imp = fea_imp.sort_values(by='Importance')

print(fea_imp)

cr = classification_report(y_test,grid_pred)
print(cr)


filtered_features = fea_imp[fea_imp['Importance'] > 0.009]
print(filtered_features)


sns.barplot(data=filtered_features,x=filtered_features.index,y='Importance')
plt.xticks(rotation =60)
plt.show()