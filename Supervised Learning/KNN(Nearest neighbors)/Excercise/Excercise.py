import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix



df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\sonar.all-data.csv")

# df['Target'] = df['Label'].map({'M': 1, 'R': 0}) # Creating a new column and mapping
df['Label'] = df['Label'].replace({'M': 1, 'R': 0}) # making changes in the same column
print(df['Label'].value_counts())

X = df.drop('Label',axis=1)
y = df['Label']
# if y was a series, use this code to replace the label values
# y.replace({'M':1,'R':0})

#Task 1
df.describe() # EDA: data is neatly organised
print(np.abs(df.corr()['Label']).sort_values(ascending=False).head(6))
# use the absolute values as there could negative values with higher correlation
sns.heatmap(df.corr())
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = StandardScaler()
knn = KNeighborsClassifier()

operations = [('scaler',scaler),('knn',knn)]
pipe = Pipeline(operations)

k_values = list(range(1,30))
param_grid = {'knn__n_neighbors':k_values}

full_cv_classifier = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
full_cv_classifier.fit(X_train,y_train)
print(full_cv_classifier.best_estimator_.get_params())

y_pred = full_cv_classifier.predict(X_test)

CR = classification_report(y_test,y_pred)
print(CR)
print(full_cv_classifier.cv_results_)
df1 = pd.DataFrame(full_cv_classifier.cv_results_)['mean_test_score'].plot() # Accuracy vs K Neighbour
plt.show()
print(df1)

CM = confusion_matrix(y_test,y_pred)
print(CM)
# Model predicted 2 instances wrongly, where it falsely identifed mine as rock(FN). We can look at ROC curve for tradeoff