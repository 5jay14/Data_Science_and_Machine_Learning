import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\data_banknote_authentication.csv")
print(df.info())

sns.pairplot(data=df, hue='Class')
plt.show()

X = df.drop('Class',axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.15, random_state=101)



n_estimators = [64,100,128,200]
max_features = [2,3,4]
bootstrap = [True,False]
oob_score = [True,False]
# There are going to be warnings or errors since for some instance the boot strap will be set to false

param_grid = {'n_estimators':n_estimators,
              'max_features':max_features,
              'bootstrap':bootstrap,
              'oob_score':oob_score}

RFC = RandomForestClassifier()
grid_model = GridSearchCV(RFC,param_grid)
grid_model.fit(X_train,y_train)
#print(grid_model.best_params_) #{'bootstrap': False, 'max_features': 2, 'n_estimators': 100, 'oob_score': False}

RFC = RandomForestClassifier(bootstrap=True,max_features=2,n_estimators=100,oob_score=True)

RFC.fit(X_train,y_train)

preds = RFC.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay,accuracy_score
report = classification_report(y_test,preds)
print(report)
cm = confusion_matrix(y_test,preds)
cmd = ConfusionMatrixDisplay(cm,display_labels=RFC.classes_)
cmd.plot()
plt.show()


# Plotting errors vs estimators
errors = []
misclassification = []



for n in range(1,200):
    RFC = RandomForestClassifier(n_estimators=n,max_features=2)
    RFC.fit(X_train,y_train)
    preds = RFC.predict(X_test)
    err = 1 - accuracy_score(y_test,preds)
    n_miscla = np.sum(y_test != preds)
    errors.append(err)
    misclassification.append(n_miscla)

print(errors)
print(misclassification)

plt.plot(range(1,200),errors)
plt.show()

plt.plot(range(1,200),misclassification)
plt.show()