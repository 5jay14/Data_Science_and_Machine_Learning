import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

'''
Train a single decision tree model (feel free to grid search for optimal hyperparameters).
Evaluate performance metrics from decision tree, including classification report and plotting a confusion matrix.
Calculate feature importances from the decision tree.
OPTIONAL: Plot your tree, note, the tree could be huge depending on your pruning, so it may crash your notebook if you display it with plot_tree.
'''
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Telco-Customer-Churn.csv")
df = df.drop('customerID', axis=1)

categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

df = pd.get_dummies(data=df, columns=categorical, drop_first=True)
X = df.select_dtypes(include=['bool', 'float']).astype(int)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=101)
DT = DecisionTreeClassifier(max_depth=6)
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
cr = classification_report(y_test, y_pred)
print(cr)

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DT.classes_)
cmd.plot()
plt.show()

feature_imp = pd.DataFrame(data=DT.feature_importances_,
                           index=X.columns,
                           columns=['Feature Importance']).sort_values(by='Feature Importance')

plt.figure(figsize=(6, 80,), dpi=78)
sns.barplot(data=feature_imp, y='Feature Importance', x=X.columns)
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(24, 12))
plot_tree(DT, feature_names=X.columns, max_depth=4)
plt.show()

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(max_depth=6)
RFC.fit(X_train, y_train)
RFC_pred = RFC.predict(X_test)

cr = classification_report(y_test, RFC_pred)
print(cr)

cm = confusion_matrix(y_test, RFC_pred)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=RFC.classes_)
cmd.plot()
plt.show()

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

ADA_model = AdaBoostClassifier(n_estimators=100)
GB_model = GradientBoostingClassifier()
ADA_model.fit(X_train, y_train)
GB_model.fit(X_train, y_train)

ADA_predict = ADA_model.predict(X_test)
GB_predict = GB_model.predict(X_test)


cr = classification_report(y_test, ADA_predict)
print(cr)

cm = confusion_matrix(y_test, ADA_predict)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ADA_model.classes_)
cmd.plot()
plt.show()


cr = classification_report(y_test, GB_predict)
print(cr)

cm = confusion_matrix(y_test, GB_predict)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GB_model.classes_)
cmd.plot()
plt.show()