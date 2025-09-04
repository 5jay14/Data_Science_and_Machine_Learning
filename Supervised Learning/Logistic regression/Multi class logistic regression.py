import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# In this project, we will predict for the multiclass dataset using Grid search cv for finding the optimum hyperparameter
# evaluating the performance


df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\iris.csv")
print(df.head())
X = df.drop('species', axis=1)
y = df['species']

# We are guessing what type of species the flower is, there are three type of  flowers and label is a string

print(df.info())
print(df.describe())

print(df['species'].value_counts())  # balanced instances of each classes dataset
sns.countplot(data=df, x='species')
plt.show()  # balanced instances of each classes dataset

sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
plt.show()

sns.heatmap(X.corr(), annot=True)  # high correlation between petal and petal width
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
scaler = StandardScaler()
scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV  # we can also import CV class of logistic regression

model = LogisticRegression(solver='saga', multi_class='ovr', max_iter=5000)
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
# solver is the optimization algorithm like gradient descent which tries to reduce the cost function to minimum
# penalty = default l2 multiclass = ovr = one versus rest, we know that logistic function(sigmoid function) tries to
# classify the object between 1 and 0 but for if the dataset has more than 2 labels, it separates one class from all
# the other and it does for the other class


penalty = ['l1', 'l2', 'elasticnet']
l1_ratio = np.linspace(0, 1, 20)
C = np.linspace(0, 1, 20)  # lambda/alpha

# keys need to match what is expected in the Logistic Regressio class
param_grid = {'penalty': penalty,
              'l1_ratio': l1_ratio,
              'C': C}

grid_model = GridSearchCV(model, param_grid=param_grid)
grid_model.fit(scaled_x_train, y_train)
print(grid_model.best_params_)

y_pred = grid_model.predict(scaled_x_test)
print(y_pred)

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

accu_score = accuracy_score(y_test, y_pred)
print(accu_score)

"""
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
"""

print(classification_report(y_test, y_pred))

# Built in ROC curve of scikit learn does not automatically work with multiclass situation since it is based off of binary classification

# Plotting ROC curve for multilable problems
from sklearn.metrics import roc_curve, auc
def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5, 5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()

plot_multiclass_roc(grid_model,scaled_x_test,y_test,n_classes=3)