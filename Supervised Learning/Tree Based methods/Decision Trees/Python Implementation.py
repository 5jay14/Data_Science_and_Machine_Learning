import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\penguins_size.csv")

df = df.dropna()
print(df)
print(df.isnull().sum())
print(df['sex'].unique())  # there is one row with 3rd category '.', delete it or fill it with reasonable assumption
df = df[df['sex'] != '.']
# df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()  # data leans towards female
# df = df.at[336, 'sex'] == 'Female'  # reassignment of value, this creates a copy and does not change the original data frame

#sns.pairplot(df, hue='species')
#plt.show()

#sns.catplot(df, x='species',y = 'culmen_length_mm',kind='box', col='sex')
#plt.show()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

X = pd.get_dummies(df.drop('species', axis=1), drop_first=True)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_model = DecisionTreeClassifier()
base_model.fit(X_train, y_train)

y_pred = base_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=base_model.classes_)
cmd.plot()
plt.show()

# Lot of Important attributes
print(base_model.feature_importances_)
fea_impo = pd.DataFrame(index=X.columns, data=base_model.feature_importances_,
                        columns=['Feature_importance']).sort_values(
    "Feature_importance")
print(fea_impo)
# This shows the importance of each feature in the decision making and it is listed in the same order as X columns
# Now you may observe that the weight of the penguin is not considering as the important, since the para is to default

plt.figure(figsize=(8, 8), dpi=200)
plot_tree(base_model, feature_names=X.columns, filled=True)
plt.show()
# This model is overfitting, to the last leaf

"""
Samples = Numbers of samples considered during that split
if you observe first spllit has 233 samples, length of the column is 233

Value array = it simply is a label, so there are 3 lable in the array in the flipper_length_mm
3 label = 3 species (y classes)
"""


def model_func(model):
    y = model.predict(X_test)
    print(classification_report(y_test, y))
    print('\n')
    plt.figure(figsize=(8, 8), dpi=100)
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.show()


model1 = DecisionTreeClassifier(max_depth=2)
model1.fit(X_train, y_train)

model_func(model1)
# Splitting method is default to GI, another available criterion is 'Entropy'
# Default splitter is based on the best feature selected by criterion
# max depth is set to max by default


pruned_tree = DecisionTreeClassifier(max_depth=3)
model_func(pruned_tree)

#Different mathematical criterion for spit, Information gain AKA entropy

entropy = DecisionTreeClassifier(criterion='entropy')
entropy.fit(X_train, y_train)

model_func(entropy)