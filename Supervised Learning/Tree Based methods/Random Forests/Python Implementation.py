# using the same penguin dataset used in decision tree
import matplotlib.pyplot as plt
import  pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\penguins_size.csv")

df = df.dropna()
print(df.head(2))

X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.ensemble import RandomForestClassifier

RFC_model = RandomForestClassifier(n_estimators=10,random_state=101)
# every tree is split on some feature and there is randomness each time we run the model, so setting the state is a good
# idea as same split is considered every time and also can be used to evaluate between multiple models
RFC_model.fit(X_train,y_train)

preds = RFC_model.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report,ConfusionMatrixDisplay
report = classification_report(y_test,preds)
print(report)
cm = confusion_matrix(y_test,preds)
cmd = ConfusionMatrixDisplay(cm,display_labels=RFC_model.classes_)
cmd.plot()
plt.show()