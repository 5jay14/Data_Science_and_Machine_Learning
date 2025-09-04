import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\hearing_test.csv")
print(df.head())

X = df.drop('test_result', axis=1)
y = df['test_result']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression()
logistic_model.fit(scaled_X_train, y_train)
print(logistic_model.coef_)  # [[-0.89962977  3.44877023]] Age and Physical score
# Age is negative, means as the age increase the odds of belonging to 1 class decrease
# PS is positive, means the physical score increases, the odds of belonging to class 1 increases


y_pred_class = logistic_model.predict(scaled_X_test)  # gives back class category
y_pred_prob = logistic_model.predict_proba(scaled_X_test)  # gives back the probability of belonging to the class
print(y_pred_class)
print(y_pred_prob)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score
# standard performance metrics
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
print(accuracy, precision, recall)

CM = confusion_matrix(y_test, y_pred_class)  # shows for the test numbers only
print(CM)
# gives a matrix TP, FN, TN, FP

# There is a built in plot that helps to visualise the confusion matrix, IT WILL ALSO DOES THE PREDICTION BASED ON THE
# SCALED DATA.  WE CAN ALSO SWAP THE MODEL WHILE KEEPING THE SAME FEATURES


from sklearn.metrics import ConfusionMatrixDisplay

# Plotting heatmap
cm = confusion_matrix(y_test, y_pred_class, labels=logistic_model.classes_)  # ,normalize='all', this will shot it in %
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logistic_model.classes_)
disp.plot()
plt.show()  # these are only test data numbers

# gives the report for recall, precision and f1 metrics
cr = classification_report(y_test, y_pred_class)
print(cr)

# if precision and recall values for the class matching up closely with accuracy,
# then the data is not relatively imbalanced

# Plotting ROC and precision recall curves

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

RocCurveDisplay.from_estimator(estimator=logistic_model, X=scaled_X_test, y=y_test)
plt.show()

PrecisionRecallDisplay.from_estimator(estimator=logistic_model, X=scaled_X_test, y=y_test)
plt.show()


# Grabbing the probability for specific rows
print(logistic_model.predict_proba(scaled_X_test)[0])
print(logistic_model.predict(scaled_X_test)[0])
print(y_test[0])
print()


