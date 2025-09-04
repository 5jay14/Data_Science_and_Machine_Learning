import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Gene expression levels are calculated by the ratio between the expression of the target gene(gene of interests) and
# the expression of one or more reference genes


df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\gene_expression.csv")
print(df.head())
df.isnull().sum() # no nulls
print(df['Cancer Present'].value_counts()) # data set is balanced

sns.scatterplot(data=df, x='Gene One', y='Gene Two', hue='Cancer Present',alpha = 0.5, style='Cancer Present')
# plt.xlim(2,6) #setting the limit
# plt.ylim(4,8) #setting the limits
plt.show( )

X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
# there are other KNN algos for regression and transformed
knn_model = KNeighborsClassifier(n_neighbors=1)
knn_model.fit(X_train_scaled,y_train)


y_pred = knn_model.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CM = confusion_matrix(y_test,y_pred)
print(CM)

CR = classification_report(y_test,y_pred)
print(CR)


## Elbow Method for Choosing Reasonable K Values
## NOTE: This uses the test set for the hyperparameter selection of K

# Finding the accuracy for different K
# What is the ideal K value?
# usually lower k value will have high error and it drops as we increase k value.
# eventually the it will try to level off meaning off, increasing k will not resukt in better error
# We have to consider the optimum k value while also keeping in mind the complexity that comes with higher k value

error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train_scaled, y_train)

    y_pred = knn_model.predict(X_test_scaled)
    test_error = (1 - accuracy_score(y_test,y_pred))
    error_rates.append(test_error)

plt.plot(range(1,30),error_rates)
plt.ylabel('Error')
plt.xlabel('K Value')
plt.show() # We are not really improving the error rate after 6 neighbors





