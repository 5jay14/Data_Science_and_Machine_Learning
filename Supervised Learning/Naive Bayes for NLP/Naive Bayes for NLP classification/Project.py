import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\moviereviews.csv")

# 35 rows are missing the review but they are labelled, also there are some strings with whitespace

df = df[df['review'].str.strip().ne('')]
df = df.dropna()
print(df.isnull().sum())
print(df.isna().sum())
sns.countplot(df, x='label')
plt.show()

X = df['label']
y = df['review']


# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
#
# sns.countplot(data=df, x='label')
#
# pipe = Pipeline([('tfidf', TfidfVectorizer()),
#                  ('lsvc', LinearSVC())])
# pipe.fit(X_train, y_train)
#
# from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
#
# pred = pipe.predict(X_test)
# cr = classification_report(y_test, pred)
# print(cr)
# cm = confusion_matrix(y_test, pred)
# cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
# cmd.plot()
# plt.show()