import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\airline_tweets.csv")

'''
A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from February of 
2015 and contributors were asked to first classify positive, negative, and neutral tweets, followed by categorizing 
negative reasons (such as "late flight" or "rude service").

The Goal: Create a Machine Learning Algorithm that can predict if a tweet is positive, neutral, or negative. In the 
future we could use such an algorithm to automatically read and flag tweets for an airline for a customer service 
agent to reach out to contact
'''

# sns.countplot(data=df, x='airline_sentiment')
plt.show()

# We are focusing on the negative tweets and ignore the positives and neutral for this model

# sns.countplot(data=df, x='airline', hue='airline_sentiment')
plt.show()

# sns.countplot(data=df, x='negativereason')
# plt.xticks(rotation=90)
plt.show()

from sklearn.model_selection import train_test_split

data = df[['airline_sentiment', 'text']]
X = data['text']
y = data['airline_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_tfidf, y_train)

from sklearn.svm import SVC, LinearSVC

rbf = SVC()
rbf.fit(X_train_tfidf, y_train)

lsvc = LinearSVC()
lsvc.fit(X_train_tfidf, y_train)

from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix


def func_model(model):
    pred = model.predict(X_test_tfidf)
    cr = classification_report(y_test, pred)
    print(cr)
    cm = confusion_matrix(y_test, pred)
    cmd = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cmd.plot()
    plt.show()


#func_model(nb)
#func_model(rbf)
#func_model(lsvc)
#func_model(lr)
# linear svc seems to be giving better results, we can run gridsearch cv to tune the hyperparameters for all the models
#

from sklearn.pipeline import Pipeline

# Creating a pipeline, model deployment
pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('SVC', LinearSVC())])
pipe.fit(X, y)

print(pipe.predict(['not so bad not so good']))

