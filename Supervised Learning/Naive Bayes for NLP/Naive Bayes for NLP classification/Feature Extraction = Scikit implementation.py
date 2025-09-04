import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

text = ['this is a line', 'this is another line', 'completely different line']
vectorizer = CountVectorizer()  # (stop_words='english') # can also pass a list for custom words
sparse_matrix = vectorizer.fit_transform(text)
# each item in the list will be treated as separate document, python stores in a sparse matrix which is memory efficient
# use .todense() to store it a matrix form

from sklearn.feature_extraction.text import TfidfTransformer

# transforming into TF-IDF
tfidf = TfidfTransformer()
result = tfidf.fit_transform(sparse_matrix)  # BOW ==> TF IDF, stored in a sparse matrix form
print(result.todense())

# both in one step

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer()
tfidfv_results = tfidfv.fit_transform(text)
print(tfidfv_results.todense())
