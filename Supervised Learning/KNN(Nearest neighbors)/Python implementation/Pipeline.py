
# Pipeline = object in the sk learn which can set up a sequence of repeated operaations such as a scaler and a model
# this way only the pipeline needs to be called instead of having to call both scaler and model
# We need to match the exact variable names or string code, so need to be careful


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\gene_expression.csv")
X = df.drop('Cancer Present',axis=1)
y = df['Cancer Present']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
knn = KNeighborsClassifier() # All the parameters set to default
print(knn.get_params()) # get_params().keys()) for only keys

'''
When you use the StandardScaler as a step inside a Pipeline then scikit-learn will internally do the job for you.

What happens can be described as follows:

Step 0: The data are split into TRAINING data and TEST data according to the cv parameter that you specified in the GridSearchCV.
Step 1: the scaler is fitted on the TRAINING data
Step 2: the scaler transforms TRAINING data
Step 3: the models are fitted/trained using the transformed TRAINING data
Step 4: the scaler is used to transform the TEST data
Step 5: the trained models predict using the transformed TEST data'''

# 1. Setting up operations with a list of tuple pairs
operations = [('scaler',scaler),('knn',knn)] # it should be in the same order of building model

# 2. Creating pipeline
from  sklearn.pipeline import Pipeline
pipe = Pipeline(operations)

# 3. Grid search for best K values
from sklearn.model_selection import GridSearchCV
k_values = list(range(1,30))

#4. Creating param grid, we need to match the string names
# if the param grid is going inside the pipeline, we need to specify the parameters in the following manner
'''
chosen_string_name + two underscores + parameter key name
model_name + __ + parameter name
knn_model + __ + n_neighbors
knn_model__n_neighbors

The reason we have to do this is because it let's scikit-learn know what operation in the pipeline these parameters are
related to (otherwise it might think n_neighbors was a parameter in the scaler).
'''
param_grid = {'knn__n_neighbors':k_values}

full_cv = GridSearchCV(pipe,param_grid,cv=5,scoring='accuracy')
# Use full X and y if you DON'T want a hold-out test set
# Use X_train and y_train if you DO want a holdout test set (X_test,y_test)
full_cv.fit(X_train,y_train) # observe, we are passing the unscaled version

print(full_cv.best_estimator_.get_params()) # observe that there are parameters for scaler and KNN


full_prediction = full_cv.predict(X_test)
print(classification_report(y_test,full_prediction))