import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# SVC is for classification

# data used here, is a study on mouse which was infected with virus and given two medicines with different doses and
# checking 2 weeks later to see if the virus is still present

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\mouse_viral_study.csv")
sns.scatterplot(x='Med_1_mL', y='Med_2_mL', hue='Virus Present', data=df)
plt.show()  # This plot shows medicines with higher doses has no virus present


# using a custom function to visualize the margins and support vectors
def plot_svm_boundary(model, X, y):
    X = X.values
    y = y.values

    # Scatter Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='seismic')

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.show()


X = df.iloc[:, :2]
y = df.iloc[:, 2]

linear_model = SVC(C=0.05, kernel='linear')
linear_model.fit(X, y)
plot_svm_boundary(linear_model, X, y)

# C is a regularization parameter, it is just a measurement of how many points we are allowing inside the margin
# But C in Scikit is a inverse metric meaning, lower the C value, more point are allowed to be inside the margin
# Best C can be found using grid search and CV. Decreasing C corresponds to more regularization


rbf_model = SVC(kernel="rbf", C=10, gamma='scale')
# Original data set is transformed into higher dimension/larger feature space
# using kernel trick, but it is projected into 2d to visualize
# Gamma defines how much influence a single training example has, higher the gamma, model starts picking too much noise
# Larger the gamma, the closer the other examples points have to be to be effective

rbf_model.fit(X, y)
plot_svm_boundary(rbf_model, X, y)
"""
Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
if gamma='scale' (default) is passed then it uses 1 / (n_features * X. var()) as value of gamma,
if 'auto', uses 1 / n_features
if float, must be non-negative
"""

# Sigmoid and Poly Kernel
sigmoid_model = SVC(kernel='sigmoid')
sigmoid_model.fit(X, y)
plot_svm_boundary(sigmoid_model, X, y)  # not recommended for this dataset

poly_model = SVC(kernel='poly', degree=1)
# degree = 1 should show a graph similar to Linear kernel, as we increase the power of the degree, the plane starts to curve
poly_model.fit(X, y)
plot_svm_boundary(poly_model, X, y)

from sklearn.model_selection import GridSearchCV

base_svm_model = SVC()
param_grid = {'C': [0.01, 0.1, 1], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(base_svm_model, param_grid)
# Need to do the train test split for X and y
grid.fit(X, y)
print(grid.best_params_)
