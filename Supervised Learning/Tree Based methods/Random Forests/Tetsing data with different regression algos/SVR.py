import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

# For SVM's we need to do the Grid CV to find the

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\rock_density_xray.csv")
df.columns = ['Signal','Density']
X = df['Signal'].values.reshape(-1,1)
y = df['Density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

def run_model(model, X_train, X_test, y_train, y_test):
    model = model.fit(X_train,y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test,preds)
    rmse = np.sqrt(mean_squared_error(y_test,preds))
    print(mae)
    print(rmse)

    signal_range = np.arange(1,100)
    signal_pred=model.predict(signal_range.reshape(-1,1))

    sns.scatterplot(df, x='Signal', y='Density',color='red')
    plt.plot(signal_range, signal_pred)  # See the model is trying to fit a straght line even for higher frequency
    plt.show()

svr = SVR()
param_grid = {'C':[0.01,0.1,1,5,10,100,100],
              'gamma': ['auto','scale']}
grid = GridSearchCV(estimator=svr,param_grid=param_grid)

run_model(grid,X_train, X_test, y_train, y_test)
# This model is trying to not adjust for the variance and not picking too much noise