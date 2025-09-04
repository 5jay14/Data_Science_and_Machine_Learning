import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor



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

n_values =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
for i in n_values:
    model = KNeighborsRegressor(n_neighbors=i)
    run_model(model,X_train, X_test, y_train, y_test)

# if you observe, after certain k neaighbors, there is no change in the variance and also for the starting and ending points
# we can see the line flattening out meaning it is adding more bias there

