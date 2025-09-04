import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split,GridSearchCV



df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\rock_density_xray.csv")
df.columns = ['Signal','Density']

sns.scatterplot(df,x='Signal',y='Density')
plt.show()

#not scaling as it is simple data with similar magnitude

X = df['Signal'].values.reshape(-1,1)

# We would need to reshape when using a single feature because of whats happening in the background
# We need to specify because SKlearn gets confused if its a dataframe or a series. Meaning, single row or single feature
y = df['Density']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)


#Linear Regression

from  sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

LR = LinearRegression()
LR.fit(X_train,y_train)
LR_preds = LR.predict(X_test)
MAE = mean_absolute_error(y_test,LR_preds)
print(MAE)
# On an average, the prediction is off by '0.211198973318633' to the actual value.
# Y values rage between 1.3 to 2.8

RMSE = np.sqrt(mean_squared_error(y_test,LR_preds))
print(RMSE)

# IMPORTANT issue addressed here : So MAE or RMSE of 10% isnt that bad right but if you observe the model is just giving
# average, meaning the LR model is trying to fit a linear line
print(y_test,LR_preds) # There are many instances where the predictions is off by > 30 or 40 %

random_data = np.arange(1,200)
model_2 = LR.predict(random_data.reshape(-1,1))
print(model_2) # Even for signal strenght more than 100, model is still showing the rock density to be around 2.20


sns.scatterplot(df,x='Signal',y='Density')
plt.plot(random_data,model_2) # See the model is trying to fit a straght line even for higher frequency
plt.show()