import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from  sklearn.linear_model import LinearRegression

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

model = LinearRegression()
run_model(model, X_train, X_test, y_train, y_test)


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

pipe = make_pipeline(PolynomialFeatures(degree=5),LinearRegression())
run_model(pipe,X_train, X_test, y_train, y_test)

# Observe, degree like 10 is picking up too much noise thus adding variance to the training data. Higher degree like 25 is also
# picking to much noise at the end
# Degree of 5 seems to be fitting nicely and the errors are reduced to half but we dont know yet how the model will performa
# for the data which exceeds 200signal strength, increasing the np range to 200 showed model performing poorly for any degree
