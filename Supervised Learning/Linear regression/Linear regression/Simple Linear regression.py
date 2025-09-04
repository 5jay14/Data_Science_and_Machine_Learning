import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
Create a best fit line to map out to a linear relationship between total advertising spend and resulting sales
spend is the independent variable
resulting sales is the dependent variable
'''
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\08-Linear-Regression-Models\Advertising.csv")
# amounts are in Thousand dollars
# There are three features, create a new feature that is sum of all the three features in oder to apply OLS

df['total_spend'] = df['TV'] + df['radio'] + df['newspaper']

sns.scatterplot(data=df, x='total_spend', y='sales')
plt.show()

# seaborn has inbuilt plot called regplot that finds the best fit by applying ols which is y= mx+b
sns.regplot(data=df,x='total_spend', y='sales')
plt.show()


'''
Manually creating the best fit using polynomial fit
y = mx+b
y = B1x+B0

What polynomial fit does is, it returns the beta coefficients when we provide X and Y
beta coefficients = B1 and B0
'''
X= df['total_spend']
y = df['sales']

# Use a variable to store the beta coefficients
BCE = np.polyfit(X,y,deg=1)
print(BCE)

# Estimating sales for potential spend
potential_spend = np.linspace(0,500,100)


predicting_sales = potential_spend * BCE[0] + BCE[1] #y = B1x+B0

sns.scatterplot(data=df, x='total_spend', y = 'sales')
plt.plot(potential_spend,predicting_sales, color ='red')
plt.show()

#SNS.regplot all the above steps in a single line


