import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import xticks


load_data = pd.read_csv(r"C:\Users\vijay\Data Science and Machine Learning\DATA\insurance.csv")
print(load_data.isnull().sum())

# 1338 Entries with 7 total columns  Dont see any missing values


print(load_data.describe())

# There are 3 categorical features: Sex, Smoker & Region

numerical_cols = ['age', 'bmi', 'children', 'charges']
corr = load_data[numerical_cols].corr()
sns.heatmap(corr,annot=True)
plt.title("Correlation Heatmap of Numerical Features")
plt.show() # Age shows high correlation with the target variable


sns.histplot(load_data['age'], stat='density', kde=True, bins=47)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.show()

sns.scatterplot(x='age', y='charges', data=load_data)
plt.title('Age vs Charges')
plt.show()

