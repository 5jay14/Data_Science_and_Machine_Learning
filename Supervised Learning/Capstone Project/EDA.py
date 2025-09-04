import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1st step is to load the dataframe and go through the data
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Telco-Customer-Churn.csv")
df = df.drop('customerID', axis=1)
print(df.info())
print(df.describe())

# Check how balanced the target classes are:
# Classes are Imbalanced but have enough data in each class, we can use algorithms such as Decision tree and SVM to
# apply weight to each class samples to increase the importance of minority class during training
sns.countplot(data=df, x='Churn')
plt.show()

# Dealing with missing values : No missing values in any class
print(df.isnull().sum())

# Dealing with categorical values
# Majority of the classes seems to be Categorical, it would be easy to look for non Categorical columns
non_cat = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']

categorical1 = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

df1 = pd.get_dummies(data=df, columns=categorical, drop_first=True)  # for algorithms
df1 = df1.select_dtypes(include=['bool', 'float']).astype(int)

# Lets see the correlation between categorical columns and the class label
corr = pd.get_dummies(data=df, columns=categorical1).corr()  # for visualization
yes_churn = corr['Churn_Yes'].sort_values().iloc[1:-1]  # Removing churn yes and churn no which relation we already know
print(yes_churn)

plt.figure(figsize=(14, 7), dpi=100)
plt.xticks(rotation=90)
sns.barplot(x=yes_churn.index, y=yes_churn.values)
plt.title('Features correlation with Churning')
plt.show()
# We can see that customers with month to month contact has high churn and customers with 2 year contract has lowest churn rate

print(df1.info())

# check the distribution of monthly cost vs churn
sns.violinplot(x='Churn', y='MonthlyCharges', data=df)
plt.show()

# check the distribution of TotalCharges vs churn
sns.violinplot(x='Churn', y='TotalCharges', data=df)
plt.show()
# WHat is the story here,
# Month to Month contract needs deeper dive but we can understand that the customers choosing M2M contract come with a
# expectation of churning in few months

# One year and 2 year contact shows that the customer who churned have high total charges
# Businesses can run campaigns to these targeted users like offering some discount or coupon

print(df1['MonthlyCharges'].describe())
print(df1['TotalCharges'].describe())

# Showing the churn rate based on the contract type and total charges
# There are outliers in the month to month contract
#
sns.boxplot(data=df, x='Contract', y='TotalCharges', hue='Churn')
plt.show()

plt.figure(dpi=75)
sns.countplot(data=df,x='tenure',hue='Churn')
plt.show()