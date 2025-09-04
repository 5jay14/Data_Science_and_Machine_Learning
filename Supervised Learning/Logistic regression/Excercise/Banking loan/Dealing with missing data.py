import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\LR - Bank loan\loan-train.csv")
train_df = pd.DataFrame(train_df)

# Dealing with missing 'Married' which is less than 0.5%
# There are 3 rows with no values in Married, also when married has no value. Dependents are 0
# print(train_df[train_df['Married'].isnull()].values)
subset = ['Married']
train_df = train_df.dropna(axis=0,subset=subset)

# Dealing with null Dependents
# Filling the Dependents 1 when Married is yes
mask = (train_df['Married'] == 'Yes') & train_df['Dependents'].isna()
train_df.loc[mask, 'Dependents'] = 1

# Now the missing value is  < 1, Dropping the Dependents with null values
subset = ['Dependents']
train_df = train_df.dropna(axis=0,subset=subset)

#  Dealing with null genders
subset = ['Gender']
train_df = train_df.dropna(axis=0,subset=subset)

# Dealing with Null loan amount term
print(train_df['Loan_Amount_Term'].mean()) #341.8965517241379
train_df['Loan_Amount_Term'] = train_df['Loan_Amount_Term'].fillna(342)


# Dealing with null Loan Amount
df = ['LoanAmount','Loan_Amount_Term','ApplicantIncome','CoapplicantIncome']
df = train_df[df]
print(df.head())
sns.heatmap(df.corr(), annot=True)
plt.show()


# Treating null credit history as no credit history
# train_df['Credit_History'] = train_df['Credit_History'].fillna(0)



def missing_percent(df):
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan


missing = missing_percent(train_df)
sns.barplot(x=missing.index, y=missing)
# plt.xticks(rotation=90) # rotating the category name
# plt.ylim(0, 10)  # magnifying to show the values between 0 to 10 percent
plt.show()