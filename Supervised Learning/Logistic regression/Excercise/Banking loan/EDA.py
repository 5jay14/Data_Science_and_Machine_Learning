import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\LR - Bank loan\loan-train.csv")

print(train_df.describe())
print(train_df.info())

# Checking for duplicate data using the loan ID which is unique
print(train_df['Loan_ID'].nunique() / len(train_df))  # No duplicates

# Count of missing data(rows)
print(train_df.isnull().sum())


# Missing data in %


# Understanding the gender distribution
sns.countplot(data=train_df, x=train_df['Gender'])
plt.show()

sns.boxplot(data=train_df, x='Gender', y='ApplicantIncome')
sns.scatterplot(data=train_df, x='Gender', y='ApplicantIncome')
plt.show()
