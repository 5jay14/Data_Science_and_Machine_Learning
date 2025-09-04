# Cohort: Segment wise analysis, in this dataset we will use contract and Tenure

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1st step is to load the dataframe and go through the data
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Telco-Customer-Churn.csv")
df = df.drop('customerID', axis=1)

non_cat = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
               'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
               'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

df1 = pd.get_dummies(data=df, columns=categorical, drop_first=True)  # for algorithms
df1 = df1.select_dtypes(include=['bool', 'float']).astype(int)


# Lets check the distribution of contract type
# sns.countplot(data=df, x='Contract', hue='Churn')
# plt.show()

# sns.histplot(data=df, x='tenure', bins=50)
# plt.show()
# Similar to countplot, we can see that the majority of the population is around 1 and 2 months

# Now use the seaborn documentation as a guide to create histograms separated by two additional features,
# Churn and Contract

# sns.displot(data=df, x='tenure', col='Contract', row='Churn', bins=60)
# plt.show()
# If you observe, there are customers who are on M2M contract staying for more than 5 months, we can reach out to those
# customers and offer them a yearly contract for a lower price

# Display a scatter plot of Total Charges versus Monthly Charges, and color hue by Churn.
# sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Churn', palette='Dark2', alpha=0.35)
# plt.show()
# Can see customers with high monthly charges are churning more

# lets create a graph showing the churn rate for each tenure
# for each month(Churn) = Yes/(Yes+No)

# Churn rate for each tenure
def churn_rate_by_tenure(df):
    yes_churn = df.groupby(by=['Churn', 'tenure']).count().transpose()['Yes']
    no_churn = df.groupby(by=['Churn', 'tenure']).count().transpose()['No']
    churn_rate = (100 * (yes_churn / (yes_churn + no_churn)))
    churn_rate = churn_rate.transpose()['gender'].plot()
    plt.title('Churn Rate')
    plt.xlabel('Tenure')
    plt.ylabel('Churn %')
    plt.show()
    print(churn_rate)


churn_rate_by_tenure(df)

'''
Based on the tenure column values, create a new column called Tenure Cohort that creates 4 separate categories
'0-12 Months'
'12-24 Months'
'24-48 Months'
'Over 48 Months'
'''


def cohort(tenure):
    if tenure < 13:
        return '0-12 Months'
    elif tenure < 25:
        return '12-24 Months'
    elif tenure < 48:
        return '24-48 Months'
    else:
        return 'Over 48 Months'


df['Cohort'] = df['tenure'].apply(cohort)

print(df[['tenure', 'Cohort']])

sns.scatterplot(data=df, x='MonthlyCharges', y='TotalCharges', hue='Cohort', palette='Dark2', alpha=0.35)
plt.show()

# Create a count plot showing the churn count per cohort.

sns.countplot(data=df, x='Cohort', hue='Churn')
plt.show()

# Create a grid of Count Plots showing counts per Tenure Cohort, separated out by contract type and colored by the
# Churn hue.

sns.catplot(data=df, x='Cohort', col='Contract', hue='Churn', kind='count')
plt.show()
