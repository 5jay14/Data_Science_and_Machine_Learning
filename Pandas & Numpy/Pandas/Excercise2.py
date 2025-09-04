import numpy as np
import pandas as pd
from re import findall
purchase = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\Py_DS_ML_Bootcamp-master\Refactored_Py_DS_ML_Bootcamp-master\04"
                       r"-Pandas-Exercises\Ecommerce Purchases.csv")
purchase.columns

# How many rows and columns are there?
# purchase.info()

# What is the average Purchase Price?
purchase['Purchase Price'].mean()

# What were the highest and lowest purchase prices?
purchase['Purchase Price'].max()
purchase['Purchase Price'].min()


# How many people have English 'en' as their Language of choice on the website?
#1st method
purchase[purchase['Language']=='en'].count()

#2nd metthod using custom function
def en_string(lang):
    if 'en' in lang.lower():
        return True
    else:
        return False


sum(purchase['Language'].apply(lambda x: en_string(x)))

# How many people have the job title of "Lawyer"
purchase[purchase['Job'] == 'Lawyer'].count()

# How many people made the purchase during the AM and how many people made the purchase during PM
purchase['AM or PM'].value_counts()

# What are the 5 most common Job Titles?
purchase['Job'].value_counts().head()

# Someone made a purchase that came from Lot: "90 WT" , what was the Purchase Price for this transaction?
purchase[purchase['Lot'] == '90 WT']['Purchase Price']

# What is the email of the person with the following Credit Card Number: 4926535242672853
purchase[purchase['Credit Card'] == 4926535242672853]['Email']


# How many people have American Express as their Credit Card Provider and made a purchase above $95
purchase[(purchase['CC Provider']=='American Express') & (purchase['Purchase Price']>95)].count()['CC Provider']


# How many people have a credit card that expires in 2025
purchase[purchase['CC Exp Date'].apply(lambda x: x[3:]=='25')]['CC Exp Date'].value_counts() # This gives month wise
purchase[purchase['CC Exp Date'].apply(lambda x: x[3:]=='25')]['CC Exp Date'].count()  # overall

# What are the top 5 most popular email providers/hosts (e.g. gmail.com, yahoo.com, etc...) **

print(purchase['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5))