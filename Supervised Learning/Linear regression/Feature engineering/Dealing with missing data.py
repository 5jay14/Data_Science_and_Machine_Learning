import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Data should have information about each features.
1. Name of the feature
2. Data type of the feature
3. If the feature is categorical, details about each category
"""
# below function opens the file(notes) in read mode and closes the file automatically

with open("C:/Users/vijay/Desktop/DS ML/DATA - Copy/Ames_Housing_Feature_Description.txt", 'r') as f:
    print(f.read())

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\Ames_outliers_removed.csv")
# print(df.info())  # shows the details
# print(df.head())

# PID is a unique id, we do not need it as it does not bear any value. Index will take its place
df = df.drop("PID", axis=1)


# print(len(df.columns))


# print(df.isnull())        # Knowing what columns are null
# print(df.isnull().sum())  # replaces false with 0 and true with 1 and then it will return a series with
# number of rows that are missing data

# Couple of ways to extract the only true features
# print(df.isnull().sum() > 0)

# Checking the percentage of data that is missing
# print(100 * df.isnull().sum() / len(df))


def missing_percent(df):
    percent_nan = 100 * df.isnull().sum() / len(df)
    percent_nan = percent_nan[percent_nan > 0].sort_values()
    return percent_nan


percent_nan1 = missing_percent(df)
# print(percent_nan1)

sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.show()

# There are features where the number of rows missing data is significantly low, we should decide whether to drop
# or fill them with reasonable assumption based on the domain knowledge

sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.ylim(0, 1)  # magnifying to show the values between 0 to 1 percent
plt.show()

print(percent_nan1[percent_nan1 < 1])
# observe, there are some where some features are missing. which are less than 1%, not a bad idea to drop them
# identify if there are any rows missing multiple features

print(df[df['Electrical'].isnull()]['Garage Area']),
# this shows that while electrical is null, it contains data for garage area


df = df.dropna(axis=0, subset=['Electrical', 'Garage Cars'])
# this may not only drop the rows missing these two features data, it may also take away some of the missing features
# if it happens to be in the same row

percent_nan1 = missing_percent(df)
print(percent_nan1[percent_nan1 < 1])
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.ylim(0, 1)  # magnifying to show the values between 0 to 1 percent
plt.show()

# Also observe that most of the basement feature is missing for few rows, look at the features description to understand
# what 'NA' actually means, NA could mean not the data is available, it could be that the house does not have the
# basement

print(df[df['Bsmt Half Bath'].isnull()])
print(df[df['Bsmt Full Bath'].isnull()])
print(df[df['Bsmt Unf SF'].isnull()])
# Observe the row with index 1341 is missing all these bsmt features, we could simply replace it or delete Basement
# features with dtype numerics and having null/na values, replace with 0, while the string categorical features can be
# replaced with  'None'


# BSMT-> fill with 'None', create a list with features name and use it with original DF to choose the required columns
bsmt_cate_cols = ['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']
df[bsmt_cate_cols] = df[bsmt_cate_cols].fillna('None')

# BSMT Numerics --> fill with 0
bsmt_num_cols = ['BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Bsmt Full Bath', 'Bsmt Half Bath']
df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)

df['Mas Vnr Type'] = df['Mas Vnr Type'].fillna('None')
df['Mas Vnr Area'] = df['Mas Vnr Area'].fillna(0)

# <1% of the missing data is handled now
percent_nan1 = missing_percent(df)
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)

plt.show()

# Dealing with more than 1% of missing data
# Two main approaches here
# 1.Drop the missing columns
# 2.Fill the missing rows

'''
Dropping the feature column
    Easy to drop
    No longer need to worry about the feature in future
    Potential to lose a feature with possible important signal - Model will be trained without that feature 
    Consider to drop column when many rows are NAN

Filling the rows
    Potentially changing the ground truth in data
    Must decide on reasonable estimation based of domain knowledge 
    Must apply transformation to all the future data for predictions, meaning future data may also be missing
    the feature data for some rows. So we must fill it in by not only saving model but also Functions to handle the 
    missing values
    
    1. Simplest case of filling data: NaN can be replaced with reasonable assumption(Eg: 0 if assumed NaN implied zero)
    2. Harder cases: 
        A. Must use statistical methods based on other columns to fill NaN Values(In this case, filling the column 
        can be treated as a Label itself. 
        B. Use reasonable intuition: Ex, Data set is missing age data for some rows. We could use current career/education status
           to fill in the data(Ex, people currently in college fill in with 20 yrs)
'''

# Now look at the graph, Data  we can see same set or equal rows missing garage data. In the description file,
# No data means no garage. So we can simply replace the null values with string None, since that feature is categorical

garage_string = ['Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[garage_string] = df[garage_string].fillna('None')
percent_nan1 = missing_percent(df)
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.show()

# Replacing the garage year, can be based out of domain knowledge but now i am just simply going to use the average
print(df['Garage Yr Blt'].mean())  # 1978
df['Garage Yr Blt'] = df['Garage Yr Blt'].fillna(1978)

percent_nan1 = missing_percent(df)
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.show()

# Dropping the columns where >=75 % data is missing
df = df.drop(['Pool QC', 'Misc Feature', 'Alley', 'Fence'], axis=1)
percent_nan1 = missing_percent(df)
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.show()

# Now there are two features now where we can not drop rows or columns because not enough data is missing to drop the
# columns or not too little missing to drop the rows

# Feature Fireplace Qu is a categorical column, fill blanks with None

df['Fireplace Qu'] = df['Fireplace Qu'].fillna('None')

# lot frontage is numeric and blanks are mentioned as NaN, and with the domain knowledge we can fill in these values
# based on different column value. Column reference used is 'Neighborhood'

print(df['Lot Frontage'].head())
print(df['Neighborhood'].head())
print(df.groupby('Neighborhood')['Lot Frontage'].mean())

# Fill the NaN values of lot frontage with the mean value of neighborhood
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda value: value.fillna(value.mean()))
# we may still have some rows having NaN values where the Neighborhood is unique(one entry with NaN)
print(df.isnull().sum())  # fill it with 0
df['Lot Frontage'] = df['Lot Frontage'].fillna(0)

percent_nan1 = missing_percent(df)
sns.barplot(x=percent_nan1.index, y=percent_nan1)
plt.xticks(rotation=90)
plt.show()  # this will throw an error with empty sequence because there is no missing column data
