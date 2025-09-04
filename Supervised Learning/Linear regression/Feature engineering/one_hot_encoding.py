import pandas as pd

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\wine_fraud.csv")
string_objects = df.select_dtypes(include='object')
numeric_objects = df.select_dtypes(exclude='object')
dummy_object_variables = pd.get_dummies(string_objects)
DF1 = pd.concat([dummy_object_variables, numeric_objects], axis=1)

# Converting the string to binary
df['quality'] = df['quality'].replace(to_replace='Legit', value=1)
df['quality'] = df['quality'].replace(to_replace='Fraud', value=0)


# String handling
# Converting the string variables in the label column to binary values
df['Fraud'] = df['quality'].map({'Legit': 0, 'Fraud': 1})

# Creating dummies for the string variables manually
# df['type'] = pd.get_dummies(df['type'],drop_first=True) # This is returning booleans
df['red_wine'] = df['type'].map({'red':1,'white':0})
df['white_wine'] = df['type'].map({'white':1,'red':0})
df = df.drop(['type','quality'],axis=1)