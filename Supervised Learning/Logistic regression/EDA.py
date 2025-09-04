import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
An experiment was conducted on 5000 participants to study the effects of age and physical health on hearing loss, 
specifically the ability to hear high pitched tones. This data displays the result of the study in which participants 
were evaluated and scored for physical ability and then had to take an audio test (pass/no pass) which evaluated their 
ability to hear high frequencies. The age of the user was also noted. Is it possible to build a model that would predict
 someone's likelihood to hear the high frequency sound based solely on their features (age and physical score)?

Features
age - Age of participant in years
physical_score - Score achieved during physical exam
Label/Target
    test_result - 0 if no pass, 1 if test passed
'''
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\hearing_test.csv")
print(df.head())
print(df.describe())
print(df['test_result'].value_counts())  # Count of each unique values
sns.countplot(data=df,x='test_result')
plt.show()
print(df['test_result'].nunique())  # how many values are unique
print(df['test_result'].unique())  # will give a list of unique values


plt.show()

sns.scatterplot(data=df, x='age', y='physical_score',hue='test_result',alpha =0.6)
plt.show()
# This shows some linear relation between age and physical score, lower the age the higher the physical score is
# there is some noise however

sns.boxplot(data=df,x='test_result',y='age') # This shows that people who pass the test are relatively low age
plt.show()

sns.boxplot(data=df,y='physical_score',x='test_result') # people who pass the t
plt.show()

sns.heatmap(df.corr(),annot=True)
plt.show()