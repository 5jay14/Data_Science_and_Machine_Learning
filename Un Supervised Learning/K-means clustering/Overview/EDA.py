import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

'''
Goal: When working with unsupervised learning methods, its usually important to lay out a general goal. In our 
case, let's attempt to find reasonable clusters of customers for marketing segmentation and study. What we end up 
doing with those clusters would depend heavily on the domain itself, in this case, marketing
'''
df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\bank-full.csv")

plt.figure(figsize=(12, 8), dpi=100)
sns.histplot(data=df, x='age', kde=True, hue='loan', bins=30)
plt.show()

sns.histplot(data=df[df['pdays'] != 999], x='pdays', hue='loan', bins=30)
plt.show()

sns.histplot(data=df, x='duration', hue='contact')
plt.xlim(0.0, 1000.0)
plt.show()

sns.countplot(data=df, x='job', order=df['job'].value_counts().index, hue='loan')
plt.show()
