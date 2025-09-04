import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\CIA_Country_Facts.csv")
df1 = df.iloc[:, 2:]
print(df.info())
print(df.describe())

# TASK: Create a histogram of the Population column. You should notice the histogram is skewed due to a few large
# countries, reset the X axis to only show countries with less than 0.5 billion people

# sns.histplot(df[df['Population'] <= 500000000], x='Population', )
# plt.show(block=False)
#
# df.isnull().sum().sort_values().plot(kind='bar')
# plt.show(block=False)

# Now let's explore GDP and Regions. Create a bar chart showing the mean GDP per Capita per region
# (recall the black bar represents std).

# plt.figure(figsize=(15, 6), dpi=80)
# sns.barplot(data=df, x='Region', y='GDP ($ per capita)', estimator=np.median)
# plt.xticks(rotation=45)
# plt.show(block=False)

# Create a scatterplot showing the relationship between Phones per 1000 people and the GDP per Capita.
# Color these points by Region

# sns.scatterplot(data=df,y='Phones (per 1000)',x='GDP ($ per capita)',hue='Region')
# plt.show()

# Create a Heatmap of the Correlation between columns in the DataFrame
sns.heatmap(df1.corr())
plt.show()


# Seaborn can auto perform hierarchal clustering through the clustermap() function. Create a clustermap of the
# correlations between each column with this function.

sns.clustermap(df1.corr())
plt.show()