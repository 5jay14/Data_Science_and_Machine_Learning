import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

df = pd.read_csv(r"C:\Users\vijay\Downloads\911.csv")

# What are the top 5 zipcodes for 911 calls?
print(df['zip'].value_counts().head())

# What are the top 5 townships (twp) for 911 calls?
print(df['twp'].value_counts().head())

# Take a look at the 'title' column, how many unique title codes are there?
print(df['title'].nunique())

'''In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic.
Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.
For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS'''

df['reason'] = df['title'].apply(lambda x: x.split(':')[0])

# What is the most common Reason for a 911 call based off of this new column?
print(df['reason'].value_counts().head(3))

# Now use seaborn to create a countplot of 911 calls by Reason.

print(sns.countplot(x='reason', data=df, palette='viridis'))
plt.show()

# What is the data type of the objects in the timeStamp column?
print(type(df['timeStamp'].iloc[0]))

# Use pd.to_datetime to convert the column from strings to DateTime objects
df['timeStamp'] = pd.to_datetime(df['timeStamp'], format='%d-%m-%Y %H:%M')
time = df['timeStamp'].iloc[0]

'''
Now that the timestamp column are actually DateTime objects, 
use .apply() to create 3 new columns called Hour, Month, and Day of Week. 
You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.
'''
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)
df['day'] = df['timeStamp'].apply(lambda x: x.day)
df['dayofweek'] = df['timeStamp'].apply(lambda x: x.dayofweek)
df['Month'] = df['timeStamp'].apply(lambda x: x.month)
df['year'] = df['timeStamp'].apply(lambda x: x.year)



'''
Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual 
string names to the day of the week: **
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
'''

dmap = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_of_week'] = df['dayofweek'].map(dmap)


bymonth = df.groupby('Month').count()

# Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column
sns.countplot(df,x=df['day_of_week'], hue=df['reason'])
plt.legend(bbox_to_anchor=(1.05,1), loc =2, borderaxespad=0.)# use this code to have legend outside of the plot
plt.show()

# now for the month
sns.countplot(df,x=df['Month'], hue=df['reason'])
plt.show()

'''
Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month.
Keep in mind you may need to reset the index to a column
'''
sns.lmplot(x='Month',y='lat',data=bymonth.reset_index())
#shaded area indicates error, error grows where there is no data


plt.show()
# Create a new column called 'Date' that contains the date from the timeStamp column.
# You'll need to use apply along with the .date() method

df['Date'] = df['timeStamp'].apply(lambda x: x.date())

# Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls
df.groupby('Date').count()['lat'].plot()
plt.show()

#Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call*
df[df['reason']=='Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Traffic')
plt.show()
df[df['reason']=='EMS'].groupby('Date').count()['lat'].plot()
plt.title('EMS')
plt.show()
df[df['reason']=='Fire'].groupby('Date').count()['lat'].plot()
plt.title('Fire')
plt.show()

'''
Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so 
that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, 
but I would recommend trying to combine groupby with an unstack method. Reference the solutions if you get stuck on this
'''

# unstack method = Grouping by multiple columns(multi level indexing) and unstack one of them to columns and index
a = df.groupby(by=['day_of_week','Hour']).count()['reason'].unstack()
sns.heatmap(a,cmap='coolwarm')
plt.show()

sns.clustermap(a, cmap='coolwarm')
plt.show()


# Now repeat these same plots and operations, for a DataFrame that shows the Month as the column
b= df.groupby(by=['day_of_week','Month']).count()['lat'].unstack()
sns.heatmap(b,cmap='coolwarm')
plt.show()

#df1 = df.to_csv("C:/Users/vijay/Downloads/copy_911.csv", index=False)