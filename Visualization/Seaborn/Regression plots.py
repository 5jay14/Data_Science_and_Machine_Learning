'''
Seaborn has many built in capabilities for regression plots
it creates a regression line between two parameters and helps to visualise linear relationship between them
linear models

correlation = defines the strength of the relation between two variables
that is,  A&B = B&A
co = together, relation = connection

connection between 2 variables,  when the increase in variable results in increase another variable is positively
correlated. ex: higher the temp, high sale of icecream
negative correlation is when, increase in one results in decrease on another variable value
higher the price, lower the demand

ex: higher the bill, higher the tips

correlation coefficient = degree of association of two variables is measured by correlation coefficient
It is a measure of linear association. represented by 'r'

r varies from +1 through 0 to -1

r= +1 , positive/strong correlation, closer to 1 is better
r= -1,  negative correlation
r= 0, no correlation

Regression: when the two variables are related, change in the value of one variable will result in the value of another
variable

if  X = independent/explanatory variable
    Y = dependent/predictable variable
 their relation is called as regression of Y on X

ex: X= temp, Y=icecream sales
Regression equation: regression simpy means Average value of Y is a function of X


Correlation and regression are studied together

cor is to check the association between two variables, if yes then what is the strenght
reg = a functional relation is established so as to make future projections on event
'''

# LM plot - it allows to display linear model with seaborn

import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset('tips')
sns.lmplot(tips, x='total_bill', y='tip')
# sns.lmplot(tips,x='total_bill', y= 'tip',hue = 'sex',markers=['o','v']) # oneplot, data seperated by color
# sns.lmplot(tips,x='total_bill', y= 'tip',col = 'sex')# two plots seperated by columns
# sns.lmplot(tips,x='total_bill', y= 'tip',col = 'sex', row='time')# additional par
# sns.lmplot(tips,x='total_bill', y= 'tip',col = 'day', row='time',hue='sex') #additional par


# size and aspec

sns.lmplot(tips, x='total_bill', y='tip', aspect=0.6 )
plt.show()
