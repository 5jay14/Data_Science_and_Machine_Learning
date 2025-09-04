'''
when we want plots categorised against numerical columns
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

tips = sns.load_dataset('tips')


'''
Bar plot = it is a general plot that allows to aggregate categorical data based off some function.
default function is Mean 
x is ccategory, y is the numerical column
estimator = assign a function for example  np.median,
We can pass any function
'''

#sns.barplot(x='sex',y='total_bill',data=tips)
# sns.barplot(x='sex',y='total_bill',data=tips, estimator=np.meadian)
#plt.show()



'''
Count plot = it is essentially the same as bar plot, except y axis is already chosen and it gives the count
Estimator is explicitly counting the number of occurrence

'''

#sns.countplot(x='sex',data=tips)
#plt.show()


'''
Box plot = aka box and whisker plot, shows the distribution of quantitative data in a way the facilitates the comparisons
between variables or across levels of a categorical variabe
box is box, whisker are the lines at the end of each side of the boxes

line inside box indicates median value
thw whiskers indicate the minimum and the maximum value depending upon the orientation
to get the range, subtract max - min
We can look at it as 4 quartile. each one having 25%

Steps arrang
 
x is categorical, y is numerical column the values in ascending order
years = 0,1,2,3,4,5,6,7,8,9,10,11,12,13
age = 3,4,5,9,7,5,8,11,2,12,10
age_asc = 2,3,4,5,5,7,8,9,10,11,12

Whiskers represent the range

Min(Whiskers A) = 2
Max(Whisker B) = 12
Range: 12-2 = 10 
Median = 7
Q1 or lower quartile: 2,3,4,5,5 = 4
Q3 or upper quartile: 8,9,10,11,12 = 10


Box will start at 4, line/median at 7, box ends at 10
Min to q1 is first quartile(25%)
q1 to mediaan is 2nd quartile 
median to q3 is 3rd qurtile
q3 to max is fourth quartile



can also add an additional categorical condition using hue
Need to understand the outliers
'''

sns.boxplot(x='day',y='total_bill',data=tips)
plt.show()

'''
Voilin plot = one of the strrength, its ability to dsiplay distribution of multiple categories side by side
Voilin plot: KDE is the outer layout and this shows the density of data at different values 
Width of the plot shows the density of data, wider section is higher density/more datapoints
It is typical symmetrical
inside we shall find the box plot
x is categorical, y is numerical column
it is a box and kde plot
box is the inner layer and kde is the outer plot

Can also pass hue
as a additional argument i can pass 'split = True'  what this does is, it seperates each half for the hue category
ex- in the sae violin plot, one half for male and other for female
'''
#sns.violinplot(x='smoker',y='total_bill',data=tips)
#plt.show()
sns.violinplot(x='smoker',y='total_bill',data=tips,hue='sex')
plt.show()
#sns.violinplot(x='smoker',y='total_bill',data=tips, hue='sex',split=True)
#plt.show()


'''
Strip plot = x is cat, y is numerical. it is essentialy a scattered plot based on category
disadvantage is we cant see many points are added on top of each other
jitter can help here
'''

sns.stripplot(x='day',y='total_bill',data=tips,jitter=True, hue='sex')
#plt.show()

'''
swarm plot = combination of strip and violin plot
this shows all the points
disadvantages = when the dataset increase, it is not advised to use this
'''


sns.swarmplot(x='day',y='total_bill',data=tips)
# sns.violinplot(x='day',y='total_bill',data=tips) #use this to get the violin plot along with swarm plot
#plt.show()