#  These DV are built of off MPL, but they allow to call DV directly out of dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"C:\Users\vijay\Desktop\df1.csv", index_col=0)
df2 = pd.read_csv(r"C:\Users\vijay\Desktop\df2.csv")
df3 = pd.read_csv(r"C:\Users\vijay\Desktop\df3.csv")

print(df1.head())

# Observe now, calling the MPL plot directly from dataframe, can also pass seaborn styles
# Pretty much all the plots can be done based off the dataframe

df1['A'].hist(bins=30)
#plt.show()

# another way
df1['A'].plot(kind='hist', bins=30)
#plt.show()

# another way
df1['A'].plot.hist()
#plt.show()

# examples
#--------------------------------------------------
# bar plots : takes index value/comuns as the category and the series/rows as the x value
df2.plot.bar()
df2.plot.bar(stacked=True)  #
#plt.show()


#--------------------------------------------------
# ex2
df1.plot.scatter(x='A',y='B',c= 'C', )
# passing three parameters,
# there will be color pallette on the right to differentiate oor 3dimensional plot
plt.show()

# if we need to show them by size instead of color
#instead c, it will be s and data frame column
# multyping to see the large size
df1.plot.scatter(x='A',y='B',s= df1['C']*100)

plt.show()
#---------------------------------------------------

#box plot
df1.plot.box()
plt.show()

#hexbin = essesntially like a scatter pot but showed in hexagonals
datasett = pd.DataFrame(np.random.randn(1000,2), columns=['a','b'])
datasett.plot.hexbin(x='a',y='b',gridsize = 20)
plt.show()

#KDE
df1.plot.kde()
df1['B'].plot.kde() #or .density()
plt.show()