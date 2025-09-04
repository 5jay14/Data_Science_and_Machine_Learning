import numpy as np
import pandas as pd

# Pandas series is built on top of Numpy array objects. It uses index referencing to work on data
# What differentiates series against np arrays, Series can have access label, meaning the values can be indexed by a label
# Series also works with normal python lists and numpy arrays

labels = ['a', 'b', 'c', 'd',]  # python list
my_data = [1, 2, 3, 4]  # python list
dicta = {'a': 'Australia', 'b': 2, 'c': 3.00, 'd': [0,1,2,3]}  # Python Dictionary
arr = np.array(my_data)  # Np array
arr1 = np.array(labels)
print(pd.Series(labels))  # Indexing reference is on the left side, its like a roll number against the name of a person
#it uses integers starting from 0 if we donnt pass explicit labels

# We can also use other objects for indexing
print(pd.Series(data = my_data,index=labels)) # mentioning argument details
print(pd.Series(my_data, labels)) # now what this does is, it maps the elements from Labels as the indexing label for
                                  # objects in 'my_data', as long as we know the order of the lables is correct

print(pd.Series(my_data[1], labels))  # this assigns the element 2 in position/index 1 to all the labels


#series also works with dictionaries, it automatically takes keys as index and set values as the corresponding datapoint
# it can contain any type of python data type
print(pd.Series(dicta))

#working with series with another series,
ser1 = pd.Series([1,2,3,4],['a','b','c','d'])
ser2 = pd.Series([1,2,4,5],['a','b','c','f'])

print(ser1+ser2) # pandas try to retain as much data as possible by converting the int to float
print(ser1['a'])