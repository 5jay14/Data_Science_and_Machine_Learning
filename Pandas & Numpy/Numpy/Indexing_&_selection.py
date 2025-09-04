import numpy as np

arr = np.arange(0, 25)
print(arr[5])
print(arr[0:5])
print(arr[:8])
print(arr[0:])
print(arr[:])

# Broad CASTING = This makes the changes to original array. Data is not just copied, it is a view
# This is done to avoid memory issue with big arrays
arr[0:5] = 100
print(arr)

# copying the array, this creates the copy of the array as it was during copying
arr_copy = arr.copy()
############################
# 2d Array

arr_2d = np.array([[0, 1, 2], [4, 5, 6], [7, 8, 9]])
print(arr_2d.shape)
# There are two ways of grabbing the element

print(arr_2d[2][1])
print(arr_2d[1, 0])


# we can use slicing
# 1st is row and 2nd is columns
print('Here=====>', (arr_2d[:2, 1:])) #grab from first to 2nd row, grab from 1st column to all

# another example
Rand_array = (np.arange(0, 10))

print(Rand_array)

#always have same rando
Rand_array = np.random.seed(101)

###############################################

# We can have array of booleans and have conditional selections

intt = np.arange(15)
bool_array = intt > 5
print(bool_array)

# or
true_indices = intt[intt > 5]
print(true_indices)

# Transpose. Convert rows into columns
abc = np.arange(100).reshape(10, 10)
print(abc.transpose())

# when we want to calculate the sum of each array (row), you can use the sum function along the appropriate axis
# In a 2D array (like a matrix), you have two axes:
# Axis 0: Corresponds to rows.
# Axis 1: Corresponds to columns.
print(abc.sum(axis=0))
print(abc.sum(axis=1))
