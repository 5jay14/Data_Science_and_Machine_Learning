# creating numpy arrays
import numpy as np

# 1d or vector array
my_list = [0, 1, 2, 3, 4]
my_list = np.array(my_list)
print(my_list)

# 2d or Matrices array
my_mat = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
my_mat = np.array(my_mat)
print(my_mat)

# range =  can pass 3 arguments, 1st is implicit, 2nd is explicit and 3rd is a step which means it shows
# what type output we need, odd or even numbers
r = np.arange(10)  # this generates number from 0 to 10
r = np.arange(10, 15)  # this generates numbers from 10 to 14
r = np.arange(0, 10, 3)  # this outputs the gap by 3

print(r)

#   reshaping array from and to 1-2d
r.reshape(2,2)  # converted single array into a matrix

twoDArray = np.arange(100,200).reshape(10,10)
print(twoDArray)

# to figure out the shape of the array. This Function does not use parenthesis
print('Shape', r.shape)

# we can generate the arrays with inbuilt functions
zeroes = np.zeros(2)  # 1d
zeroes = np.zeros((2, 3))  # 2d
print('zeroes',zeroes)

ones = np.ones(5)  # 1d
ones = np.ones((5, 5))  # 2d
print(ones)

# linspace = it is similar to range but the 3rd gives evenly spaced points
ls = np.linspace(0, 100, 10)  # it is a 1 dimension array
print(ls,'ls')

# identity matrices= it is a 2d square matrix. meaning the number of rows = number of columns
# it only takes single digit as the argument as the resultant matrix should be a square/box
im = np.eye(4)  # diagonals of 1 and others are 0's
print(im)

# Useful functions
np.random.rand(5)
np.random.rand(5, 5)
aa = np.random.randint(0, 100, )  # generates a random number between 0 to 100
bb = np.random.randint(0, 100, 5)  # generates 5 random number between 0 to 100

bb.max()  # gives the highest number
bb.min()  # lowest number
bb.argmax()  # gives the index of the highest number
bb.argmin()  # gives the index of the highest number

#Data type
print(bb.dtype)