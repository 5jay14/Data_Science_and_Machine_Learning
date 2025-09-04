import numpy as np

# Create an array of 10 zeros
ten0s = np.zeros(10)
print(ten0s)

# Create an array of 10 ones
ten1s = np.ones(10)
print(ten1s)

# Create an array of 10 fives
ten5s = np.linspace(5, 5, 10)
print(ten5s)
# other ways

ten5s = np.zeros(10) + 5
print(ten5s)

# Create an array of the integers from 10 to 50
array10to50 = np.arange(10, 51)
print(array10to50)

# Create an array of all the even integers from 10 to 50
array10to50 = np.arange(10, 51, 2)
print(array10to50)

# Create a 3x3 matrix with values ranging from 0 to 8
threeByThree = np.arange(9).reshape(3, 3)
print(threeByThree)

# Create a 3x3 identity matrix
threeByThree = np.eye(3)
print(threeByThree)

# Use NumPy to generate a random number between 0 and 1
rand0 = np.random.randint(0, 1)
print(rand0)

#   Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution
rand25 = np.random.randn(25)
print(rand25)

# Create the following matrix:
a = np.arange(0.01, 1.01, 0.01).reshape(10, 10)
a1 = np.arange(1, 101).reshape(10, 10) / 100  # same result
a2 = np.linspace(0.01, 1, 100).reshape(10, 10)  # same result
print(a)

# Create an array of 20 linearly spaced points between 0 and 1:
b = np.linspace(0, 1, 20)
print(b)

# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:
mat = np.arange(1, 26).reshape(5, 5)

# [
# [ 1,  2,  3,  4,  5],
# [ 6,  7,  8,  9, 10],
# [11, 12, 13, 14, 15],
# [16, 17, 18, 19, 20],
# [21, 22, 23, 24, 25]
#                    ]

# result should be
#     1.    [[12, 13, 14, 15],
#           [17, 18, 19, 20],
#           [22, 23, 24, 25]]
#     2.    20
#     3.    [[2],
#            [7],
#            [12]
#      4.   [21, 22, 23, 24, 25]
#      5.   [[16, 17, 18, 19, 20],
#           [21, 22, 23, 24, 25]]
#      6.   Get the sum of all the values in mat. 325
#      7.   Get the standard deviation of the values in mat.  7.2111025509279782
#      8.   Get the sum of all the columns in mat. [55, 60, 65, 70, 75]
#

print(mat[2:, 1:5])
print(mat[3][4])
print(mat[:3, 1:2])
print(mat[4:])
print(mat[3:])
print(mat.sum())
print(mat.std())
mat1 = mat.transpose()
print(mat.sum(axis=1))