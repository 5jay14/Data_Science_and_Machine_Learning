import numpy as np
# Array with array operations

arr1 = np.arange(11)
arr2 = arr1.copy()

print(arr1 + arr2)
print(arr1 * arr2)
print(arr1 - arr2)
print(arr1 / arr2)  # Np array does not throw an error but it shows a warning

# array with scalars operations
# scalar is single number
print(arr1 + 100)  # this will add 100 to each element
print(100 - arr1)
print(arr1 / 100)
print(arr1 * 100)

# Universal array functions
print(np.sqrt(arr1))
np.max(arr1)
np.min(arr1)
print(np.greater_equal(arr1,arr2))