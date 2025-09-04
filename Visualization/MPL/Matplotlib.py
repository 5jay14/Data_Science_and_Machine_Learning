import matplotlib.pyplot as plt
import numpy as np

# There are two ways creating matplot libplot
# 1 Functional method
# 2.Object-oriented method

# Functional
x = np.linspace(0, 11, 10)
y = x ** 2

plt.plot(x, y)

# Adding Label
plt.xlabel('X label')
plt.ylabel('Y Label')
plt.title('Title')
plt.show()  # same as printing, need to keep typing everytime

# Multi/sub plotting on the same canvas

plt.subplot(1, 2, 1)
plt.plot(x, y, 'r')
plt.subplot(1, 2, 2)
plt.plot(y, x, 'b')

# Using object-oriented method

# step 1 - Create a figure with a list
fig = plt.figure()

# Step 2 - add axes to figure
# axes is list of mpl axes objects which can be iterated and indexed
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # 1st = left, 2nd = bottom, 3rd = width size of canvas, 4th = height of
# the canvas
# all these values must be within 1 if its a decimal or 10 if the values are integers, as this is in percentage
# these values are relational to blank canvas

# step 3 - Plotting
axes.plot(x, y)

plt.show()

# Sub/Multi-plotting in the same canvas using object-oriented method
fig1 = plt.figure()
axes1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
axes1.set_title('Larger plot')
axes2 = fig1.add_axes([0.2, 0.5, 0.5, 0.3])
axes2.set_title('Smaller plot')
axes1.plot(x, y, 'r')
axes2.plot(y, x, 'b')
plt.show()

# Subplotting on the objects using subplots()
# subplots - we can specify number of rows and columns, it is an axes manager

fig, axes = plt.subplots(nrows=1, ncols=2)
# difference between plt.figure and plt.subplots. Subplots is automatically adding axes unlinke fig() function
# see now there are 3 rows with two columns
# Since axes it a list, it can be indexed
axes[0].plot(x, y)  # i can plot on the first figure
axes[0].set_title('First axes')

axes[1].plot(y, x)  # i can plot only on the second figure
axes[1].set_title('2nd axes')
plt.show()

# Dealing with overlapping plots, recommended to use at the end of the plot statements
plt.tight_layout()
