import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 11, 10)
y = x ** 2

fig = plt.figure(figsize=(8, 3), dpi=100)
# figure = measurement in inches, 1st is horizontal width, 2nd is vertical height
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y)
plt.show()


# Applying the figsize using subplots()
fig1, axes1 = plt.subplots(nrows=2,ncols=1,figsize=(8, 3))
axes1[0].plot(x,y)
axes1[1].plot(y,x)
plt.show()
plt.tight_layout()