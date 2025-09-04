import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 100)
y = x * 2
z = x ** 2

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.set_xlabel('X')
axes.set_ylabel('Y')
axes.set_title('Title')
axes.plot(x, y)

fig1 = plt.figure()
ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
ax1.set_title('Outer')
ax1.plot(x, y)
ax2 = fig1.add_axes([0.2, 0.5, 0.2, 0.2])
ax2.set_title("Inner")
ax2.set_xlim(20,22)
ax2.set_ylim(30,50)

ax2.plot(x, y)

fig2 = plt.figure()
ax3 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax3.set_title('Larger plot')
ax3.plot(x, z)
ax4 = fig2.add_axes([0.2, 0.5, 0.3, 0.3])
ax4.set_title('Smaller plot')
ax4.plot(x, y / 100)

fig3, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
axes1[0].plot(x, y, linestyle='--')
axes1[1].plot(x, z, linewidth=2, linestyle='-.', color='red')

plt.show()
plt.tight_layout()