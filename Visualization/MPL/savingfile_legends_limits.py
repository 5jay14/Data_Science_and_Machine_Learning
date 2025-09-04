import matplotlib.pyplot as plt
import numpy as np

# MPL can save figures in multiple extensions

x = np.linspace(0, 11, 10)
y = x ** 2
#fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3), dpi=200)
#axes[0].plot(x, y)
#axes[1].plot(y, x)
#fig.savefig("my_fig.png")  # other formats .jpg,


#Legends
#fig1 = plt.figure()
#ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
#ax.plot(x, x ** 2, label ='X Square')
#ax.plot(x, x ** 3, label ='X Cube')
#ax.legend(loc =0) # optional, can specify where we want the legend to be
# best =0, upper right 1, and so on till 10



fig2 = plt.figure()

ax1= fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.set_title('limits')
ax2.plot(x,y)
ax2.set_xlim()
ax2.set_ylim()

plt.show()
