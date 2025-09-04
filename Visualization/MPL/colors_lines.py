import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 11)
y = x ** 2

fig = plt.figure()
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y) # color='green', linewidth=2, alpha=1, linestyle=':', marker='o',
                # markerfacecolor='yellow', markeredgewidth=2, markeredgecolor='black')

# Can also pass RGB hex codes starting with
# Alpha = transparency
# lw is shortcode for linewidth
# ls or linestyle = '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# marker =+,o,*,1
# markersize = 1 to
plt.show()
