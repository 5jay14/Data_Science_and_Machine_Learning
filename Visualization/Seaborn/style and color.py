import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')
sns.set_style('ticks')
# white, ticks, darkgrid, whitegrid

sns.countplot(tips, x='sex')
# ticks = gives border
#sns.despine(left=True, bottom=True) # specify which side the spine/border needs to be removed


plt.show()

