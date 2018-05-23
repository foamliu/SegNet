import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from config import colors, classes

if __name__ == '__main__':
    handles = []
    for i in range(len(classes)):
        color = tuple(np.array(colors[i])[::-1] / 255.)
        label = classes[i]
        handles.append(mpatches.Patch(color=color, label=label))

    plt.legend(handles=handles)

    plt.show()
