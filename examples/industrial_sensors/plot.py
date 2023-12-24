import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def plot(y_fit, lr_y_fit, y_pred, lr_y_pred, title, show_plot=True, save_plot=False):
    real = np.concatenate([y_fit, y_pred], axis=0)
    model = np.concatenate([lr_y_fit, lr_y_pred], axis=0)

    plt.plot(range(len(real)), real, color='blue')
    plt.plot(range(len(model)), model, color='orange')

    plt.axvline(len(lr_y_fit), color='g', linestyle='--')

    plt.title(title)

    if save_plot:
        timestamp = str(datetime.now().time()).replace(':', '.')
        plt.gcf().savefig('plots/' + timestamp + ' ' + title + '.jpg', bbox_inches='tight')

    if show_plot:
        plt.show()

    plt.close()
