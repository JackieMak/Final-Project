"""
Plot diagram of Loss, Accuracy and F1 score of the model.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_graph(x_range, train_loss, train_f1, val_loss, val_f1, test_loss, test_f1):
    x = list(range(x_range))
    for i, k in enumerate(x):
        x[i] = k + 1

    y1, y2, y3, y4, y5, y6 = train_loss, train_f1, val_loss, val_f1, test_loss, test_f1

    f, ax = plt.subplots()
    ax.plot(x, y1, label="Training loss")
    ax.plot(x, y3, label="Validation loss")
    ax.plot(x, y5, label="Testing loss")
    ax.set_xlabel("Number of epoch")
    ax.set_title("Loss")
    ax.legend()
    plt.show()

    f, ax = plt.subplots()
    ax.plot(x, y2, label="Training F1")
    ax.plot(x, y4, label="Validation F1")
    ax.plot(x, y6, label="Testing F1")
    ax.set_xlabel("Number of epoch")
    ax.set_title("F1 score")
    ax.legend()
    plt.show()

