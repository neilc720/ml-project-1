import numpy as np
import matplotlib.pyplot as plt

def explore_data(x, y, title):
    nb_samples = x.shape[0]
    nb_negative = np.sum(y == -1)
    nb_positive = np.sum(y == 1)
    print("Number of training examples: ", nb_samples)
    print(f"Number of negative labels: {nb_negative} ({100*nb_negative/nb_samples}%)")
    print(f"Number of positive labels: {nb_positive} ({100*nb_positive/nb_samples}%)")
    print("Number of features: ", x.shape[1])
    plt.hist(y)
    plt.xlabel("Class")
    plt.ylabel("Number of examples")
    plt.xticks([-1, 1])
    plt.title(f"Distribution of labels in the {title}")
    plt.show()