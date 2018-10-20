import numpy as np
import math
import csv
import matplotlib.pyplot as plt

labels = {'HEALTHY': 1, 'MEDICATION': 2, 'SURGERY': 3}
lines = csv.reader(open("./Medical_data.csv"))
dataset = list(lines)
del dataset[0]
trainData = [dataset[index] for index in range(1, int(len(dataset)))]
x_train = [None] * len(dataset)
y_train = [None] * len(dataset)
for index in range(len(dataset)):
    x_train[index] = [
        1.0,
        float(dataset[index][1]),
        float(dataset[index][2]),
        float(dataset[index][3])
    ]
    y_train[index] = labels[dataset[index][0]]

lines = csv.reader(open("./test_medical.csv"))
dataset = list(lines)
del dataset[0]
testData = [dataset[index] for index in range(1, int(len(dataset)))]
x_test = [None] * len(dataset)
y_test = [None] * len(dataset)
for index in range(len(dataset)):
    x_test[index] = [
        1.0,
        float(dataset[index][1]),
        float(dataset[index][2]),
        float(dataset[index][3])
    ]
    y_test[index] = labels[dataset[index][0]]

X = np.array(x_train)
y = np.array(y_train)
label_dict = {1: 'HEALTHY', 2: 'MEDICATION', 3: 'SURGERY'}

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))

for ax, cnt in zip(axes.ravel(), range(4)):

    min_b = math.floor(np.min(X[:, cnt]))
    max_b = math.ceil(np.max(X[:, cnt]))
    bins = np.linspace(min_b, max_b, 100)

    for lab, col in zip(range(1, 4), ('blue', 'red', 'green')):
        X_temp = []
        for i in range(len(X)):
            if ((y[i] - lab) == 0):
                X_temp.append(X[i, cnt])
        ax.hist(
            X_temp,
            color=col,
            label='class %s' % label_dict[lab],
            bins=bins,
            alpha=0.5,
        )
    ylims = ax.get_ylim()

    leg = ax.legend(loc='upper right', fancybox=True, fontsize=8)
    leg.get_frame().set_alpha(0.5)
    ax.set_ylim([0, max(ylims) + 2])
    ax.set_xlabel('feature #%s' % str(cnt))

    ax.tick_params(
        axis="both",
        which="both",
        bottom="off",
        top="off",
        labelbottom="on",
        left="off",
        right="off",
        labelleft="on")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

axes[0][0].set_ylabel('count')
axes[1][0].set_ylabel('count')

fig.tight_layout()

plt.show()
