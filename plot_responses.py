import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit()

arm_length = [0.5, 0.5]

data = np.load(sys.argv[1])
test_errors = data['test_errors']
test_responses = data['test_responses']

sort_indices = np.argsort(test_errors)
test_errors = np.sort(test_errors)
test_responses = test_responses[sort_indices, :]

nice_fonts = {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
}
matplotlib.rcParams.update(nice_fonts)
plt.scatter(test_responses[:, 0], test_responses[:, 1], c=test_errors, cmap='inferno', s=0.5)
plt.colorbar(label='Euclidean distance')
plt.xlim(-np.sum(arm_length) - 0.2, np.sum(arm_length) + 0.2)
plt.ylim(-np.sum(arm_length) - 0.2, np.sum(arm_length) + 0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
