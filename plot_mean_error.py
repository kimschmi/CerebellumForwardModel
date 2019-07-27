import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit()

number_training_examples = 5000
number_epochs = 2000
recording_interval = 2

data = np.load(sys.argv[1])
training_errors = data['training_errors']
training_errors = np.reshape(training_errors, (int(number_epochs / recording_interval), number_training_examples))
mean_training_errors = np.mean(training_errors[:, :], axis=1)

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
plt.plot(np.arange(len(mean_training_errors))*recording_interval, mean_training_errors, lw=1.)
plt.xlabel(r'Epoch')
plt.ylabel(r'Mean Error')
plt.show()
