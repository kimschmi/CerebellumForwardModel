import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from arm_helper import compute_end_effector_positions

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit()

arm_length = [0.5, 0.5]
number_training_examples = 5000
number_training_trajectories = 500

data = np.load(sys.argv[1])
training_angles = data['training_angles']
responses = data['responses']

x = np.zeros((np.size(training_angles, 0), 3))
y = np.zeros((np.size(training_angles, 0), 3))
x[:, 1] = np.cos(training_angles[:, 0]) * arm_length[0]
y[:, 1] = np.sin(training_angles[:, 0]) * arm_length[0]
x[:, 2], y[:, 2] = compute_end_effector_positions(arm_length, training_angles[:, 0], training_angles[:, 1])
new_angle = training_angles[:, :2] + training_angles[:, 2:]
x_next = np.zeros((np.size(new_angle, 0), 3))
y_next = np.zeros((np.size(new_angle, 0), 3))
x_next[:, 1] = np.cos(new_angle[:, 0]) * arm_length[0]
y_next[:, 1] = np.sin(new_angle[:, 0]) * arm_length[0]
x_next[:, 2], y_next[:, 2] = compute_end_effector_positions(arm_length, new_angle[:, 0], new_angle[:, 1])

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
sample = random.randint(0, number_training_trajectories)
plt.plot(x[sample, :], y[sample, :], 'ko-', markersize=3)
plt.plot(x_next[sample, :], y_next[sample, :], 'ro-', markersize=3)
t = np.arange(np.size(responses, 1))
plt.scatter(responses[sample, :, 0], responses[sample, :, 1], c=t, s=2)
plt.colorbar(label='Epoch')
plt.xlim(-(np.sum(arm_length) + 0.2), np.sum(arm_length) + 0.2)
plt.ylim(-(np.sum(arm_length) + 0.2), np.sum(arm_length) + 0.2)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

