import sys
import os
import time
import numpy as np

from arm_helper import compute_end_effector_positions
from network import number_in
from network import number_pn
from network import create_network
from network import train
from network import test


if len(sys.argv) < 3:
    print("Please indicate an input file and a directory to which results should be saved.")
    exit()
if os.path.isfile(sys.argv[1]):
    print("The input will be read from " + sys.argv[1])
else:
    print("The input file does not exist.")
    exit()
if os.path.isdir(sys.argv[2]):
    print("Results will be saved to " + sys.argv[2])
else:
    print("Indicated directory doesn't exist.")
    exit()

arm_length = [0.5, 0.5]

# Create training and test set
print("Read input file")
data = np.load(sys.argv[1])
training_set = data['training_set']    # theta_0, theta_1, delta theta_0, delta theta_1
test_set = data['test_set']    # theta_0, theta_1, delta theta_0, delta theta_1
max_delta = data['max_delta']    # upper bound of delta theta

number_training_examples = np.size(training_set, axis=0)
number_test_examples = np.size(test_set, axis=0)
training_input = np.zeros((number_training_examples, number_in))    # input: x_prev, y_prev, delta theta_0, delta theta_1
training_targets = np.zeros((number_training_examples, number_pn))    # target:  x_next, y_next
test_input = np.zeros((number_test_examples, number_in))
test_targets = np.zeros((number_test_examples, number_pn))

# Compute previous end effector positions: x_prev, y_prev
training_input[:, 0], training_input[:, 1] = compute_end_effector_positions(arm_length, theta0=training_set[:, 0],
                                                                            theta1=training_set[:, 1])
test_input[:, 0], test_input[:, 1] = compute_end_effector_positions(arm_length, theta0=test_set[:, 0],
                                                                    theta1=test_set[:, 1])
training_input[:, 2:] = training_set[:, 2:]
test_input[:, 2:] = test_set[:, 2:]
# Compute next end effector positions: x_next, y_next
training_targets[:, 0], training_targets[:, 1] = compute_end_effector_positions(arm_length,
    theta0=training_set[:, 0] + training_set[:, 2],
    theta1=training_set[:, 1] + training_set[:, 3])
test_targets[:, 0], test_targets[:, 1] = compute_end_effector_positions(arm_length,
    theta0=test_set[:, 0] + test_set[:, 2],
    theta1=test_set[:, 1] + test_set[:, 3])

# Scale inputs to range [-1. 1.]
training_input[:, :2] /= np.sum(arm_length)
test_input[:, :2] /= np.sum(arm_length)
training_input[:, 2:] /= max_delta
test_input[:, 2:] /= max_delta

# Baseline MF input to PN: required to map network response to target range
pn_baselines = np.random.normal(1.2, 0.1, number_pn)
net = create_network(pn_baselines)

# Train with algorithm by Bouvier et al. 2018
print("Train network...")
training_errors, error_estimates, training_responses = train(net, training_input, training_targets, pn_baselines, arm_length)

# Test network performance
print("Test network performance...")
test_errors, test_responses, pn_activity = test(net, test_input, test_targets, pn_baselines, arm_length)

filename = time.strftime("%Y%m%d%H%M")
filename = filename + "_results.npz"
filename = os.path.join(sys.argv[2], filename)

np.savez_compressed(filename, training_errors=training_errors, error_estimates=error_estimates,
                    training_angles=training_set, training_targets=training_targets, responses=training_responses,
                    pn_activity=pn_activity, test_angles=test_set, test_targets=test_targets,
                    test_responses=test_responses, test_errors=test_errors)

print("Results have been saved to " + filename)
