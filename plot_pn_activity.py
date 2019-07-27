import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Please provide an input file.")
    exit()

data = np.load(sys.argv[1])
pn_activity = data['pn_activity']
targets = data['test_targets']
d_delay = 50
d_stim = 20
d_response = 5
arm_length = [0.5, 0.5]

pn_baselines_approx = np.max(pn_activity[0, 25:, :], axis=0)
targets = (targets + np.sum(arm_length)) / (2 * np.sum(arm_length))
targets = targets * (pn_baselines_approx / 2) + (pn_baselines_approx / 4)

nice_fonts = {
            "text.usetex": True,
            "font.family": "serif",
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.fontsize": 6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
}
matplotlib.rcParams.update(nice_fonts)
pattern = random.randint(0, np.size(pn_activity, axis=0))
plt.plot(pn_activity[pattern, :, 0], linewidth=1., label='PN 0')
plt.plot(pn_activity[pattern, :, 1], linewidth=1., label='PN 1')
plt.axhline(y=targets[pattern, 0], linewidth=1., label='target 0', color='tab:blue', ls='dashed')
plt.axhline(y=targets[pattern, 1], linewidth=1., label='target 1', color='tab:orange', ls='dashed')
plt.axvline(x=d_stim, c='k', ls='dashed', lw=1.)
plt.axvline(x=d_stim+d_delay, c='k', ls='dotted', lw=1.)
plt.xlabel('Time (ms)')
plt.ylabel('PN firing rate')
plt.ylim(0.0, 2.2)
plt.legend()

plt.show()