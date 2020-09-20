import numpy as np
from core.plot import LampoMonitor
import matplotlib.pyplot as plt

# my_plot = LampoMonitor.load("experiments/experiment1/result_4.npz")
# my_plot.visualize(last=True)

experiment_name = "experiment4"
experiment_path = "experiments/%s/" % experiment_name

successes = []
rewards = []
for id in range(10):
    try:
        res = np.load(experiment_path + "result_%d.npz" % id)
        successes.append(res["returns"])
        rewards.append(res["agent_returns"])
    except:
        pass

plt.plot(np.array(successes).T, color="black", label="success")
plt.plot(np.mean(successes, axis=0), color="blue", label="success")
# plt.plot(np.array(successes).T, colorlabel="reward")
# plt.legend(loc="best")
plt.show()