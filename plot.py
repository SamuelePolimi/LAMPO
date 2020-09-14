import matplotlib.pyplot as plt
import numpy as np
import time


class SuperReinforcePlot:

    def __init__(self, kl_bound, title=""):
        self._kl = []
        self._e = []
        self._theo_j = []
        self._iterations = []
        self._t = 0
        self._kl_bound = kl_bound
        self._returns = []
        self._agent_returns = []
        self._timing = []

        self.fig, self.axes = plt.subplots(3, 1)
        self.ax_returns = self.axes[0]
        self.ax_theo_ret = self.axes[1]
        self.ax_kl = self.axes[2]
        plt.suptitle(title)
        self._returns_line, = self.ax_returns.plot(self._returns, label="Success")
        self._agent_returns_line, = self.ax_returns.plot(self._agent_returns, label="reward")
        self.ax_returns.legend(loc="best")
        self._theo_ret_line, = self.ax_theo_ret.plot(self._theo_j)
        self.ax_theo_ret.set_title("Estimated return")
        self._kl_line, = self.ax_kl.plot(self._kl, label="kl")
        self._e_line, = self.ax_kl.plot(self._e, label="entropy")
        self.ax_kl.legend(loc="best")
        self._kl_bound_line, = self.ax_kl.plot([0, 1], [self._kl_bound, self._kl_bound])
        self.ax_kl.set_title("KL Divergence")
        self._fr = 1
        self._max_t = 200

    def notify_inner_loop(self, theo_ret, kl, entropy):
        if self._t % self._fr == 0:
            self._kl.append(kl)
            self._e.append(entropy)
            self._theo_j.append(theo_ret)
            self._timing.append(self._t)
        self._t += 1

        if len(self._timing) == self._max_t:
            self._timing = self._timing[::2]
            self._theo_j = self._theo_j[::2]
            self._kl = self._kl[::2]
            self._fr = self._fr * 2

    def notify_outer_loop(self, average_return, agent_return):
        self._returns.append(average_return)
        self._agent_returns.append(agent_return)
        self._iterations.append(self._t)

    def visualize(self, last=False):
        self._returns_line.set_ydata(self._returns)
        self._returns_line.set_xdata(self._iterations)
        self._agent_returns_line.set_ydata(self._agent_returns)
        self._agent_returns_line.set_xdata(self._iterations)
        self.ax_returns.set_xlim(0, self._t)
        self.ax_returns.set_ylim(min(np.min(self._returns), np.min(self._agent_returns)),
                                 max(np.max(self._returns), np.max(self._agent_returns)))
        self._theo_ret_line.set_ydata(self._theo_j)
        self._theo_ret_line.set_xdata(self._timing)
        self.ax_theo_ret.set_xlim(0, self._t)
        self.ax_theo_ret.set_ylim(np.min(self._theo_j),
                                np.max(self._theo_j))

        self._kl_line.set_ydata(self._kl)
        self._kl_line.set_xdata(self._timing)

        self._e_line.set_ydata(self._e)
        self._e_line.set_xdata(self._timing)

        self._kl_bound_line.set_xdata([0, self._t])

        self.ax_kl.set_xlim(0, self._t)
        self.ax_kl.set_ylim(min(self._kl_bound, np.min(np.concatenate([self._kl, self._e]))),
                                 max(self._kl_bound, np.max(np.concatenate([self._kl, self._e]))))

        if last:
            plt.show()
        else:
            plt.draw()
            plt.pause(1e-17)

