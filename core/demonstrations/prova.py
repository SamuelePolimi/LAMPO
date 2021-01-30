import numpy as np

f = np.concatenate([np.load("obstacle.npy"), np.load("reacher2d_obstacle.npy")], axis=0)
np.save("reacher2d_obstacle.npy", f)