import numpy as np
import matplotlib.pyplot as plt
data = np.load("transitions.npz")
states = data["states"]
names = ["cart_pos", "cart_vel", "pole_angle", "pole_ang_vel"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    ax.hist(states[:, i], bins=50, edgecolor="black", alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
plt.tight_layout()
plt.savefig("state_histograms.png")
plt.show()