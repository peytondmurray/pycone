import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

beta = -0.9
wtau = np.linspace(0, np.pi * 10, 10000)
h = 1 / np.sqrt((1 + beta**2) - 2 * beta * np.cos(wtau))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(wtau / np.pi, h, "-k", label=r"$\beta = -0.9$")
ax.xaxis.set_major_formatter(FormatStrFormatter(r"%g$\pi$"))
ax.xaxis.set_major_locator(MultipleLocator(base=1.0))
ax.set_xlabel(r"$\omega \tau$ [rad/s]", fontsize="xx-large")
ax.set_ylabel(r"$|H(i\omega)|$", fontsize="xx-large")
ax.set_xlim(0, 10)
ax.tick_params(axis="both", which="major", labelsize=18, width=2, length=10)
ax.tick_params(axis="both", which="minor", labelsize=14, width=2, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(2)
fig.legend(loc="upper right")
fig.tight_layout()
fig.savefig("comb_filter.svg")
