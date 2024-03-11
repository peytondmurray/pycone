import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

alpha = 1
wtau = np.linspace(0, np.pi * 10, 10000)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

betas = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
colors = plt.cm.inferno_r(
    0.25 + 0.75 * (betas - min(betas)) / (max(betas) - min(betas))
)

for beta, color in zip(betas, colors, strict=True):
    h = alpha / np.sqrt(1 + beta**2 - 2 * beta * np.cos(wtau))
    ax.plot(wtau / np.pi, h, "-", color=color, label=rf"$\beta = {beta:3.2}$")
ax.xaxis.set_major_formatter(FormatStrFormatter(r"%g$\pi$"))
ax.xaxis.set_major_locator(MultipleLocator(base=1.0))
ax.set_xlabel(r"$\omega \tau$ [rad/s]", fontsize="xx-large")
ax.set_ylabel(r"$|H(i\omega)|$", fontsize="xx-large")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.tick_params(axis="both", which="major", labelsize=18, width=2, length=10)
ax.tick_params(axis="both", which="minor", labelsize=14, width=2, length=10)
for spine in ax.spines.values():
    spine.set_linewidth(2)
fig.legend(loc="upper right")
fig.tight_layout()
fig.savefig("comb_filter.svg")
