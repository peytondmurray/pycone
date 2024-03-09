import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

beta = -0.9
wtau = np.linspace(0, np.pi * 10, 1000)
h = 1 / np.sqrt((1 + beta**2) - 2 * beta * np.cos(wtau))

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(wtau / np.pi, h, "-k")
ax.xaxis.set_major_formatter(FormatStrFormatter(r"%g$\pi$"))
ax.xaxis.set_major_locator(MultipleLocator(base=1.0))
ax.set_xlabel(r"$\omega \tau$ [rad/s]")
ax.set_ylabel(r"$|H(e^{i\omega})|$")
plt.show()
