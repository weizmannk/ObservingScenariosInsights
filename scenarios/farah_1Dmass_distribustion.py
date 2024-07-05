import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("agg")

fig_width_pt = 500.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 1.2 * fig_width * golden_mean  # height in inches
fig_size = [fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
matplotlib.rcParams.update(params)


outdir = "./paper_plots"

if not os.path.isdir(outdir):
    os.makedirs(outdir)


ALPHA_1 = -2.16
ALPHA_2 = -1.46
A = 0.97
M_GAP_LO = 2.72
M_GAP_HI = 6.13
ETA_GAP_LO = 50
ETA_GAP_HI = 50
ETA_MIN = 50
ETA_MAX = 4.91
BETA = 1.89
M_MIN = 1.16
M_MAX = 54.38


def lopass(m, m_0, eta):
    return 1 / (1 + (m / m_0) ** eta)


def hipass(m, m_0, eta):
    return 1 - lopass(m, m_0, eta)


def bandpass(m, m_lo, m_hi, eta_lo, eta_hi, A):
    return 1 - A * hipass(m, m_lo, eta_lo) * lopass(m, m_hi, eta_hi)


def pairing_function(m1, m2):
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    return np.where((m1 <= 60) | (m2 >= 2.5), (m2 / m1) ** BETA, 0)


def mass_distribution_1d(m):
    return (
        bandpass(m, M_GAP_LO, M_GAP_HI, ETA_GAP_LO, ETA_GAP_HI, A)
        * hipass(m, M_MIN, ETA_MIN)
        * lopass(m, M_MAX, ETA_MAX)
        * (m / M_GAP_HI) ** np.where(m < M_GAP_HI, ALPHA_1, ALPHA_2)
    )


def mass_distribution_2d(m1, m2):
    return (
        mass_distribution_1d(m1) * mass_distribution_1d(m2) * pairing_function(m1, m2)
    )


# Plot 1D distribution of component mass.

m = np.geomspace(1, 100, 1000000)

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")

ax.plot(m, m * mass_distribution_1d(m), "g")
ax.set_xlim(1, 100)
ax.set_ylim(0, 99.99)
ax.set_xlabel(r"mass, $m$ [$M_\odot$]")
ax.set_ylabel(r"$m \,  p(m|\lambda)$")
# ax.set_yticks([])

# Fill between x=1.9 and x=2.9 for the entire y-range dynamically determined
y_min, y_max = ax.get_ylim()
ax.fill_between([1.9, 2.9], y_min, y_max, color="blue", alpha=0.1, zorder=1.5)

# the x-values 1.9 and 2.9
ax.axvline(x=1.9, color="blue", linestyle="--", alpha=0.7)
ax.axvline(x=2.9, color="blue", linestyle="--", alpha=0.7)
ax.text(
    2.4,
    y_min - 0.01,
    r"$2.4^{+0.5}_{-0.5}$",
    horizontalalignment="center",
    fontweight="bold",
    fontsize=11,
)
# ax.text(2.9, y_min-0.01, '2.9', horizontalalignment='right', color='red')


ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xscale(ax.get_xscale())
ax2.set_xticks([M_MIN, M_GAP_LO, M_GAP_HI, M_MAX])
ax2.set_xticklabels(
    [
        r"$M_\mathrm{min}$",
        r"$M^\mathrm{gap}_\mathrm{low}$",
        r"$M^\mathrm{gap}_\mathrm{high}$",
        r"$M_\mathrm{max}$",
    ]
)
ax2.set_xticks([], minor=True)
ax2.grid(axis="x")


fig.tight_layout()
plt.savefig(f"{outdir}/supress_mass_gap.pdf", bbox_inches="tight")
plt.close()

# Plot joint 2D distribution of m1, m2.

# m1, m2 = np.meshgrid(m, m)
# fig, ax = plt.subplots(subplot_kw=dict(aspect=1))
# ax.set_xlim(1, 100)
# ax.set_ylim(1, 100)
# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel(r'Primary mass, $m_1$ ($M_\odot$)')
# ax.set_ylabel(r'Secondary mass, $m_2$ ($M_\odot$)')
# img = ax.pcolormesh(m, m, m1 * m2 * mass_distribution_2d(m1, m2),
#                     vmin=0, vmax=25, shading='gouraud', rasterized=True)
# cbar = plt.colorbar(img, ax=ax)
# cbar.set_label(r'$m_1 \, m_2 \, p(m_1, m_2 | \Lambda)$')
# cbar.set_ticks([])

# ax.fill_between([1, 100],
#                 [1, 100],
#                 [100, 100],
#                 color='white', linewidth=0, alpha=0.75, zorder=1.5)
# ax.plot([1, 100], [1, 100], '--k')

# ax.annotate('',
#             xy=(0.975, 1.025), xycoords='axes fraction',
#             xytext=(1.025, 0.975), textcoords='axes fraction',
#             ha='center', va='center',
#             arrowprops=dict(
#                 arrowstyle='->', shrinkA=0, shrinkB=0,
#                 connectionstyle='angle,angleA=90,angleB=180,rad=7'))
# ax.text(0.975, 1.025, '$m_1 \geq m_2$ by definition  ',
#         ha='right', va='center', transform=ax.transAxes, fontsize='small')

# fig.savefig('2D_distribution .png')  # Save the figure to a file

plt.close()

# Now safely switch the backend


fig.show()
