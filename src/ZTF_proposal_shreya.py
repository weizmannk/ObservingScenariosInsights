# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : December 2023
@description    : Analyzes Compact Binary Coalescence (CBC) events' detection potential
                  and sky localization using ZTF for BNS, NSBH, and BBH mergers
                  in different observing scenarios (O4, O5).

---------------------------------------------------------------------------------------------------
"""

import numpy as np
from pathlib import Path

import os
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
from astropy.coordinates import Distance
import astropy.units as u

import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("agg")

matplotlib.rcParams["xtick.labelsize"] = 12.0
matplotlib.rcParams["ytick.labelsize"] = 12.0
matplotlib.rcParams["legend.fontsize"] = 18
matplotlib.rcParams["axes.titlesize"] = 18


def populations_bool(table, pop, ns_max_mass=3.0):

    """Splits Compact Binary Coalescence (CBC) events based on source frame mass.

    :param table: Table containing the CBC data
    :type table: astropy.table.Table

    :param ns_max_mass: Maximum neutron star mass threshold
    :type ns_max_mass: float

    :param pop: Type of population (BNS, NSBH, and BBH.)
    :type pop: str

    :return: Subset of data based on population criteria
    :rtype: Bool
    """
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1

    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1

    if pop == "BNS":
        data = (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    elif pop == "NSBH":
        data = (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    else:
        data = (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)

    return data


datapath = "../data/runs"

outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)


run_names = ["O4", "O5"]

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3.0

# read in the files
# allsky_BNS = pd.read_csv(datapath+'BNS_O4_allsky.dat',skiprows=1,delimiter=' ')
# allsky_NSBH = pd.read_csv(datapath+'NSBH_O4_allsky.dat',skiprows=1,delimiter=' ')


# How far can ZTF detect a KN assuming GW170817-like luminosity?
Mabs = -16
mlim = 22  # assuming clear weather conditions and 300s exposures
distmod = mlim - Mabs
d = Distance(distmod=distmod, unit=u.Mpc)

# generate 100,000 realizations of the NS mergers assuming the annual rate from OS 2022
# then determine the distribution of events falling within the distance cut

# this rate from kiendrebeogo et al.2023
Number_BNS = {"O4a": 11, "O4": 36, "O5": 180}
Number_NSBH = {"O4a": 2, "O4": 6, "O5": 31}


# Figure Plot
plt.clf()

rows, cols = 1, 2
gs = gridspec.GridSpec(rows, cols)
sax = []
for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols * r + c]))

for run_name in run_names:

    path = Path(f"{datapath}/{run_name}/farah")

    allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")
    injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")

    BNS = populations_bool(table=injections, pop="BNS", ns_max_mass=ns_max_mass)
    NSBH = populations_bool(table=injections, pop="NSBH", ns_max_mass=ns_max_mass)

    allsky_BNS = allsky[BNS].to_pandas()
    allsky_NSBH = allsky[NSBH].to_pandas()

    # generate 100,000 realizations of the NS mergers assuming the annual rate from OS 2022
    # then determine the distribution of events falling within the distance cut
    # the rate in O4 ( 18months) and O5 (24months)

    N_BNS = Number_BNS[run_name]
    N_NSBH = Number_NSBH[run_name]

    rng = np.random.default_rng(42)
    realizations_BNS = [
        np.sum(rng.choice(allsky_BNS["distmean"].to_numpy(), N_BNS) < d.value)
        for i in range(0, 100000)
    ]
    # realizations_BNS_200 = [np.sum(rng.choice(allsky_BNS['distmean'].to_numpy(),N_BNS)<200) for i in range(0,100000)]
    realizations_NSBH = [
        np.sum(rng.choice(allsky_NSBH["distmean"].to_numpy(), N_NSBH) < d.value)
        for i in range(0, 100000)
    ]

    # now we can calculate the percentiles to get errorbars on number of detected events
    Ndet_lower = np.percentile(realizations_BNS, q=5)
    Ndet_higher = np.percentile(realizations_BNS, q=95)
    Ndet_med = np.percentile(realizations_BNS, q=50)

    print(f"The Run {run_name}")
    print("\n==Median values==")
    print(
        "number of detected BNS mergers: %d^{+%d}_{-%d}"
        % (Ndet_med, Ndet_higher - Ndet_med, Ndet_med - Ndet_lower)
    )

    Ndet_lower = np.percentile(realizations_NSBH, q=5)
    Ndet_higher = np.percentile(realizations_NSBH, q=95)
    Ndet_med = np.percentile(realizations_NSBH, q=50)

    print(
        "number of detected NSBH mergers: %d^{+%d}_{-%d}"
        % (Ndet_med, Ndet_higher - Ndet_med, Ndet_med - Ndet_lower)
    )

    print(" ")
    print("**Mean values**")
    print(
        "number of detected BNS mergers: %d" % int(np.round(np.mean(realizations_BNS)))
    )
    print(
        "number of detected NSBH mergers: %d"
        % int(np.round(np.mean(realizations_NSBH)))
    )
    print(" ")

    bins = np.arange(0, 30, 1)

    if run_name == "O4":
        sax[0].hist(
            realizations_BNS,
            bins=bins,
            label="O4 BNS in 400 Mpc",
            density=True,
            cumulative=True,
            histtype="step",
            linestyle="--",
            color="goldenrod",
            linewidth=4,
        )
        sax[0].hist(
            realizations_NSBH,
            bins=bins,
            label="O4 NSBH in 400 Mpc",
            density=True,
            cumulative=True,
            histtype="step",
            linestyle=":",
            color="teal",
            linewidth=4,
        )

        bbox = dict(
            facecolor="white", alpha=0.8, edgecolor="teal", linestyle=":", linewidth=2.5
        )
        sax[0].text(2.8, 0.4, "NSBH", color="k", fontsize=24, bbox=bbox)
        bbox = dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="goldenrod",
            linestyle="--",
            linewidth=2.5,
        )
        sax[0].text(20, 0.4, "BNS", color="k", fontsize=24, bbox=bbox)
        sax[0].text(
            2.25, 0.3, f"<N>={int(np.round(np.mean(realizations_NSBH)))}", fontsize=24
        )
        sax[0].text(
            18, 0.3, f"<N>={int(np.round(np.mean(realizations_BNS)))}", fontsize=24
        )
        sax[0].text(
            -2,
            1.01,
            r"Run O4",
            color="blue",
            fontname="Times New Roman",
            fontweight="bold",
            fontsize=24,
        )

        sax[0].tick_params(axis="both", labelsize=18, width=2)
        for axis in ["top", "bottom", "left", "right"]:
            sax[0].spines[axis].set_linewidth(2)

    else:
        sax[1].hist(
            realizations_BNS,
            bins=bins,
            label="O5 BNS in 400 Mpc",
            density=True,
            cumulative=True,
            histtype="step",
            linestyle="--",
            color="goldenrod",
            linewidth=4,
        )
        sax[1].hist(
            realizations_NSBH,
            bins=bins,
            label="O5 NSBH in 400 Mpc",
            density=True,
            cumulative=True,
            histtype="step",
            linestyle=":",
            color="teal",
            linewidth=4,
        )

        bbox = dict(
            facecolor="white", alpha=0.8, edgecolor="teal", linestyle=":", linewidth=2
        )
        sax[1].text(2.8, 0.4, "NSBH", color="k", fontsize=24, bbox=bbox)
        bbox = dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="goldenrod",
            linestyle="--",
            linewidth=2,
        )
        sax[1].text(20, 0.4, "BNS", color="k", fontsize=24, bbox=bbox)
        sax[1].text(
            2.25, 0.3, f"<N>={int(np.round(np.mean(realizations_NSBH)))}", fontsize=24
        )
        sax[1].text(
            18, 0.3, f"<N>={int(np.round(np.mean(realizations_BNS)))}", fontsize=24
        )
        sax[1].text(
            -2,
            1.01,
            r"Run O5",
            color="blue",
            fontname="Times New Roman",
            fontweight="bold",
            fontsize=24,
        )

        sax[1].tick_params(axis="both", labelsize=18, width=2)
        for axis in ["top", "bottom", "left", "right"]:
            sax[1].spines[axis].set_linewidth(2)


# sax[0].set_title('Annual LVK-detected NS Mergers in O4 within 400 Mpc')
sax[0].set_ylabel("cumulative probability density", size=27, fontname="Times New Roman")
sax[0].set_xlabel("number of events", size=27, fontname="Times New Roman")

# sax[1].set_title('Annual LVK-detected NS Mergers in O5 within 400 Mpc')
sax[1].set_xlabel("number of events", size=27, fontname="Times New Roman")
sax[0].set_xlim(-4, 25)
sax[1].set_xlim(-4, 28.9)

plt.gcf().set_size_inches(20, 10)
plt.subplots_adjust(right=0.9, wspace=0.4, hspace=0.4)
plt.tight_layout()

plt.savefig(f"{outdir}/ndet_22mag_allsky_BNS_NSBH_os2022.pdf")
plt.close()


# This plot shows the 90% C.R.
# shows the Caltech, public survey, and partnership allocations in probing sky localizations
# this figure was used in the proposal for ZTF O4

# Figure Plot
plt.clf()
# fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, constrained_layout=True)
rows, cols = 1, 2
gs = gridspec.GridSpec(rows, cols)
sax = []

for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols * r + c]))


for run_name in run_names:
    path = Path(f"{datapath}/{run_name}/farah")
    allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")

    if run_name == "O4":
        sax[0].hist(
            allsky["area(90)"],
            bins=np.arange(0, 30000, 100),
            histtype="step",
            density=True,
            cumulative=True,
            label="90% C.R.",
            color="b",
        )

        # three filter strategy (30, 50, 100)
        sax[0].axvline(480, linestyle=":", color="m", label=None)
        sax[0].axvline(800, linestyle="-", color="goldenrod", label=None)
        sax[0].axvline(1600, linestyle="--", color="k", label=None, alpha=0.7)

        sax[0].text(400, 0.12, "3 visits", rotation="vertical", fontsize=24, color="m")
        sax[0].text(500, 0.22, "P", rotation="vertical", fontsize=24, color="m")
        sax[0].text(
            720, 0.18, "3 visits", rotation="vertical", fontsize=24, color="goldenrod"
        )
        sax[0].text(
            820, 0.27, "C+P", rotation="vertical", fontsize=24, color="goldenrod"
        )
        sax[0].text(1510, 0.25, "3 visits", rotation="vertical", fontsize=24, alpha=0.9)
        sax[0].text(1620, 0.25, "C+P+M", rotation="vertical", fontsize=24.0, alpha=0.9)

        # from 30 to 50%
        # plt.arrow(480, 0.05, 510, 0, width=0.01, head_width=0.1, head_length=70, length_includes_head=True, fc='b', ec='b')
        sax[0].arrow(
            480,
            0.09,
            320,
            0,
            width=0.01,
            head_width=0.1,
            head_length=70,
            length_includes_head=True,
            fc="k",
            ec="k",
        )
        # plt.ylim(0, 400)

        sax[0].tick_params(axis="both", labelsize=18, width=2)
        for axis in ["top", "bottom", "left", "right"]:
            sax[0].spines[axis].set_linewidth(2)

    else:
        sax[1].hist(
            allsky["area(90)"],
            bins=np.arange(0, 30000, 100),
            histtype="step",
            density=True,
            cumulative=True,
            label="90% C.R.",
            color="b",
        )

        # three filter strategy (30, 50, 100)
        sax[1].axvline(480, linestyle=":", color="m", label=None)
        sax[1].axvline(800, linestyle="-", color="goldenrod", label=None)
        sax[1].axvline(1600, linestyle="--", color="k", label=None, alpha=0.7)

        # three filter
        sax[1].text(400, 0.15, "3 visits", rotation="vertical", fontsize=22, color="m")
        sax[1].text(500, 0.25, "P", rotation="vertical", fontsize=22, color="m")
        sax[1].text(
            720, 0.18, "3 visits", rotation="vertical", fontsize=22, color="goldenrod"
        )
        sax[1].text(
            820, 0.27, "C+P", rotation="vertical", fontsize=22, color="goldenrod"
        )
        sax[1].text(1510, 0.25, "3 visits", rotation="vertical", fontsize=22, alpha=0.9)
        sax[1].text(1620, 0.25, "C+P+M", rotation="vertical", fontsize=22, alpha=0.9)

        # from 30 to 50%
        # plt.arrow(480, 0.05, 510, 0, width=0.01, head_width=0.1, head_length=70, length_includes_head=True, fc='b', ec='b')
        sax[1].arrow(
            480,
            0.09,
            320,
            0,
            width=0.01,
            head_width=0.1,
            head_length=70,
            length_includes_head=True,
            fc="k",
            ec="k",
        )
        # plt.ylim(0, 400)

        sax[1].tick_params(axis="both", labelsize=18, width=2)
        for axis in ["top", "bottom", "left", "right"]:
            sax[1].spines[axis].set_linewidth(2)


sax[0].set_ylabel(f"fraction of O4 triggers", size=26, fontname="Times New Roman")
sax[0].set_xlabel("sky area (deg)", size=25, fontname="Times New Roman")
sax[0].set_xlim(-1, 2000)

sax[1].set_ylabel(f"fraction of O5 triggers", size=26, fontname="Times New Roman")
sax[1].set_xlabel("sky area (deg)", size=25, fontname="Times New Roman")
sax[1].set_xlim(-1, 2000)

sax[0].legend(loc="upper left")
sax[1].legend(loc="upper left")


plt.gcf().set_size_inches(20, 10)
plt.subplots_adjust(right=0.9, wspace=0.4, hspace=0.4)
plt.tight_layout()

plt.savefig(f"{outdir}/observing_scenarios_areas_trigger_zoom_8h_NSBH_BNS.pdf")
plt.close()
