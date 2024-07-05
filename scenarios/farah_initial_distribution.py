import os
from tqdm.auto import tqdm

from astropy.table import join, Table, vstack
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
from scipy.stats import gaussian_kde

import pandas as pd
import numpy as np

from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
import matplotlib

plt.style.use("seaborn-v0_8-paper")

data_dir = "./farah.h5"

outdir = "./"


Farah = Table.read(f"{data_dir}")  # [:1000]
Farah.sort("mass1")

params = ["mass"]  # , 'spin']

ns_max_mass = 3.0

plt.clf()
# Figure Plot
fig, axs = plt.subplots(nrows=1, ncols=1)

for param in params:
    if param == "mass":
        mass1 = np.log10(Farah["mass1"])  # Farah['mass1']
        mass2 = np.log10(Farah["mass2"])  # Farah['mass2']
        xy = np.vstack([mass1, mass2])

        z = gaussian_kde(xy)(xy)
        index = z.argsort()
        mass1, mass2, z = mass1[index], mass2[index], z[index]

        mass_density_Farah = axs.scatter(mass1, mass2, c=z, s=5)

        axs.set_xlabel(
            r"Primary mass, $\log_{10}(m_1)$ ($M_\odot$)",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )
        axs.set_ylabel(
            r"Secondary mass, $ \log_{10}m_2$ ($M_\odot$)",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )
        axs.text(
            0.05,
            0.95,
            "PDB/GWTC-3",
            transform=axs.transAxes,
            color="navy",
            va="top",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )

        # axs.fill_between([1, 100], [1, 100], [1, 100], color = 'white', linewidth=0, alpha=0.75, zorder=1.5)
        # axs.plot([1, 100], [1,  100], '--k')

    else:
        spin1z = Farah["spin1z"]
        spin2z = Farah["spin2z"]
        xy = np.vstack([spin1z, spin2z])

        z = gaussian_kde(xy)(xy)
        index = z.argsort()
        spin1z, spin2z, z = spin1z[index], spin2z[index], z[index]

        spin_density_Farah = axs[1].scatter(spin1z, spin2z, c=z, s=5)

        axs[1].set_xlabel(
            r"$\mathrm{spin}_1$", fontname="Times New Roman", size=16, fontweight="bold"
        )
        axs[1].set_ylabel(
            r"$\mathrm{spin}_2$", fontname="Times New Roman", size=16, fontweight="bold"
        )
        axs[1].text(
            0.05,
            0.95,
            "PDB/GWTC-3",
            transform=axs[1].transAxes,
            color="navy",
            va="top",
            fontname="Times New Roman",
            size=16,
            fontweight="bold",
        )

# axs.set_xlim(1, 100)
# axs.set_ylim(1, 100)
cbar1 = fig.colorbar(mass_density_Farah, ax=axs)
# cbar1 = fig.colorbar(spin_density_Farah, ax=axs[1])


fig.text(
    0.5,
    0.03,
    "PDB/GWTC-3 distribustion",
    ha="center",
    va="center",
    fontname="Times New Roman",
    fontsize=20,
    color="navy",
)


plt.gcf().set_size_inches(12, 12)
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
# fig.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(f"{outdir}/Farah_gaussian_kde_distribution_of_masses.pdf")
plt.savefig(f"{outdir}/Farah_gaussian_kde_distribution_of_masses.png")
plt.close()


# print the number of each population in Farah distribution
print(
    f"number of BNS: {len(Farah[(Farah['mass1']<ns_max_mass)])}, number of NSBH:"
    f"{len(Farah[(Farah['mass1']>=ns_max_mass) & (Farah['mass2']<ns_max_mass)])}, number of BBH: "
    f"{len(Farah[(Farah['mass2']>=ns_max_mass)])}"
)


ns_max_mass = 3.0

# print the number of each population in Farah distribution
print(
    f"number of BNS: {len(cols[(cols['mass1']<ns_max_mass)])}, number of NSBH:"
    f"{len(cols[(cols['mass1']>=ns_max_mass) & (cols['mass2']<ns_max_mass)])}, number of BBH: "
    f"{len(cols[(cols['mass2']>=ns_max_mass)])}"
)
