# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO from Leo Psinger
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script generates and visualizes HEALPix tessellations for given
                  fields of view (FoV) on separate subplots within the same figure. It's
                  specifically tailored for the ULTRASAT, LSST, and ZTF missions with
                  FoVs of 204, 9.6, and 47 square degrees respectively. Each tessellation is
                  visualized using an astro globe projection to represent the celestial sphere.
                  Dependencies: astropy, matplotlib, ligo.skymap, dorado-scheduling
                  Install with: pip install astropy matplotlib ligo.skymap dorado-scheduling
---------------------------------------------------------------------------------------------------
"""

import os
from astropy import units as u
from dorado.scheduling import skygrid
from matplotlib import pyplot as plt

import ligo.skymap.plot  # we need ligo.skymap for "astro globe" projection

# Define the output directory for plots and ensure it exists
outdir = "../output/Plots"
os.makedirs(outdir, exist_ok=True)

# Define the areas (FOV) and method to use for tessellation
areas = {"LSST": 9.6 * u.deg**2, "ZTF": 47 * u.deg**2, "ULTRASAT": 204 * u.deg**2}
method = skygrid.healpix

# Create a figure for the subplots
fig, axes = plt.subplots(
    1,
    3,
    figsize=(12, 5.5),
    subplot_kw={"projection": "astro globe", "center": "0d 25d"},
)

# Loop through each mission and create its subplot
for ax, (mission, area) in zip(axes, areas.items()):
    ax.coords["ra"].set_ticklabel_visible(False)
    ax.coords["dec"].set_ticks_visible(False)
    # Generate and plot the tessellation
    ax.plot_coord(method(area), "o", markersize=1.5, color="navy")
    ax.grid()
    # Set the title with mission name and area size
    ax.set_title(
        f"{mission} - FOV: {area.to_string(format='latex')}",
        pad=20,
        color="navy",
        fontsize=14,
    )

fig.text(
    0.5,
    0.05,
    "HEALPix sky grid",
    ha="center",
    va="center",
    fontname="Times New Roman",
    fontsize=20,
)  # Adjust 'Times New Roman' as needed for your system

# Adjust layout to prevent overlap and accommodate the suptitle
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(outdir, "sky_grid.pdf"))
