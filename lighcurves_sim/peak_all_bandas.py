import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from os.path import exists

# Set up the output directory for the plots
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Define parameters for plotting
run_names = ["O4", "O5"]
telescopes = ["ZTF", "Rubin", "ULTRASAT"]
pops = ["BNS"]
path_dir = f"{os.path.dirname(os.path.realpath('__file__'))}/lightcurve_data/output_lc"

# Progress bar to monitor the plotting process
with tqdm(total=len(run_names) * len(telescopes) * len(pops)) as progress:
    for pop in pops:
        plt.clf()
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4.5))

        for i, run_name in enumerate(run_names):
            for telescope in telescopes:
                path = Path(f"{path_dir}/Farah_lc_{telescope}_{run_name}_{pop}.csv")
                if exists(f"{path}"):
                    lc = pd.read_csv(f"{path}")
                    lc = lc[lc["mag_unc"] != 99.0]

                else:
                    continue
                # Calculate the histogram for the peak magnitudes
                hist, bins = np.histogram(
                    lc.groupby(["sim"]).apply(lambda x: x["mag"].min()),
                    bins=np.arange(16, 26, 0.5),
                )

                # Normalize the histogram
                hist_normalized = np.pad(hist, (1, 0)) / hist.max()

                # Plotting the histogram
                if telescope == "ZTF":
                    color = "green"
                elif telescope == "Rubin":
                    color = "red"
                else:
                    color = "blue"

                if telescope == "ZTF":
                    label = "ZTF"
                elif telescope == "Rubin":
                    label = "LSST"
                else:
                    label = "ULTRSAT"

                axs[i].step(bins, hist_normalized, color=color, lw=1, label=label)

                title = "O4b" if run_name == "O4" else run_name

                # Customize the subplots
                axs[i].text(
                    0.85,
                    0.95,
                    title,
                    transform=axs[i].transAxes,
                    va="top",
                    color="navy",
                    fontname="Times New Roman",
                    size=13,
                    fontweight="bold",
                )
                axs[i].legend(loc="upper left")
                axs[i].set_xlabel("peak mag")
                axs[i].get_yaxis().set_visible(
                    False
                )  # This will hide the y-axis values

                progress.update()

    # Adjust layout and save the figure
    plt.gcf().set_size_inches(9, 6)
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    fig.tight_layout()
    plt.savefig(f"{outdir}/peak_magnitude_all_bands.png", dpi=300)
    plt.close()
