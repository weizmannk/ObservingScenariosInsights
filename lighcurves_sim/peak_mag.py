import os
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Set matplotlib to 'agg' backend
import matplotlib as mpl

mpl.use("agg")

# Path setup
path_dir = os.path.join(
    os.path.dirname(os.path.realpath("__file__")), "lightcurve_data", "output_lc"
)
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Configuration for runs, distributions, telescopes, and populations
ztf_filters = ["g", "r", "i"]
rubin_filters = ["g", "r", "i", "u", "z", "y"]
run_names = ["O4", "O5"]
distribution = ["Farah"]
telescopes = ["ZTF", "Rubin"]
pops = ["BNS"]

# Color mappings for ZTF and Rubin observations
band_colors_ztf = {"ztfg": "darkgreen", "ztfr": "darkred", "ztfi": "orange"}
band_colors_rubin = {
    "ps1__g": "darkgreen",
    "ps1__r": "darkred",
    "ps1__i": "orange",
    "sdssu": "blue",
    "ps1__y": "grey",
    "ps1__z": "darkviolet",
}

# Progress bar to track processing
with tqdm(
    total=len(distribution) * len(telescopes) * len(run_names) * len(pops)
) as progress:
    for dist in distribution:
        for pop in pops:
            # Clear the figure at the start of each distribution processing
            plt.clf()
            fig, sax = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

            for run_name in run_names:
                for telescope in telescopes:
                    path = Path(
                        f"{path_dir}/{dist}_lc_{telescope}_{run_name}_{pop}.csv"
                    )
                    lc = pd.read_csv(path)
                    lc = lc[
                        lc["mag_unc"] != 99.0
                    ]  # Filter out bad magnitude uncertainties

                    # Replace the old fiters names
                    filter_mapping = {
                        "ZTF": {"g": "ztfg", "r": "ztfr", "i": "ztfi"},
                        "Rubin": {
                            "u": "sdssu",
                            "g": "ps1__g",
                            "r": "ps1__r",
                            "i": "ps1__i",
                            "z": "ps1__z",
                            "y": "ps1__y",
                        },
                    }
                    lc["filter"] = lc["filter"].map(filter_mapping[telescope])

                    # Determine subplot indices and titles
                    ax_index = (
                        (0, 0 if telescope == "ZTF" else 1)
                        if run_name == "O4"
                        else (1, 0 if telescope == "ZTF" else 1)
                    )

                    if run_name == "O4" and telescope == "ZTF":
                        title = "ZTF O4b"
                    elif run_name == "O4" and telescope == "Rubin":
                        title = "LSST O4b"
                    elif run_name == "O5" and telescope == "ZTF":
                        title = "ZTF O5"
                    else:
                        title = "LSST O5"

                    color_map = (
                        band_colors_ztf if telescope == "ZTF" else band_colors_rubin
                    )

                    # Each band
                    for name, group in (
                        lc.groupby(["sim", "filter"])
                        .apply(lambda x: x["mag"].min())
                        .groupby(level=1)
                    ):
                        hist, bins = np.histogram(group, bins=np.arange(16, 26, 0.5))
                        sax[ax_index].step(
                            bins,
                            np.pad(hist, (1, 0)) / hist.max(),
                            alpha=0.7,
                            color=color_map[name],
                            lw=1,
                            label=name,
                        )

                    ## All bands
                    hist, bins = np.histogram(
                        lc.groupby(["sim"]).apply(lambda x: x["mag"].min()),
                        bins=np.arange(16, 26, 0.5),
                    )
                    sax[ax_index].step(
                        bins,
                        np.pad(hist, (1, 0)) / hist.max(),
                        color="k",
                        lw=1.5,
                        label="all bands",
                    )

                    if telescope == "ZTF":
                        sax[ax_index].text(
                            0.05,
                            0.9,
                            title,
                            transform=sax[ax_index].transAxes,
                            va="top",
                            color="navy",
                            fontname="Times New Roman",
                            size=11,
                            fontweight="bold",
                        )
                    else:
                        sax[ax_index].text(
                            0.35,
                            0.9,
                            title,
                            transform=sax[ax_index].transAxes,
                            va="top",
                            color="navy",
                            fontname="Times New Roman",
                            size=11,
                            fontweight="bold",
                        )

                    sax[ax_index].legend()
                    sax[ax_index].set_xlabel("peak mag")
                    sax[ax_index].get_yaxis().set_visible(False)

                    progress.update()

            # Adjust the layout and save the figure

            plt.gcf().set_size_inches(9, 6)
            plt.subplots_adjust(
                left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
            )
            fig.tight_layout()
            plt.savefig(os.path.join(outdir, f"{dist}_magnitude_{pop}.png"), dpi=300)
            plt.close("all")
