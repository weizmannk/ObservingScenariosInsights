# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script visualizes the annual EM detection rate for LSST, ZTF, and ULTRASAT
                  across different observational runs. It calculates the rates based on
                  the data analysis of gravitational waves and annotates the plots with
                  the median detection rates and error bars.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os

# Define the output directory for plots
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Specify file paths for the data
path = "../data/gwem_detection"
data_paths = {
    "O4a": os.path.join(path, "gwem_O4a_BNS_detection.dat"),
    "O4b": os.path.join(path, "gwem_O4_BNS_detection.dat"),
    "O5": os.path.join(path, "gwem_O5_BNS_detection.dat"),
}

# Create a table with predicted gravitational wave detection rates for different observational runs
gw_rates_table = Table(
    [
        {"BNS": "O4a", "lower": 3, "mid": 11, "upper": 28},
        {"BNS": "O4b", "lower": 14, "mid": 36, "upper": 85},
        {"BNS": "O5", "lower": 80, "mid": 180, "upper": 400},
    ]
)

# Convert the table to a pandas DataFrame and set the index
gw_rates_df = gw_rates_table.to_pandas()
gw_rates_df.set_index("BNS", inplace=True)

# Initialize a dictionary to hold data from each observational run
data_dict = {}

# Read and process the data from each file
for key, file_path in data_paths.items():
    df = pd.read_csv(file_path, delimiter="\t")
    df["ULTRASAT_detected"] = df["ULTRASAT_detection"] == 1
    df["ZTF_detected"] = df["ZTF_detection"] == 1
    df["LSST_detected"] = df["LSST_detection"] == 1

    data_dict[key] = df

# Initialize dictionaries to hold EM fractions and rates for each telescope
em_fractions = {"ULTRASAT": {}, "ZTF": {}, "LSST": {}}
em_rates = {"ULTRASAT": {}, "ZTF": {}, "LSST": {}}

# Calculate EM fractions and rates
for run, df in data_dict.items():
    total_injections = len(df)

    # Calculations for each telescope
    for telescope in ["ULTRASAT", "ZTF", "LSST"]:
        detected = df[f"{telescope}_detected"].sum()
        print(run, " : ", telescope, ":", detected)

        fraction = detected / total_injections
        em_rates[telescope][run] = {
            "EM_rate_mid": fraction * gw_rates_df.loc[run, "mid"],
            "EM_rate_error_lower": fraction
            * (gw_rates_df.loc[run, "mid"] - gw_rates_df.loc[run, "lower"]),
            "EM_rate_error_upper": fraction
            * (gw_rates_df.loc[run, "upper"] - gw_rates_df.loc[run, "mid"]),
        }

# Plot the EM rates for each run with error bars
fig, axes = plt.subplots(1, 3, figsize=(10, 6))  # One subplot for each run
telescopes = ["ULTRASAT", "ZTF", "LSST"]
colors = ["blue", "green", "red"]
bar_width = 0.2

# Plotting for each run
for idx, (run, ax) in enumerate(zip(data_paths.keys(), axes)):
    for t_idx, telescope in enumerate(telescopes):

        if telescope == "ULTRASAT" and run in ["O4a", "O4b"]:

            if run == "O4b":
                y = 0.7
            else:
                y = 0.3
            ax.text(
                0.2,
                y,  # Adjust based on the y-axis scale
                "N/A",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                bbox=dict(
                    facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"
                ),
            )
        else:
            rate = em_rates[telescope][run]["EM_rate_mid"]
            error_lower = em_rates[telescope][run]["EM_rate_error_lower"]
            error_upper = em_rates[telescope][run]["EM_rate_error_upper"]
            ax.bar(
                t_idx,
                rate,
                yerr=[[error_lower], [error_upper]],
                color=colors[t_idx],
                width=bar_width,
                label=telescope,
                capsize=5,
                error_kw={"elinewidth": 1, "ecolor": "black"},
            )
            ax.text(
                t_idx - bar_width / 2,
                rate,  # Adjusted for visibility
                f"{int(np.round(rate))}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=12,
            )

    # Adding run names to each subplot
    ax.text(
        0.5,
        0.7,
        run,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
        color="navy",
    )

    ax.set_xticks(np.arange(len(telescopes)))
    ax.set_xticklabels(telescopes, rotation=45)

    if idx == 0:
        ax.set_ylabel("Annual EM Detection Rate (events yr$^{-1}$)")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "EM_detection_rate.pdf"))
