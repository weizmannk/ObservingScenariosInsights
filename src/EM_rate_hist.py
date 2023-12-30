# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script visualizes the annual EM detection rate for LSST and ZTF
                  across different observational runs. It calculates the rates based on
                  the data analysis of gravitational waves and annotates the plots with
                  the median detection rates.
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
    "O4b": os.path.join(path, "gwem_O4_BNS_detection.dat"),  # Update if necessary
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
    df["LSST_detected"] = df["LSST_detection"] == 1
    df["ZTF_detected"] = df["ZTF_detection"] == 1
    data_dict[key] = df

# Initialize dictionaries to hold EM fractions and rates for each telescope
em_fractions = {"LSST": {}, "ZTF": {}}
em_rates = {"LSST": {}, "ZTF": {}}

# Calculate EM fractions and rates
for run, df in data_dict.items():
    total_injections = len(df)
    lsst_detected = df["LSST_detected"].sum()
    ztf_detected = df["ZTF_detected"].sum()

    # LSST calculations
    em_fractions["LSST"][run] = lsst_detected / total_injections
    em_rates["LSST"][run] = {
        "EM_rate_lower": em_fractions["LSST"][run] * gw_rates_df.loc[run, "lower"],
        "EM_rate_mid": em_fractions["LSST"][run] * gw_rates_df.loc[run, "mid"],
        "EM_rate_upper": em_fractions["LSST"][run] * gw_rates_df.loc[run, "upper"],
    }

    # ZTF calculations
    em_fractions["ZTF"][run] = ztf_detected / total_injections
    em_rates["ZTF"][run] = {
        "EM_rate_lower": em_fractions["ZTF"][run] * gw_rates_df.loc[run, "lower"],
        "EM_rate_mid": em_fractions["ZTF"][run] * gw_rates_df.loc[run, "mid"],
        "EM_rate_upper": em_fractions["ZTF"][run] * gw_rates_df.loc[run, "upper"],
    }

# Convert the EM rates to a pandas DataFrame for easier visualization and saving
em_rates_lsst_df = pd.DataFrame.from_dict(em_rates["LSST"], orient="index")
em_rates_ztf_df = pd.DataFrame.from_dict(em_rates["ZTF"], orient="index")

# Plot the EM rates with adjusted bar width and annotated values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))  # Set figure size

# Define bar width and colors
bar_width = 0.3  # Thinner bars
lsst_color = "red"
ztf_color = "green"

# Plot LSST EM Rates
lsst_bars = ax1.bar(
    em_rates_lsst_df.index,
    em_rates_lsst_df["EM_rate_mid"],
    yerr=[
        em_rates_lsst_df["EM_rate_mid"] - em_rates_lsst_df["EM_rate_lower"],
        em_rates_lsst_df["EM_rate_upper"] - em_rates_lsst_df["EM_rate_mid"],
    ],
    capsize=5,
    color=lsst_color,
    width=bar_width,
)
ax1.set_title("LSST EM Rates")
ax1.set_ylabel("Annual EM Detection Rate (events yr$^{-1}$)")
ax1.set_xlabel("Observational Run")
ax1.set_xticks(range(len(em_rates_lsst_df.index)))
ax1.set_xticklabels(em_rates_lsst_df.index, rotation=45)

# Plot ZTF EM Rates
ztf_bars = ax2.bar(
    em_rates_ztf_df.index,
    em_rates_ztf_df["EM_rate_mid"],
    yerr=[
        em_rates_ztf_df["EM_rate_mid"] - em_rates_ztf_df["EM_rate_lower"],
        em_rates_ztf_df["EM_rate_upper"] - em_rates_ztf_df["EM_rate_mid"],
    ],
    capsize=5,
    color=ztf_color,
    width=bar_width,
)
ax2.set_title("ZTF EM Rates")
ax2.set_xlabel("Observational Run")
ax2.set_xticks(range(len(em_rates_ztf_df.index)))
ax2.set_xticklabels(em_rates_ztf_df.index, rotation=45)

# Annotate bars with the 'mid' values
for bar, value in zip(lsst_bars, em_rates_lsst_df["EM_rate_mid"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2 - 0.09,
        bar.get_height(),
        f"{int(np.round(value))}",
        ha="center",
        va="bottom",
        color="navy",
        fontsize=12,
    )

for bar, value in zip(ztf_bars, em_rates_ztf_df["EM_rate_mid"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2 - 0.09,
        bar.get_height(),
        f"{int(np.round(value))}",
        ha="center",
        va="bottom",
        color="navy",
        fontsize=12,
    )

plt.tight_layout()
plt.savefig(os.path.join(outdir, "EM_number_hist.pdf"))

# Output
(em_rates_lsst_df, em_rates_ztf_df)
