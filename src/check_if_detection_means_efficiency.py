# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script examines gravitational wave electromagnetic (EM) detection
                  data to determine the consistency between detection flags and efficiency
                  values. It categorizes detections, filters out inconsistent cases, and
                  visualizes the results, highlighting any inconsistencies between the
                  detection flags and the reported efficiencies for LSST and ZTF detections.
                  The plot is saved as a PDF in the specified output directory.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to create output directory if it doesn't exist
def create_output_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# Function to check and categorize detections
def categorize_detections(row):
    if row["LSST_detection"] == 1 and row["ZTF_detection"] == 1:
        return "Both Detected"
    elif row["LSST_detection"] == 1:
        return "Only LSST Detected"
    elif row["ZTF_detection"] == 1:
        return "Only ZTF Detected"
    else:
        return "None Detected"


# Function to plot detection efficiencies with inconsistency highlights
def plot_efficiency_consistency(data, lsst_inconsistent, ztf_inconsistent, outdir):
    plt.figure(figsize=(8, 6))

    # Plot consistent LSST and ZTF detections
    plt.scatter(
        np.log10(data.loc[data["LSST_detection"] == 1, "distance"]),
        data.loc[data["LSST_detection"] == 1, "LSST_efficiency"],
        label="Consistent LSST Detections",
        color="blue",
        alpha=0.5,
    )

    plt.scatter(
        np.log10(data.loc[data["ZTF_detection"] == 1, "distance"]),
        data.loc[data["ZTF_detection"] == 1, "ZTF_efficiency"],
        label="Consistent ZTF Detections",
        color="green",
        alpha=0.5,
    )

    # Highlight inconsistent LSST and ZTF detections
    plt.scatter(
        np.log10(lsst_inconsistent["distance"]),
        lsst_inconsistent["LSST_efficiency"],
        label="Inconsistent LSST Detections",
        color="red",
        edgecolor="black",
        alpha=1.0,
    )

    plt.scatter(
        np.log10(ztf_inconsistent["distance"]),
        ztf_inconsistent["ZTF_efficiency"],
        label="Inconsistent ZTF Detections",
        color="orange",
        edgecolor="black",
        alpha=1.0,
    )

    # Set labels and title
    plt.xlabel("Log10(Distance) (Mpc)", fontsize=14)
    plt.ylabel("Detection Efficiency", fontsize=14)
    plt.title("Detection Efficiency Consistency Check", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(outdir, "inconsistent_EM_detection.pdf"))


# Define the output directory for plots
output_dir = "../output/Plots"
create_output_directory(output_dir)

# Load and process the data
data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"
data = pd.read_csv(data_path, delimiter="\t")

# Categorize detections
data["detection_category"] = data.apply(categorize_detections, axis=1)

# Filter out inconsistent cases
lsst_inconsistent = data[
    (data["LSST_detection"] == 1) & (data["LSST_efficiency"] <= 0.5)
]
ztf_inconsistent = data[(data["ZTF_detection"] == 1) & (data["ZTF_efficiency"] <= 0.5)]

# Plot the efficiencies with inconsistency highlights
plot_efficiency_consistency(data, lsst_inconsistent, ztf_inconsistent, output_dir)
