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
                  detection flags and the reported efficiencies for LSST, ZTF, and ULTRASAT detections.
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
    lsst_detected = row["LSST_detection"] == 1
    ztf_detected = row["ZTF_detection"] == 1
    ultrasat_detected = row["ULTRASAT_detection"] == 1

    if lsst_detected and ztf_detected and ultrasat_detected:
        return "All Detected"
    elif lsst_detected and ztf_detected:
        return "LSST and ZTF Detected"
    elif lsst_detected and ultrasat_detected:
        return "LSST and ULTRASAT Detected"
    elif ztf_detected and ultrasat_detected:
        return "ZTF and ULTRASAT Detected"
    elif lsst_detected:
        return "Only LSST Detected"
    elif ztf_detected:
        return "Only ZTF Detected"
    elif ultrasat_detected:
        return "Only ULTRASAT Detected"
    else:
        return "None Detected"


# Function to plot detection efficiencies with inconsistency highlights
def plot_efficiency_consistency(data, outdir):
    plt.figure(figsize=(10, 8))

    # Plot consistent LSST, ZTF, and ULTRASAT detections
    for telescope, color in zip(
        ["LSST", "ZTF", "ULTRASAT"], ["blue", "green", "purple"]
    ):
        consistent_data = data[
            (data[f"{telescope}_detection"] == 1)
            & (data[f"{telescope}_efficiency"] > 0.5)
        ]
        plt.scatter(
            np.log10(consistent_data["distance"]),
            consistent_data[f"{telescope}_efficiency"],
            label=f"Consistent {telescope} Detections",
            color=color,
            alpha=0.5,
        )

    # Highlight inconsistent LSST, ZTF, and ULTRASAT detections
    for telescope, color in zip(
        ["LSST", "ZTF", "ULTRASAT"], ["red", "orange", "yellow"]
    ):
        inconsistent_data = data[
            (data[f"{telescope}_detection"] == 1)
            & (data[f"{telescope}_efficiency"] <= 0.5)
        ]
        plt.scatter(
            np.log10(inconsistent_data["distance"]),
            inconsistent_data[f"{telescope}_efficiency"],
            label=f"Inconsistent {telescope} Detections",
            color=color,
            edgecolor="black",
            alpha=1.0,
        )

    # Set labels and title
    plt.xlabel("Log10(Distance) (Mpc)", fontsize=14)
    plt.ylabel("Detection Efficiency", fontsize=14)
    # plt.title("Detection Efficiency Consistency Check", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(outdir, "Telescope_inconsistent_EM_detection.pdf"))


# Define the output directory for plots
output_dir = "../output/Plots"
create_output_directory(output_dir)

# Load and process the data
data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"
data = pd.read_csv(data_path, delimiter="\t")

# Ensure ULTRASAT data is included in your dataset
if (
    "ULTRASAT_detection" not in data.columns
    or "ULTRASAT_efficiency" not in data.columns
):
    raise ValueError("ULTRASAT data is missing in the dataset.")

# Categorize detections
data["detection_category"] = data.apply(categorize_detections, axis=1)

# Plot the efficiencies with inconsistency highlights
plot_efficiency_consistency(data, output_dir)
