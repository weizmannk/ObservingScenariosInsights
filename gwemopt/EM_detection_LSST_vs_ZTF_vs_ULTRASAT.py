# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : This script analyzes gravitational wave electromagnetic (EM) detection
                  data for the O5 observational run. It categorizes detections based on
                  whether they were detected by LSST, ZTF, ULTRASAT, combinations of these, or none.
                  The detection efficiencies are then visualized in a scatter plot with
                  respect to the logarithmic distance to the source. The plot is saved as
                  a PDF in the specified output directory.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to categorize detections based on detection flags
def categorize_detections(row):
    lsst_detected = row["LSST_detection"] == 1
    ztf_detected = row["ZTF_detection"] == 1
    ultrasat_detected = row["ULTRASAT_detection"] == 1

    if lsst_detected and ztf_detected and ultrasat_detected:
        return "Detected by All"
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


# Function to plot detection efficiencies comparison
def plot_efficiency_comparison(data, output_path):
    # Define colors for each detection category
    color_map = {
        "None Detected": "lightgray",
        "Detected by All": "olive",
        "LSST and ZTF Detected": "purple",
        "LSST and ULTRASAT Detected": "cyan",
        "ZTF and ULTRASAT Detected": "darkgoldenrod",
        "Only LSST Detected": "red",
        "Only ZTF Detected": "blue",
        "Only ULTRASAT Detected": "darkgreen",
    }

    # Apply log10 to distance
    data["log_distance"] = np.log10(data["distance"])

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(8, 6))

    # Track labels to avoid duplicates in the legend
    added_labels = set()

    # Plot efficiencies based on detection category
    for category, color in color_map.items():
        category_data = data[data["detection_category"] == category]

        # Add scatter plots for each telescope efficiency
        for telescope in ["LSST", "ZTF", "ULTRASAT"]:
            efficiency_column = f"{telescope}_efficiency"
            if efficiency_column in category_data.columns:
                label = f"{category}" if category not in added_labels else None
                ax.scatter(
                    category_data["log_distance"],
                    category_data[efficiency_column],
                    color=color,
                    alpha=0.7,
                    edgecolors="black",
                    s=50,
                    label=label,
                )
                added_labels.add(category)

    # Set the labels and title for the axes
    ax.set_xlabel("Log10(Distance) (Mpc)", fontsize=14)
    ax.set_ylabel("Detection Efficiency", fontsize=14)
    # ax.set_title(title, fontsize=16)
    ax.grid(True)

    # Define the legend
    ax.legend(loc="upper right", title="Detection Category")

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)


# Output directory for plots
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Data path for the O5 observational run
data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"

# Read the data and process
data = pd.read_csv(data_path, delimiter="\t")
data["detection_category"] = data.apply(categorize_detections, axis=1)

# Plot and save the detection efficiencies comparison graph
output_file_path = os.path.join(outdir, "Telescopes_EM_efficiencies.pdf")
plot_efficiency_comparison(data, output_file_path)
