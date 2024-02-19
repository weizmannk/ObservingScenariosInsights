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
                  whether they were detected by LSST, ZTF, both, or none. The detection
                  efficiencies are then visualized in a scatter plot with respect to the
                  logarithmic distance to the source. The plot is saved as a PDF in the
                  specified output directory.
---------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to categorize detections based on detection flags
def categorize_detections(row):
    if row["LSST_detection"] == 1 and row["ZTF_detection"] == 1:
        return "Both Detected"
    elif row["LSST_detection"] == 1:
        return "Only LSST Detected"
    elif row["ZTF_detection"] == 1:
        return "Only ZTF Detected"
    else:
        return "None Detected"


# Function to plot detection efficiencies comparison
def plot_efficiency_comparison(data, title, output_path):
    # Define colors for each detection category
    color_map = {
        "None Detected": "lightgray",
        "Both Detected": "green",
        "Only LSST Detected": "red",
        "Only ZTF Detected": "blue",
    }

    # Apply log10 to distance
    data["log_distance"] = np.log10(data["distance"])

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot efficiencies based on detection category
    for category, color in color_map.items():
        category_data = data[data["detection_category"] == category]
        if (
            "LSST_efficiency" in category_data.columns
            and category != "Only ZTF Detected"
        ):
            ax.scatter(
                category_data["log_distance"],
                category_data["LSST_efficiency"],
                color=color,
                alpha=0.7,
                edgecolors="black",
                s=50,
                label=category,
            )
        if (
            "ZTF_efficiency" in category_data.columns
            and category != "Only LSST Detected"
        ):
            ax.scatter(
                category_data["log_distance"],
                category_data["ZTF_efficiency"],
                color=color,
                alpha=0.7,
                edgecolors="black",
                s=50,
                label=category,
            )

    # Set the labels and title for the axes
    ax.set_xlabel("Log10(Distance) (Mpc)", fontsize=14)
    ax.set_ylabel("Detection Efficiency", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True)

    # Define the legend and remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper right",
        title="Detection Category",
    )

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
output_file_path = os.path.join(outdir, "EM_efficiencies.pdf")
plot_efficiency_comparison(data, "EM Detection Efficiency Comparison", output_file_path)
