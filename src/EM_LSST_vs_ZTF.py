import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)


data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"

# Define the categorize_detections function based on your detection flag columns
def categorize_detections(row):
    if row["LSST_detection"] == 1 and row["ZTF_detection"] == 1:
        return "Both Detected"
    elif row["LSST_detection"] == 1:
        return "Only LSST Detected"
    elif row["ZTF_detection"] == 1:
        return "Only ZTF Detected"
    else:
        return "None Detected"


# Read the data
data = pd.read_table(data_path)

# Apply the categorize_detections function to categorize each detection
data["detection_category"] = data.apply(categorize_detections, axis=1)

# Define the plotting function
def plot_efficiency_comparison(data, title):
    # Create a figure and a set of subplots
    _, ax = plt.subplots(figsize=(8, 6))

    # Define colors for each detection category
    color_map = {
        "None Detected": "lightgray",
        "Both Detected": "green",
        "Only LSST Detected": "red",
        "Only ZTF Detected": "blue",
    }

    # Apply log10 to distance
    # data['log_distance'] = np.log10(data['distance'])
    data["log_distance"] = np.log10(data["distance"])

    # Plotting for all data with colors based on detection category
    for category, color in color_map.items():
        # Filter the data for the current category
        category_data = data[data["detection_category"] == category]

        # Plot LSST efficiency against log10(distance) if the category is not 'Only ZTF Detected'
        if "LSST_efficiency" in data.columns and category != "Only ZTF Detected":
            ax.scatter(
                category_data["log_distance"],
                category_data["LSST_efficiency"],
                color=color,
                alpha=0.7,
                edgecolor="black",
                s=50,
                label=category,
            )

        # Plot ZTF efficiency against log10(distance) if the category is not 'Only LSST Detected'
        if "ZTF_efficiency" in data.columns and category != "Only LSST Detected":
            ax.scatter(
                category_data["log_distance"],
                category_data["ZTF_efficiency"],
                color=color,
                alpha=0.7,
                edgecolor="black",
                s=50,
                label=category,
            )

    # Set the labels for the axes
    ax.set_xlabel("Log10(Distance) (Mpc)", fontsize=14)
    ax.set_ylabel("Detection Efficiency", fontsize=14)
    ax.grid(True)

    # Define the legend and remove duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(
        unique_labels.values(),
        unique_labels.keys(),
        loc="upper right",
        title="Detection Category",
    )

    # Set the title and adjust the layout
    plt.title(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{outdir}/EM_efficiencies.pdf")


# Call the function to plot the graph
plot_efficiency_comparison(data, "EM Detection Efficiency Comparison")
