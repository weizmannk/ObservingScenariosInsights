import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to categorize detections based on ULTRASAT detection flag
def categorize_detections(row):
    ultrasat_detected = row["ULTRASAT_detection"] == 1
    return "ULTRASAT Detected" if ultrasat_detected else "ULTRASAT Not Detected"


# Function to plot ULTRASAT detection efficiencies for multiple runs on the same graph
def plot_combined_ultrasat_efficiency(data, output_path):
    # Apply log10 to distance
    data["log_distance"] = np.log10(data["distance"])

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors and markers for each run
    run_styles = {
        "O4a": {"color": "blue", "marker": "o"},
        "O4b": {"color": "green", "marker": "s"},
        "O5": {"color": "red", "marker": "^"},
    }

    # Plot data for each run
    for run_name, group_data in data.groupby("run"):
        style = run_styles[run_name]
        detected_data = group_data[
            group_data["detection_category"] == "ULTRASAT Detected"
        ]
        non_detected_data = group_data[
            group_data["detection_category"] == "ULTRASAT Not Detected"
        ]

        ax.scatter(
            detected_data["log_distance"],
            np.ones(len(detected_data)),
            color=style["color"],
            marker=style["marker"],
            label=f"{run_name} Detected",
            alpha=0.7,
            edgecolors="black",
        )
        ax.scatter(
            non_detected_data["log_distance"],
            np.zeros(len(non_detected_data)),
            color=style["color"],
            marker=style["marker"],
            label=f"{run_name} Not Detected",
            alpha=0.5,
            edgecolors="black",
        )

    # Set the labels and title for the axes
    ax.set_xlabel("Log10(Distance) (Mpc)", fontsize=14)
    ax.set_ylabel("ULTRASAT Detection Status", fontsize=14)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not Detected", "Detected"])
    ax.grid(True, which="both", axis="x", linestyle="--", linewidth=0.5)

    # Define the legend
    ax.legend(loc="best")

    # Adjust the layout and save the plot
    plt.tight_layout()
    plt.savefig(output_path)


# Output directory for plots
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Combined data frame for all runs
combined_data = pd.DataFrame()

# Data paths for the O4a, O4b, and O5 observational runs
data_paths = {
    "O4a": "../data/gwem_detection/gwem_O4a_BNS_detection.dat",
    "O4b": "../data/gwem_detection/gwem_O4_BNS_detection.dat",
    "O5": "../data/gwem_detection/gwem_O5_BNS_detection.dat",
}

# Read, label, and concatenate data for each run
for run_name, data_path in data_paths.items():
    data = pd.read_csv(data_path, delimiter="\t")
    data["run"] = run_name  # Label the data with the run name
    data["detection_category"] = data.apply(categorize_detections, axis=1)
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Plot and save the combined ULTRASAT detection efficiencies graph
output_file_path = os.path.join(outdir, "Combined_ULTRASAT_Detection_Efficiency.pdf")
plot_combined_ultrasat_efficiency(combined_data, output_file_path)
