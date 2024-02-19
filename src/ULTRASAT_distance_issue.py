import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to create output directory if it doesn't exist
def create_output_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# Function to plot ULTRASAT detection efficiencies for each run
def plot_ultrasat_efficiencies(data, outdir):
    plt.figure(figsize=(10, 8))

    # Define colors for each run
    run_colors = {
        "O4a": "blue",
        "O4b": "green",
        "O5": "red",
    }

    # Loop through each run and plot detection efficiencies
    for run_name, group_data in data.groupby("run"):
        color = run_colors[run_name]

        # Only include data where ULTRASAT_detection is marked (i.e., = 1)
        ultrasat_detected_data = group_data[group_data["ULTRASAT_detection"] == 1]

        plt.scatter(
            np.log10(ultrasat_detected_data["distance"]),
            ultrasat_detected_data["ULTRASAT_efficiency"],
            label=f"{run_name}",
            color=color,
            alpha=0.7,
        )

    # Set labels and legend
    plt.xlabel("Log10(Distance) (Mpc)", fontsize=14)
    plt.ylabel("ULTRASAT Detection Efficiency", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(outdir, "ULTRASAT_Efficiencies_by_Run.pdf"))


# Define the output directory for plots
output_dir = "../output/Plots"
create_output_directory(output_dir)

# Data paths for the O4a, O4b, and O5 observational runs
data_paths = {
    "O4a": "../data/gwem_detection/gwem_O4a_BNS_detection.dat",
    "O4b": "../data/gwem_detection/gwem_O4_BNS_detection.dat",
    "O5": "../data/gwem_detection/gwem_O5_BNS_detection.dat",
}

# Combined DataFrame for all runs
combined_data = pd.DataFrame()

# Read, label, and concatenate data for each run
for run_name, data_path in data_paths.items():
    data = pd.read_csv(data_path, delimiter="\t")
    data["run"] = run_name  # Label the data with the run name
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Ensure ULTRASAT detection and efficiency data is included in the dataset
if (
    "ULTRASAT_detection" not in combined_data.columns
    or "ULTRASAT_efficiency" not in combined_data.columns
):
    raise ValueError("ULTRASAT detection or efficiency data is missing in the dataset.")

# Plot the ULTRASAT detection efficiencies for all runs
plot_ultrasat_efficiencies(combined_data, output_dir)
