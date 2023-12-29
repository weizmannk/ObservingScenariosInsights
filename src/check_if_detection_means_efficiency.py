import pandas as pd

# Load the data
data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"
data = pd.read_csv(data_path, delimiter="\t")

# Check if the columns exist in the dataframe
required_columns = [
    "LSST_detection",
    "ZTF_detection",
    "LSST_efficiency",
    "ZTF_efficiency",
]
missing_columns = [column for column in required_columns if column not in data.columns]

# If there are no missing columns, proceed to check the relationship between detection flags and efficiencies
if not missing_columns:
    # Check if when detection == 1, the corresponding efficiency > 0.5
    lsst_compatible = data.loc[data["LSST_detection"] == 1, "LSST_efficiency"] > 0.5
    ztf_compatible = data.loc[data["ZTF_detection"] == 1, "ZTF_efficiency"] > 0.5

    # Determine the number of incompatible cases
    lsst_incompatible_cases = (~lsst_compatible).sum()
    ztf_incompatible_cases = (~ztf_compatible).sum()

    compatibility_check = {
        "missing_columns": missing_columns,
        "lsst_incompatible_cases": lsst_incompatible_cases,
        "ztf_incompatible_cases": ztf_incompatible_cases,
    }
else:
    compatibility_check = {
        "missing_columns": missing_columns,
        "error": "One or more required columns are missing from the data.",
    }

compatibility_check


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)


# Reload the data
data_path = "../data/gwem_detection/gwem_O5_BNS_detection.dat"
data = pd.read_csv(data_path, delimiter="\t")

# Define the categorize_detections function
def categorize_detections(row):
    if row["LSST_detection"] == 1 and row["ZTF_detection"] == 1:
        return "Both Detected"
    elif row["LSST_detection"] == 1:
        return "Only LSST Detected"
    elif row["ZTF_detection"] == 1:
        return "Only ZTF Detected"
    else:
        return "None Detected"


# Add a new column to categorize detections based on whether the detection flag is 1
data["detection_category"] = data.apply(categorize_detections, axis=1)

# Filter out the cases where the detection flag is 1 but efficiency is 0.5 or below
lsst_inconsistent = data[
    (data["LSST_detection"] == 1) & (data["LSST_efficiency"] <= 0.5)
]
ztf_inconsistent = data[(data["ZTF_detection"] == 1) & (data["ZTF_efficiency"] <= 0.5)]

# Start plotting
plt.figure(figsize=(8, 6))

# Plot consistent LSST detections
plt.scatter(
    np.log10(data.loc[data["LSST_detection"] == 1, "distance"]),
    data.loc[data["LSST_detection"] == 1, "LSST_efficiency"],
    label="Consistent LSST Detections",
    color="blue",
    alpha=0.5,
)

# Plot consistent ZTF detections
plt.scatter(
    np.log10(data.loc[data["ZTF_detection"] == 1, "distance"]),
    data.loc[data["ZTF_detection"] == 1, "ZTF_efficiency"],
    label="Consistent ZTF Detections",
    color="green",
    alpha=0.5,
)

# Highlight inconsistent LSST detections
plt.scatter(
    np.log10(lsst_inconsistent["distance"]),
    lsst_inconsistent["LSST_efficiency"],
    label="Inconsistent LSST Detections",
    color="red",
    edgecolor="black",
    alpha=1.0,
)

# Highlight inconsistent ZTF detections
plt.scatter(
    np.log10(ztf_inconsistent["distance"]),
    ztf_inconsistent["ZTF_efficiency"],
    label="Inconsistent ZTF Detections",
    color="orange",
    edgecolor="black",
    alpha=1.0,
)

# Set the labels and title
plt.xlabel("Log10(Distance) (Mpc)", fontsize=14)
plt.ylabel("Detection Efficiency", fontsize=14)
plt.title("Detection Efficiency Consistency Check", fontsize=16)
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(f"{outdir}/inconsistent_EM_detection.pdf")
