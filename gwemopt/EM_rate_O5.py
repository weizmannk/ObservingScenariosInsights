import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the output directory for plots
outdir = "../output/Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Specify file path for the O5 data
data_path = os.path.join("../data/gwem_detection", "gwem_O5_BNS_detection.dat")

# Read and process the O5 data
df = pd.read_csv(data_path, delimiter="\t")
df["ULTRASAT_detected"] = df["ULTRASAT_detection"] == 1
df["ZTF_detected"] = df["ZTF_detection"] == 1
df["LSST_detected"] = df["LSST_detection"] == 1

print(len(df["LSST_detected"]))

# Predicted gravitational wave detection rates for the O5 observational run
gw_rates = {"lower": 80, "mid": 180, "upper": 400}

# Calculate EM fractions and rates for O5
total_injections = len(df)
em_rates = {}
for telescope in ["ULTRASAT", "ZTF", "LSST"]:
    detected = df[f"{telescope}_detected"].sum()
    fraction = detected / total_injections
    em_rates[telescope] = {
        "EM_rate_mid": fraction * gw_rates["mid"],
        "EM_rate_error_lower": fraction * (gw_rates["mid"] - gw_rates["lower"]),
        "EM_rate_error_upper": fraction * (gw_rates["upper"] - gw_rates["mid"]),
    }

# Plot the EM rates for O5 with error bars
fig, ax = plt.subplots(figsize=(4, 6))
telescopes = ["ULTRASAT", "ZTF", "LSST"]
colors = ["blue", "green", "red"]
bar_width = 0.4

x_positions = np.arange(len(telescopes))

for t_idx, telescope in enumerate(telescopes):
    rate = em_rates[telescope]["EM_rate_mid"]
    error = [
        [em_rates[telescope]["EM_rate_error_lower"]],
        [em_rates[telescope]["EM_rate_error_upper"]],
    ]
    ax.bar(
        x_positions[t_idx],
        rate,
        yerr=error,
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
        "O5",
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
        color="navy",
    )

    ax.set_xticks(np.arange(len(telescopes)))
    ax.set_xticklabels(telescopes, rotation=45)
    ax.set_ylabel("Annual EM Detection Rate (events yr$^{-1}$)")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(outdir, "EM_detection_rate_O5.pdf"))
