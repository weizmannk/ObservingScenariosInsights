import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy.table import Table
from pathlib import Path
from astropy.cosmology import Planck15 as cosmo


# Set the style for the plots
sns.set_style("whitegrid")


"""


# Set up figure size and plot parameters
fig_width_pt = 700.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = 2*0.9 * fig_width * golden_mean  # height in inches
fig_size = [11.5, 10.5] #[fig_width, fig_height]
params = {
    "backend": "pdf",
    "axes.labelsize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "text.usetex": True,
    "font.family": "Times New Roman",
    "figure.figsize": fig_size,
}
mpl.rcParams.update(params)

"""

# Define the runs data
run_names = ["O4a", "O4b", "O5"]
pops = ["BNS", "NSBH", "BBH"]

# Initial population numbers for Farah's sample
initial = {
    "O4a": {"BNS": 892762, "NSBH": 35962, "BBH": 71276},
    "O4b": {"BNS": 892762, "NSBH": 35962, "BBH": 71276},
    "O5": {"BNS": 892762, "NSBH": 35962, "BBH": 71276},
}

# Detected numbers for each run
detection = {
    "O4a": {"BNS": 398, "NSBH": 89, "BBH": 4230},
    "O4b": {"BNS": 1004, "NSBH": 184, "BBH": 7070},
    "O5": {"BNS": 2003, "NSBH": 356, "BBH": 9809},
}

# Resampling numbers for each run
resampling = {
    "O4a": {"BNS": 247450, "NSBH": 170732, "BBH": 581818},
    "O4b": {"BNS": 587016, "NSBH": 60357, "BBH": 352627},
    "O5": {"BNS": 768739, "NSBH": 54642, "BBH": 176619},
}

# Initialize lists to store detection numbers and resampling numbers
BNS_detect, NSBH_detect, BBH_detect = [], [], []
BNS_resamp, NSBH_resamp, BBH_resamp = [], [], []

# Fill in the lists with the corresponding numbers
for run_name in run_names:
    BNS_detect.append(detection[run_name]["BNS"])
    NSBH_detect.append(detection[run_name]["NSBH"])
    BBH_detect.append(detection[run_name]["BBH"])

    BNS_resamp.append(resampling[run_name]["BNS"])
    NSBH_resamp.append(resampling[run_name]["NSBH"])
    BBH_resamp.append(resampling[run_name]["BBH"])

# Convert lists to numpy arrays
BNS_detect = np.array(BNS_detect)
NSBH_detect = np.array(NSBH_detect)
BBH_detect = np.array(BBH_detect)
BNS_resamp = np.array(BNS_resamp)
NSBH_resamp = np.array(NSBH_resamp)
BBH_resamp = np.array(BBH_resamp)

# Calculate the percentage of detections
gwtc_3_event_nsbh = (NSBH_detect / NSBH_resamp) * 100
gwtc_3_event_bns = (BNS_detect / BNS_resamp) * 100
gwtc_3_event_bbh = (BBH_detect / BBH_resamp) * 100

# Calculate the total GWTC-3 detections
gwtc_3_tot = (BNS_detect + NSBH_detect + BBH_detect) / 1e6 * 100

# Define colors for the plot
bns_color, nsbh_color, bbh_color = sns.color_palette("rocket", 3)

# Create the plot
fig, ax1 = plt.subplots(figsize=(17.25, 10.5))
ax1.set_yscale("log")
x = np.arange(len(run_names))  # the label locations
width = 0.3  # the width of the bars


# Plot the bars for detection numbers with black edge color
rects1 = ax1.bar(
    x - width,
    BNS_detect,
    width,
    label="BNS",
    color=bns_color,
    edgecolor=["black"] * len(BNS_detect),
    linewidth=1.5,
)
rects2 = ax1.bar(
    x,
    NSBH_detect,
    width,
    label="NSBH",
    color=nsbh_color,
    edgecolor=["black"] * len(NSBH_detect),
    linewidth=1.5,
)
rects3 = ax1.bar(
    x + width,
    BBH_detect,
    width,
    label="BBH",
    color=bbh_color,
    edgecolor=["black"] * len(BBH_detect),
    linewidth=1.5,
)


scatter_size = 60  # Adjust this value as needed

# Plot scatter points for the resampling numbers
ax1.scatter(x - width, BNS_resamp, color=bns_color, marker="o", s=scatter_size)
ax1.scatter(x, NSBH_resamp, color=nsbh_color, marker="o", s=scatter_size)
ax1.scatter(x + width, BBH_resamp, color=bbh_color, marker="o", s=scatter_size)


# Plot lines to connect scatter points to show evolution
ax1.plot(
    x - width,
    BNS_resamp,
    color=bns_color,
    marker="o",
    markersize=np.sqrt(scatter_size),
    zorder=4,
)
ax1.plot(
    x,
    NSBH_resamp,
    color=nsbh_color,
    marker="o",
    markersize=np.sqrt(scatter_size),
    zorder=4,
)
ax1.plot(
    x + width,
    BBH_resamp,
    color=bbh_color,
    marker="o",
    markersize=np.sqrt(scatter_size),
    zorder=4,
)


# Add text annotations for the resampling numbers
for i, value in enumerate(BNS_resamp):
    ax1.text(
        x[i] - width,
        value + value * 0.1,
        f"{value}",
        ha="center",
        color="navy",
        fontsize=18,
        fontweight="bold",
    )
for i, value in enumerate(NSBH_resamp):
    ax1.text(
        x[i],
        value + value * 0.1,
        f"{value}",
        ha="center",
        color="navy",
        fontsize=18,
        fontweight="bold",
    )
    if i == 0:
        ax1.text(
            x[i],
            1e5 / 6,
            "HL",
            ha="center",
            color="k",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman",
        )
    else:
        ax1.text(
            x[i],
            1e5 / 6,
            "HLVK",
            ha="center",
            color="k",
            fontsize=20,
            fontweight="bold",
            fontname="Times New Roman",
        )

for i, value in enumerate(BBH_resamp):
    ax1.text(
        x[i] + width,
        value + value * 0.1,
        f"{value}",
        ha="center",
        color="navy",
        fontsize=18,
        fontweight="bold",
    )


for i in range(len(run_names)):
    if len(run_names) - i != 1:
        ax1.axvline(i + width * 1.7, color="grey", linestyle="--", alpha=0.7)

# Text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel(
    "Detection Number", fontsize=22, fontweight="bold", fontname="Times New Roman"
)
# ax1.set_title('Detections by Observation Run and Population')
ax1.set_xticks(x)
ax1.set_xticklabels(
    run_names, fontsize=22, fontweight="bold", fontname="Times New Roman"
)

ax1.tick_params(axis="y", labelsize=22, labelcolor="k")
for label in ax1.get_yticklabels():
    label.set_fontweight("bold")

legend_fontsize = 20
ax1.legend(
    shadow=True,
    loc="best",
    fontsize=legend_fontsize,
    # title_fontsize='large',
    prop={"size": legend_fontsize, "weight": "bold"},
)

# Add annotations
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(
            "{}".format(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="navy",
            fontsize=20,
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()

# Save the figure
plt.savefig("CBC_detection_number_GWTC-3.pdf")
