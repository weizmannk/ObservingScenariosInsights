# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : December 2023
@description    : This script calculates, and prints the Binary Neutron Star (BNS)
                  inspiral range for runs O4a, O4b and O5. It computes the range
                  using SNR=8 and a 1.4 solar mass binary system.
---------------------------------------------------------------------------------------------------
"""

import os
from gwpy.frequencyseries import FrequencySeries
from gwpy.astro import inspiral_range
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Set font and text properties
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "font.size": 12,
        "legend.fontsize": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "figure.figsize": (6, 4),  # Set your desired figure size
        "savefig.dpi": 300,
    }
)

# Ensure grid is behind other plot elements
# plt.rc("axes", axisbelow=True)


# colors = {"L1": "#4ba6ff", "H1": "#9b59b6", "V1": "#ee0000", "K1": "#00FF00"}


class BNSInspiralRangeCalculator:
    def __init__(self, run_names, ifos, path, data_type="ASD"):
        """
        Initialize the BNSInspiralRangeCalculator class.

        :param run_names: List of run names.
        :type run_names: list
        :param ifos: Dictionary of interferometer names.
        :type ifos: dict
        :param path: Path to the directory containing sensitivity data.
        :type path: str
        :param data_type: Type of data, 'ASD' or 'PSD' (default: 'ASD').
        :type data_type: str
        """
        self.run_names = run_names
        self.ifos = ifos
        self.path = path
        self.data_type = data_type
        shared_O4_values = {
            "L1": "aligo_O4high.txt",
            "H1": "aligo_O4high.txt",
            "V1": "asd_O4a_Virgo_78.txt",
            "K1": "kagra_10Mpc.txt",
        }
        self.sensitivity_files = {
            "O4a": {"L1": "asd_O4a_messured_L1.txt", "H1": "asd_O4a_mesured_H1.txt"},
            "O4b": {**shared_O4_values},
            "O4": {**shared_O4_values},
            "O5": {
                "L1": "AplusDesign.txt",
                "H1": "AplusDesign.txt",
                "V1": "avirgo_O5low_NEW.txt",
                "K1": "kagra_128Mpc.txt",
            },
        }

    def load_data(self, run_name, ifo):
        """
        Load sensitivity data based on run_name and ifo.

        :param run_name: Name of the run.
        :type run_name: str
        :param ifo: Interferometer name.
        :type ifo: str
        :return: Frequency and PSD data.
        :rtype: tuple
        """
        path = self.path
        filename = f"{path}/{self.sensitivity_files.get(run_name, {}).get(ifo, '')}"

        if self.data_type.lower() == "asd":
            freq, asd = np.loadtxt(filename, unpack=True)
            psd = asd**2
        elif self.data_type.lower() == "psd":
            freq, psd = np.loadtxt(filename, unpack=True)
        else:
            raise ValueError("Invalid data type. Please specify 'PSD' or 'ASD'.")

        return freq, psd

    def calculate_bns_range(self, snr=8, mass=1.4):
        """
        Calculate BNS inspiral range for provided parameters.

        :param snr: Signal-to-noise ratio (default: 8).
        :type snr: int or float
        :param mass: Mass of the binary neutron stars (default: 1.4).
        :type mass: float
        """

        for run_name in self.run_names:
            print("\n======================================\n")
            print(f"The BNS range in  Run {run_name},")
            if run_name == "O4a":
                print("with the Mesured PSD\n")
            else:
                print("with the Ideal PSD\n")

            for ifo in self.ifos:
                if ifo in self.sensitivity_files.get(run_name):
                    freq, psd = self.load_data(run_name, ifo)
                    fs = FrequencySeries(psd, f0=freq[0], df=freq[1] - freq[0])
                    fmin = 10
                    if ifo == "K1":
                        fmin = 1

                    bns_range = inspiral_range(
                        fs, snr=snr, fmin=fmin, mass1=mass, mass2=mass
                    ).value
                    print(f"{ifo} BNS Inspirial Range: {round(bns_range, 0)} Mpc")
                else:
                    continue
        print("\n======================================\n")

    def plot_sensitivity_curves(self, run_names, outdir):
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        for idx, run_name in enumerate(run_names):
            dashed_handles = []  # Handles for dashed lines
            dashed_labels = []  # Labels for dashed lines
            solid_handles = []  # Handles for solid lines
            solid_labels = []  # Labels for solid lines

            for ifo in self.ifos:
                if ifo in self.sensitivity_files.get(run_name):
                    if run_name == "O4b" and ifo == "H1":
                        continue
                    freq, psd = self.load_data(run_name, ifo)
                    asd = np.sqrt(psd)

                    if run_name == "O4b" and ifo == "L1":
                        label = f"LIGO (L1-H1)"
                        linestyle = "--" if run_name == "O4a" else "-"
                    else:
                        label = f"{ifo}"
                        linestyle = "--" if run_name == "O4a" else "-"

                    (line,) = axs[idx].loglog(
                        freq,
                        asd,
                        label=label,
                        linewidth=1,
                        alpha=1.0,
                        linestyle=linestyle,
                    )

                    if linestyle == "--":
                        dashed_handles.append(line)
                        dashed_labels.append(label)
                    else:
                        solid_handles.append(line)
                        solid_labels.append(label)

            axs[idx].set_xlabel(r"$\mathrm{Frequency}\,\mathrm{[Hz]}$", fontsize=12)
            axs[idx].set_ylabel(r"$\mathrm{ASD}\,[1/\sqrt{\mathrm{Hz}}]$", fontsize=12)
            axs[idx].tick_params(labelsize=10)  # Adjust tick label size
            axs[idx].set_xlim([10, 4000])
            axs[idx].set_ylim(1e-24, 1e-19)
            axs[idx].grid(True)
            axs[idx].set_title(f"Run {run_name}", fontsize=14)

            legend = axs[idx].legend(
                handles=dashed_handles if run_name == "O4a" else solid_handles,
                labels=dashed_labels if run_name == "O4a" else solid_labels,
                loc="upper center" if run_name == "O4a" else "upper center",
                fontsize=10,  # Adjust legend font size
            )
            # Assign colors to legend labels
            for lh, ll in zip(legend.legend_handles, legend.get_texts()):
                ll.set_color(lh.get_color())

        plt.tight_layout()
        plt.savefig(f"{outdir}/Strain_HLV_subplots.png", dpi=300)
        plt.close()


# Outdir

outdir = "../Plots"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# the runs data direction
path = Path("../data/ASD")

IFOs = ["L1", "H1", "V1", "K1"]
run_names = ["O4a", "O4b", "O5"]
data_type = "ASD"  # You could give PSD if the input is PSD

calculator = BNSInspiralRangeCalculator(
    run_names=run_names, ifos=IFOs, path=path, data_type=data_type
)
calculator.calculate_bns_range()

calculator.plot_sensitivity_curves(run_names=["O4a", "O4b"], outdir=outdir)
