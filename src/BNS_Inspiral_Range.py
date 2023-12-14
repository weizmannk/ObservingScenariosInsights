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


# the runs data direction
path = Path("../data/ASD")

IFOs = ["L1", "H1", "V1", "K1"]
run_names = ["O4a", "O4b", "O5"]
data_type = "ASD"  # You could give PSD if the input is PSD

calculator = BNSInspiralRangeCalculator(
    run_names=run_names, ifos=IFOs, path=path, data_type=data_type
)
calculator.calculate_bns_range()
