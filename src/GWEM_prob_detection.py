# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : December 2023
Description     : Tools for reading and analyzing the probabilities of telescopes' EM detection
                  with gwemopt and nmma combination.
---------------------------------------------------------------------------------------------------
"""

import os
import numpy as np
import json
import pandas as pd
from astropy.table import Table, join
from pathlib import Path
from tqdm.auto import tqdm

# data are on LHO " ssh weizmann.kiendrebeogo@ldas-pcdev1.ligo-wa.caltech.edu"

# Define directory paths and telescope configurations
data_dir = Path("/home/weizmann.kiendrebeogo/ULTRASAT").expanduser()
output_dir = Path("./outdir").expanduser()
telescopes = ["LSST", "ZTF"]
run_names = ["O4", "O4a", "O5"]
models = ["Bu2019lm"]

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate through telescopes, runs, and models to process EM detection probabilities
with tqdm(
    total=len(telescopes) * len(run_names) * len(models),
    desc="Read EM Detection Probabilities",
) as progress:
    for telescope in telescopes:
        for run_name in run_names:
            for model in models:
                # Define population type based on model
                pop = "BNS" if model == "Bu2019lm" else "NSBH"

                label = f"injection_{model}_{pop}_{run_name}"
                injection_file = (
                    data_dir
                    / telescope
                    / run_name
                    / "absolute_mag_lc"
                    / f"outdir_{pop}"
                    / f"injection_{model}_{run_name}.json"
                )
                data_path = (
                    data_dir / telescope / run_name / "detection_mag" / f"outdir_{pop}"
                )

                # Initialize arrays for detection and efficiency metrics
                detection, efficiency, efficiency_err = [], [], []

                # Read and process injection file if it's JSON
                if injection_file.suffix == ".json":
                    try:
                        with open(injection_file, "rb") as f:
                            injection_data = json.load(f)
                            datadict = injection_data["injections"]["content"]
                            dataframe_from_inj = pd.DataFrame.from_dict(datadict)
                            simulation_id = dataframe_from_inj["simulation_id"]
                    except FileNotFoundError:
                        print(f"File not found: {injection_file}")
                        continue
                else:
                    print("Only JSON files supported.")
                    continue

                # Process each injection data for efficiency and detection metrics
                if len(dataframe_from_inj) > 0:
                    for index, row in dataframe_from_inj.iterrows():
                        path = data_path / str(index)
                        efffile_true = (
                            path / f"efficiency_true_{label}_{simulation_id[index]}.txt"
                        )

                        detection.append(1 if np.loadtxt(efffile_true) == 1 else 0)
                        efffile = path / "efficiency.txt"

                        try:
                            if efffile.exists():
                                data = Table.read(efffile, format="ascii.fast_tab")
                                efficiency.append(data["efficiencyMetric"][0])
                                efficiency_err.append(data["efficiencyMetric_err"][0])
                            else:
                                efficiency.append(0)
                                efficiency_err.append(0)
                        except FileNotFoundError as e:
                            print(f"File missing: {efffile}")

                # Compile and save results into a table
                efficiency, efficiency_err = map(
                    lambda x: np.array(x, dtype="float64"), [efficiency, efficiency_err]
                )
                data_dict = {
                    "simulation_id": simulation_id,
                    f"{telescope}_efficiency": efficiency,
                    f"{telescope}_efficiency_err": efficiency_err,
                    f"{telescope}_detection": detection,
                }
                Table(data_dict).write(
                    output_dir / f"{telescope}_{run_name}_{pop}_EM_detection.dat",
                    format="ascii.tab",
                    overwrite=True,
                )
                progress.update()

# Combine telescope EM detection results with observing scenarios injections parameters
obs_dir = Path(
    "home/weizmann.kiendrebeogo/GitHubStock/obs-scenarios-data-2022/nmma_GWdata/Farah/runs/"
).expanduser()

with tqdm(total=len(run_names) * len(models), desc="Combining Results") as progress:
    for run_name in run_names:
        for model in models:
            pop = "BNS" if model == "Bu2019lm" else "NSBH"
            path = (
                obs_dir
                / run_name
                / f"{pop.lower()}_farah/{pop.lower()}_{run_name}_injections.dat"
            )

            try:
                injections = Table.read(str(path), format="ascii.fast_tab")
                tables = injections

                # Join EM detection data with the telescope simulation data
                for telescope in telescopes:
                    EM_data = Table.read(
                        output_dir / f"{telescope}_{run_name}_{pop}_EM_detection.dat",
                        format="ascii.fast_tab",
                    )
                    tables = join(tables, EM_data)
                    del EM_data

                tables.write(
                    output_dir / f"gwem_{run_name}_{pop}_detection.dat",
                    format="ascii.tab",
                    overwrite=True,
                )
                progress.update()
            except FileNotFoundError as e:
                print(f"File not found: {path}")
                continue
