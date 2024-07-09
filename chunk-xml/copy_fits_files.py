# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
---------------------------------------------------------------------------------------------------
Author          : Ramodgwendé Weizmann KIENDREBEOGO
Email           : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
Repository      : https://github.com/weizmannk/ObservingScenariosInsights.git
Created On      : July 2023
Description     : Script for copying BNS and NSBH skymap files from simulation data runs.
                  The script reads injection data, creates output directories for BNS and NSBH populations,
                  and copies the corresponding FITS and XML files to the appropriate directories.
---------------------------------------------------------------------------------------------------
"""

import shutil
from tqdm.auto import tqdm
from pathlib import Path
from astropy.table import Table

# Directory containing simulation data runs
datapath = "./runs"

# For splitting into BNS, NSBH populations
ns_max_mass = 3.0

pops = ["BNS", "NSBH"]
run_names = run_dirs = ["O4"]

with tqdm(total=len(run_names) * len(pops), desc="Total Progress") as total_progress:
    for run_name, run_dir in zip(run_names, run_dirs):
        path = Path(f"{datapath}/{run_dir}/farah")
        
        allsky = Table.read(f"{path}/allsky.dat", format="ascii.fast_tab")
        
        for pop in pops:
            with tqdm(total=1, desc=f"Processing {pop} {run_name}") as pop_progress:
                # Read injections.dat files
                injections = Table.read(f"{path}/subpopulations/{pop.lower()}_{run_name}_injections.dat", format="ascii.fast_tab")
                simulation_IDs = injections["simulation_id"]
                
                # Create a mask for filtering
                mask = [id in simulation_IDs for id in allsky["simulation_id"]]
                filtered_allsky = allsky[mask]
                filtered_allsky.write(
                    Path(f"{path}/runs/O4/farah/{pop.lower()}_{run_name}_allsky.dat"),
                    format="ascii.tab",
                    overwrite=True
                )
                
                # Create output directories to save split BNS and NSBH results.
                allsky_dir = Path(f"{path}/runs/O4/farah/{pop.lower()}_allsky")
                events_dir = Path(f"{path}/runs/O4/farah/{pop.lower()}_events")

                allsky_dir.mkdir(parents=True, exist_ok=True)
                events_dir.mkdir(parents=True, exist_ok=True)
                
                with tqdm(total=len(simulation_IDs), desc=f"Copying files for {pop} {run_name}") as file_progress:
                    for ID in simulation_IDs:
                        fits_file = Path(f"{path}/allsky/{ID}.fits")
                        xml_file = Path(f"{path}/events/{ID}.xml.gz")

                        if fits_file.exists() and xml_file.exists():
                            shutil.copy(fits_file, allsky_dir)
                            shutil.copy(xml_file, events_dir)
                        
                        file_progress.update(1)
                
                pop_progress.update(1)
            total_progress.update(1)

