# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : July 2024
@description    : This script calculates effective rate densities for BNS, NSBH, and BBH populations
                  and retrieves the simulated rate from a LIGO-LW process table.
---------------------------------------------------------------------------------------------------
"""

import os
import logging
from tqdm.auto import tqdm
from pathlib import Path

from astropy.table import Table, join
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo, z_at_value

from ligo.skymap.io import read_sky_map
from gwpy.table import Table as gwpy_Table

import numpy as np
from scipy import stats
from ligo.skymap.util import sqlite

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


# Input file paths
farah_distribution = "./runs/farah.h5"
sqlite_file = "./runs/O4/farah/events.sqlite"

# Lower 5% and upper 95% quantiles of log normal distribution
rates_table = Table(
    [
        # O3 R&P paper Table II row 1 last column
        {"population": "BNS", "lower": 100.0, "mid": 240.0, "upper": 510.0},
        {"population": "NSBH", "lower": 100.0, "mid": 240.0, "upper": 510.0},
        {"population": "BBH", "lower": 100.0, "mid": 240.0, "upper": 510.0},
    ]
)

# For splitting into BNS, NSBH, and BBH populations
ns_max_mass = 3

# Calculate effective rate density for each sub-population
table = Table.read(farah_distribution)
source_mass1 = table["mass1"]
source_mass2 = table["mass2"]
rates_table["mass_fraction"] = np.asarray(
    [
        np.sum((source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)),
        np.sum((source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)),
        np.sum((source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)),
    ]
) / len(table)

for key in ["lower", "mid", "upper"]:
    rates_table[key] *= rates_table["mass_fraction"]

del table, source_mass1, source_mass2

(standard_90pct_interval,) = np.diff(stats.norm.interval(0.9))
rates_table["mu"] = np.log(rates_table["mid"])
rates_table["sigma"] = (
    np.log(rates_table["upper"]) - np.log(rates_table["lower"])
) / standard_90pct_interval



logging.info("Rates table with calculated mass fractions and statistical parameters (mu, sigma):\n")
logging.info(rates_table)



## Other method when using the SQlITE files

# Simulated rate
with sqlite.open(sqlite_file, "r") as db:
    # Get simulated rate from LIGO-LW process table
    ((result,),) = db.execute(
        "SELECT comment FROM process WHERE program = ?", ("bayestar-inject",)
    )
    sim_rate = u.Quantity(result)

logging.info("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
logging.info("Simulated rate retrieved from the LIGO-LW process table:")
logging.info(sim_rate)
logging.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

logging.info("\n====================================================================================================")

for key in ["BNS", "NSBH", "BBH"]:
    (rates_row,) = rates_table[rates_table["population"] == key]

    #logging.info(f"\n Rate row for population {key}:")
    #logging.info(rates_row)

    sim_rate *= rates_row["mass_fraction"]

    logging.info(f"\n Simulated rate for {key} population adjusted by mass fraction:")
    logging.info(sim_rate)

logging.info("====================================================================================================")
