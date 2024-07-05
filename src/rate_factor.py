import os
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

# Input file paths
farah_distribution = "farah.h5"
sqlite_file = "events.sqlite"

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

print(rates_table)

# Simulated rate
with sqlite.open(sqlite_file, "r") as db:
    # Get simulated rate from LIGO-LW process table
    ((result,),) = db.execute(
        "SELECT comment FROM process WHERE program = ?", ("bayestar-inject",)
    )
    sim_rate = u.Quantity(result)

print(sim_rate)

for key in ["BNS", "NSBH", "BBH"]:
    (rates_row,) = rates_table[rates_table["population"] == key]

    print(rates_row)
    sim_rate *= rates_row["mass_fraction"]  # .to_value(u.Gpc**-3 * u.yr**-1)

    print(sim_rate)
