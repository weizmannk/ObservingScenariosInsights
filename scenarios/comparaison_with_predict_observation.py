# Summary plots and tables

# Imports
import os
import io
from pathlib import Path

# gwpy hijacks Matplotlib's default axes class
# (https://github.com/gwpy/gwpy/issues/1187).
# Take it back and restore default style.
import matplotlib
import gwpy

matplotlib.projections.register_projection(matplotlib.axes.Axes)
matplotlib.rcdefaults()

from astropy.table import join, Table
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo, z_at_value
from matplotlib import pyplot as plt
import gwpy.table  # to read ligolw documents as Astropy tables
from IPython.display import Markdown
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
from scipy import stats
from scikits import bootstrap
import seaborn
from tqdm.auto import tqdm

from astropy.table import Column

#%matplotlib inline
# plt.style.use('seaborn-v0_8-paper')


def O4_BBH_alert():

    events_O4a = [
        {"name": "S230601bf", "distance": 3565, "error": 1260, "cr": 2531},
        {"name": "S230605o", "distance": 1067, "error": 333, "cr": 1077},
        {"name": "S230606d", "distance": 2545, "error": 874, "cr": 1221},
        {"name": "S230608as", "distance": 3447, "error": 1079, "cr": 1694},
        {"name": "S230609u", "distance": 3390, "error": 1125, "cr": 1287},
        {"name": "S230624av", "distance": 2124, "error": 682, "cr": 1024},
        {"name": "S230628ax", "distance": 2047, "error": 585, "cr": 705},
        {"name": "S230630am", "distance": 5336, "error": 2001, "cr": 3965},
        {"name": "S230630bq", "distance": 999, "error": 286, "cr": 1215},
        {"name": "S230702an", "distance": 2428, "error": 849, "cr": 2267},
        {"name": "S230704f", "distance": 2759, "error": 992, "cr": 1700},
        {"name": "S230706ah", "distance": 1962, "error": 594, "cr": 1497},
        {"name": "S230702an", "distance": 2428, "error": 849, "cr": 2267},
        {"name": "S230704f", "distance": 2759, "error": 992, "cr": 1700},
        {"name": "S230706ah", "distance": 1962, "error": 594, "cr": 1497},
        {"name": "S230707ai", "distance": 4074, "error": 1485, "cr": 3181},
        {"name": "S230708cf", "distance": 3336, "error": 1076, "cr": 2032},
        {"name": "S230708t", "distance": 3010, "error": 988, "cr": 1227},
        {"name": "S230708z", "distance": 4647, "error": 1696, "cr": 3372},
        {"name": "S230708cf", "distance": 3336, "error": 1076, "cr": 2032},
        {"name": "S230709bi", "distance": 4364, "error": 1585, "cr": 2644},
        {"name": "S230723ac", "distance": 1551, "error": 436, "cr": 1117},
        {"name": "S230726a", "distance": 2132, "error": 714, "cr": 27774},
        {"name": "S230729z", "distance": 1495, "error": 444, "cr": 1945},  # issue
        {"name": "S230731an", "distance": 1001, "error": 242, "cr": 598},
        {"name": "S230802aq", "distance": 576, "error": 246, "cr": 25885},
        {"name": "S230805x", "distance": 3305, "error": 1113, "cr": 2094},
        {"name": "S230806ak", "distance": 5423, "error": 1862, "cr": 3715},
        {"name": "S230807f", "distance": 5272, "error": 1900, "cr": 5436},
        {"name": "S230811n", "distance": 1905, "error": 672, "cr": 810},
        {"name": "S230814r", "distance": 3788, "error": 1416, "cr": 3389},
        {"name": "S230814ah", "distance": 330, "error": 105, "cr": 25260},  # issue
        {"name": "S230819ax", "distance": 4216, "error": 1645, "cr": 4044},
        {"name": "S230820bq", "distance": 3600, "error": 1437, "cr": 1373},  # issue
        {"name": "S230822bm", "distance": 5154, "error": 1771, "cr": 3974},
        {"name": "S230824r", "distance": 4701, "error": 1563, "cr": 3279},
        {"name": "S230825k", "distance": 5283, "error": 2117, "cr": 3012},
        {"name": "S230831e", "distance": 4900, "error": 2126, "cr": 3803},
        {"name": "S230904n", "distance": 1095, "error": 327, "cr": 2015},
        {"name": "S230911ae", "distance": 1623, "error": 584, "cr": 27759},
        {"name": "S230914ak", "distance": 2676, "error": 827, "cr": 1532},
        {"name": "S230919bj", "distance": 1491, "error": 402, "cr": 708},  # missing
        {"name": "S230920al", "distance": 3139, "error": 1003, "cr": 2180},  # missing
        {"name": "S230922g", "distance": 1491, "error": 443, "cr": 332},
        {"name": "S230922q", "distance": 6653, "error": 2348, "cr": 4658},
        {"name": "S230924an", "distance": 2558, "error": 596, "cr": 835},
        {"name": "S230927l", "distance": 2966, "error": 1041, "cr": 1177},
        {"name": "S230927be", "distance": 1059, "error": 289, "cr": 298},
        {"name": "S230928cb", "distance": 4060, "error": 1553, "cr": 3102},
        {"name": "S230930al", "distance": 4902, "error": 1671, "cr": 3166},
        {"name": "S231001aq", "distance": 4425, "error": 1946, "cr": 3181},
        {"name": "S231005j", "distance": 6417, "error": 2246, "cr": 5480},
        {"name": "S231005ah", "distance": 3707, "error": 1335, "cr": 2497},
        {"name": "S231008ap", "distance": 3531, "error": 1320, "cr": 3102},
        {"name": "S231014r", "distance": 2857, "error": 903, "cr": 1807},
        {"name": "S231020ba", "distance": 1168, "error": 361, "cr": 1339},  # issue
        {"name": "S231020bw", "distance": 2620, "error": 694, "cr": 386},
        {"name": "S231028bg", "distance": 4221, "error": 923, "cr": 1207},
        {"name": "S231029y", "distance": 3292, "error": 1313, "cr": 29973},  # issue
        {"name": "S231102w", "distance": 3493, "error": 1015, "cr": 2343},  # issue
        {"name": "S231104ac", "distance": 1357, "error": 321, "cr": 759},  # issue
        {"name": "S231108u", "distance": 1986, "error": 494, "cr": 949},
        {"name": "S231110g", "distance": 1849, "error": 533, "cr": 636},
        {"name": "S231113bb", "distance": 3260, "error": 1181, "cr": 2172},
        {"name": "S231113bw", "distance": 1186, "error": 376, "cr": 1713},
        {"name": "S231114n", "distance": 1317, "error": 407, "cr": 1267},
        {"name": "S231118ab", "distance": 4353, "error": 1588, "cr": 2898},
        {"name": "S231118an", "distance": 1337, "error": 347, "cr": 1107},
        {"name": "S231118d", "distance": 2109, "error": 585, "cr": 956},
        {"name": "S231119u", "distance": 6597, "error": 2556, "cr": 5212},
        {"name": "S231123cg", "distance": 1148, "error": 338, "cr": 2714},
        {"name": "S231127cg", "distance": 4425, "error": 1718, "cr": 3450},
        {"name": "S231129ac", "distance": 3964, "error": 1513, "cr": 3089},
        {"name": "S231206ca", "distance": 3230, "error": 1141, "cr": 2335},
        {"name": "S231206cc", "distance": 1467, "error": 264, "cr": 342},
        {"name": "S231213ap", "distance": 3861, "error": 1257, "cr": 1469},
        {"name": "S231223j", "distance": 4468, "error": 1602, "cr": 3520},
        {"name": "S231224e", "distance": 863, "error": 213, "cr": 394},
        {"name": "S231226av", "distance": 1218, "error": 171, "cr": 199},
        {"name": "S231231ag", "distance": 1066, "error": 339, "cr": 27061},
        {"name": "S240104bl", "distance": 1978, "error": 618, "cr": 27949},
        {"name": "S240107b", "distance": 6089, "error": 2429, "cr": 4143},
        {"name": "S240109a", "distance": 1594, "error": 567, "cr": 28049},
        {"name": "S240104bl", "distance": 1978, "error": 618, "cr": 27949},
        {"name": "S240107b", "distance": 6089, "error": 2429, "cr": 4143},
        {"name": "S240109a", "distance": 1594, "error": 567, "cr": 28049},
    ]

    events_O4b = [
        # {"name": "S240406aj", "distance": 2449, "error": 692, "cr": 1724},
        {"name": "S240413p", "distance": 526, "error": 101, "cr": 34},
        {"name": "S240421ar", "distance": 7702, "error": 2899, "cr": 2601},
        {"name": "S240426dl", "distance": 5886, "error": 2242, "cr": 3469},
        {"name": "S240426s", "distance": 3452, "error": 1295, "cr": 3050},
        {"name": "S240428dr", "distance": 831, "error": 145, "cr": 186},
        {"name": "S240430ca", "distance": 6212, "error": 2593, "cr": 4061},
        {"name": "S240501an", "distance": 4022, "error": 1460, "cr": 1079},
        {"name": "S240505av", "distance": 4570, "error": 1415, "cr": 1469},
        {"name": "S240507p", "distance": 1328, "error": 370, "cr": 279},
        {"name": "S240511i", "distance": 1906, "error": 404, "cr": 85},
        {"name": "S240512r", "distance": 1082, "error": 266, "cr": 216},
        {"name": "S240513ei", "distance": 2254, "error": 458, "cr": 37},
        {"name": "S240514c", "distance": 4182, "error": 1833, "cr": 30758},
        {"name": "S240514x", "distance": 2594, "error": 587, "cr": 142},
        {"name": "S240515m", "distance": 3559, "error": 976, "cr": 978},
        {"name": "S240520cv", "distance": 1289, "error": 332, "cr": 370},
        {"name": "S240525p", "distance": 4337, "error": 1519, "cr": 1517},
        {"name": "S240527en", "distance": 7238, "error": 2059, "cr": 1779},
        {"name": "S240527fv", "distance": 1119, "error": 188, "cr": 15},
        {"name": "S240530a", "distance": 1229, "error": 393, "cr": 984},
    ]

    O4a_table = O4a_distance = [event["distance"] for event in events_O4a]
    O4b_table = O4b_distance = [event["distance"] for event in events_O4b]

    # O4a_table = Table([O4a_distance], names=('distance',))
    # O4b_table = Table([O4b_distance], names=('distance',))

    O4a_table = Column(O4a_distance, name="distance", dtype="float64")
    O4b_table = Column(O4b_distance, name="distance", dtype="float64")

    return O4a_table, O4b_table


plt.style.use("seaborn-v0_8-deep")

outdir = "../output"

data_dir = "./runs"


if not os.path.isdir(outdir):
    os.makedirs(outdir)

alpha = 0.9  # Confidence band for histograms
run_names = run_dirs = ["O4a", "O4b"]
pops = ["BBH"]  # Populations
classification_names = pops
classification_colors = seaborn.color_palette(
    "tab10"
)  # seaborn.color_palette(n_colors=len(classification_names))
fieldnames = ["distance"]
fieldlabels = ["Luminosity distance (Mpc)"]


## Fiducial rates in Gpc$^{-3}$ yr$^{-1}$

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
table = Table.read("farah.h5")
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

rates_table

fiducial_log_rates = np.asarray(rates_table["mu"])
fiducial_log_rate_errs = np.asarray(rates_table["sigma"])

## Load all data sets

tables = {}
for run_name, run_dir in zip(tqdm(run_names), run_dirs):
    path = Path(f"{data_dir}") / run_dir / "farah"
    allsky = Table.read(str(path / "allsky.dat"), format="ascii.fast_tab")
    injections = Table.read(str(path / "injections.dat"), format="ascii.fast_tab")
    allsky.rename_column("coinc_event_id", "event_id")
    injections.rename_column("simulation_id", "event_id")
    table = join(allsky, injections)

    # Convert from Mpc^3 to 10^6 Mpc^3
    for colname in ["searched_vol", "vol(20)", "vol(50)", "vol(90)"]:
        table[colname] *= 1e-6

    # Get simulated rate from LIGO-LW process table
    process_table = Table.read(
        str(path / "events.xml.gz"), format="ligolw", tablename="process"
    )
    table.meta["rate"] = u.Quantity(process_table[0]["comment"])
    table.meta["network"] = process_table[1]["ifos"].replace("1", "").replace(",", "")

    # Get number of Monte Carlo samples from LIGO-LW process_params table
    process_params_table = Table.read(
        str(path / "events.xml.gz"), format="ligolw", tablename="process_params"
    )
    (table.meta["nsamples"],) = process_params_table[
        process_params_table["param"] == "--nsamples"
    ]["value"].astype(int)

    # Split by source frame mass
    z = z_at_value(cosmo.luminosity_distance, table["distance"] * u.Mpc).to_value(
        u.dimensionless_unscaled
    )
    zp1 = z + 1
    source_mass1 = table["mass1"] / zp1
    source_mass2 = table["mass2"] / zp1
    tables[run_name] = {}
    # Note: copy() below so that we deep-copy table.meta
    # tables[run_name]["BNS"] = table[
    #    (source_mass1 < ns_max_mass) & (source_mass2 < ns_max_mass)
    # ].copy()
    # tables[run_name]["NSBH"] = table[
    #     (source_mass1 >= ns_max_mass) & (source_mass2 < ns_max_mass)
    # ].copy()
    tables[run_name]["BBH"] = table[
        (source_mass1 >= ns_max_mass) & (source_mass2 >= ns_max_mass)
    ].copy()

    for key in ["BBH"]:
        (rates_row,) = rates_table[rates_table["population"] == key]
        tables[run_name][key].meta["rate"] *= rates_row["mass_fraction"]

    del (
        allsky,
        injections,
        table,
        process_table,
        process_params_table,
        z,
        zp1,
        source_mass1,
        source_mass2,
    )


# add O4 alert from Sarah


## Load old Living Review data sets

old_tables = {}


for pop in tqdm(["BBH"]):
    url_root = f"https://git.ligo.org/emfollow/obs-scenarios-2019-fits-files/-/raw/master/O3_HLV/{pop.lower()}_astro/"
    allsky = Table.read(f"{url_root}/allsky.dat", format="ascii")
    injections = Table.read(f"{url_root}/injections.dat", format="ascii")
    coincs = Table.read(f"{url_root}/coincs.dat", format="ascii")
    table = join(allsky, coincs, "coinc_event_id")
    table = join(table, injections, "simulation_id")
    table["vol(90)"] *= 1e-6
    old_tables[pop] = table
    del allsky, injections, coincs, table


## Cumulative histograms


tables["O3"] = {}
tables["O3"] = old_tables


run_names_add = ["O4 HL", "O4 HLVK", "O3 HLV"]
axs = [plt.subplots()[1] for _ in range(len(fieldnames))]
colors = seaborn.color_palette("colorblind", len(run_names_add))
linestyles = ["-", "--"]

pops = ["simulation", "alerts"]

for ax, fieldlabel in zip(axs, fieldlabels):
    ax.set_xlabel(fieldlabel)

for ax in axs:
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1])
    ax.set_ylabel("Cumulative fraction of events")
    ax.legend(
        [
            plt.Line2D([], [], linestyle=linestyle, color="black")
            for linestyle in linestyles
        ]
        + [plt.Rectangle((0, 0), 0, 0, facecolor=color) for color in colors],
        pops + run_names_add,
    )

axs[0].set_xlim(10, 1e5)

ax = axs[0]
zax = ax.twiny()
zax.set_xlim(*ax.get_xlim())
zax.set_xscale(ax.get_xscale())

zax.minorticks_off()
n = np.arange(2, 10)
z = np.concatenate([0.001 * n, 0.01 * n, 0.1 * n, n])
minor = cosmo.luminosity_distance(z).to_value(u.Mpc)
minor = minor[minor > ax.get_xlim()[0]]
minor = minor[minor < ax.get_xlim()[1]]
zax.set_xticks(minor, minor=True)
zax.set_xticklabels([], minor=True)

z = [0.01, 0.1, 1]
zax.set_xticks(cosmo.luminosity_distance(z).to_value(u.Mpc))
zax.set_xticklabels([str(_) for _ in z])
zax.set_xlabel("Redshift")

for irun, (run_name, tables1) in enumerate(tables.items()):
    print(irun)

    for ipop, (pop, table) in enumerate(tables1.items()):
        print("******************")
        print(ipop, pop)

        for ifield, fieldname in enumerate(fieldnames):
            data = table[fieldname]
            data = data[np.isfinite(data)]
            ax = axs[ifield]
            t = np.geomspace(*ax.get_xlim(), 100)
            kde = stats.gaussian_kde(np.asarray(np.log(data)))
            ((std,),) = np.sqrt(kde.covariance)
            y = (
                stats.norm(kde.dataset.ravel(), std)
                .cdf(np.log(t)[:, np.newaxis])
                .mean(1)
            )

            if run_name == "O4a":
                irun = 0
            elif run_name == "O4b":
                irun = 1
            elif run_name == "O3":
                irun = 2

            if run_name == "O4a" or run_name == "O4b":

                ax.plot(t, y, color=colors[irun], linestyle="-")
            else:
                ax.plot(t, y, color=colors[irun], linestyle="--")


# add O4 alert from Sarah
O4a_distance, O4b_distance = O4_BBH_alert()
for alert in ["O4a_alerts", "O4b_alerts"]:

    if alert == "O4a_alerts":
        data = O4a_distance[np.isfinite(O4a_distance)]
    else:
        data = O4b_distance[np.isfinite(O4b_distance)]

    ax = axs[0]
    t = np.geomspace(*ax.get_xlim(), 100)
    kde = stats.gaussian_kde(np.asarray(np.log(data)))
    ((std,),) = np.sqrt(kde.covariance)
    y = stats.norm(kde.dataset.ravel(), std).cdf(np.log(t)[:, np.newaxis]).mean(1)

    if alert == "O4a_alerts":
        irun = 0
    elif alert == "O4b_alerts":
        irun = 1

    ax.plot(t, y, color=colors[irun], linestyle="--")


for ax, fieldname in zip(axs, fieldnames):
    ax.figure.savefig(f"{outdir}/{fieldname}_sarah_alerts_vs_sim.pdf")
    ax.figure.savefig(f"{outdir}/{fieldname}_sarah_alerts_vs_sim.svg")
