import pandas as pd
import numpy as np
import random

import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


telescope = "ztf"
type = "BNS"
lcs = pd.read_csv(telescope + "_" + type + ".csv")
lcs = lcs[lcs["mag_err"] != 99.0]

lcs.groupby("sim").ngroups

band_colors = {
    "g": "limegreen",
    "r": "orangered",
    "i": "goldenrod",
}  # ,'limegreen','darkturquoise',

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0.02, right=0.98, bottom=0.1, top=0.98, wspace=0.3, hspace=0.3)
rows, cols = 3, 2
gs = gridspec.GridSpec(rows, cols)
sax = []
for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols * r + c]))

hist, bins = np.histogram(
    lcs.groupby(["sim"]).apply(lambda x: x.shape[0]), bins=np.arange(-0.5, 10.0, 1)
)
for name, group in (
    lcs.groupby(["sim", "passband"]).apply(lambda x: x.shape[0]).groupby(level=1)
):
    hist, bins = np.histogram(group, bins=np.arange(-0.5, 10.0, 1))
    sax[0].step(
        bins,
        np.pad(hist, (1, 0)) / hist.max(),
        alpha=0.5,
        color=band_colors[name],
        lw=2,
    )
hist, bins = np.histogram(
    lcs.groupby(["sim"]).apply(lambda x: x.shape[0]), bins=np.arange(-0.5, 10.0, 1)
)
sax[0].step(bins, np.pad(hist, (1, 0)) / hist.max(), color="k", lw=0.5)

for name, group in (
    lcs.groupby(["sim", "passband"]).apply(lambda x: x["mag"].min()).groupby(level=1)
):
    hist, bins = np.histogram(group, bins=np.arange(16, 22, 0.5))
    sax[1].step(
        bins,
        np.pad(hist, (1, 0)) / hist.max(),
        alpha=0.5,
        color=band_colors[name],
        lw=2,
    )
hist, bins = np.histogram(
    lcs.groupby(["sim"]).apply(lambda x: x["mag"].min()), bins=np.arange(16, 22, 0.5)
)
sax[1].step(bins, np.pad(hist, (1, 0)) / hist.max(), color="k", lw=0.5)

for name, group in (
    lcs.groupby(["sim", "passband"])
    .apply(lambda x: (x["mjd"] - x["tc"])[x["mag"].idxmin()])
    .groupby(level=1)
):
    hist, bins = np.histogram(group, bins=np.arange(0, 4.5, 0.2))
    sax[2].step(
        bins,
        np.pad(hist, (1, 0)) / hist.max(),
        alpha=0.5,
        color=band_colors[name],
        lw=2,
    )
hist, bins = np.histogram(
    lcs.groupby(["sim"]).apply(lambda x: (x["mjd"] - x["tc"])[x["mag"].idxmin()]),
    bins=np.arange(0, 4.5, 0.2),
)
sax[2].step(bins, np.pad(hist, (1, 0)) / hist.max(), color="k", lw=0.5)

# lcs.groupby(['sim','passband']).apply(lambda x: (x['mjd']-x['tc'])[x['mag'].idxmin()]).groupby(level=1).apply(lambda x: sax[2].hist(x,alpha=0.3,color=band_colors[x.name]))

sax[0].set_xlabel("num photo")
sax[0].get_yaxis().set_visible(False)
sax[1].set_xlabel("peak mag")
sax[1].get_yaxis().set_visible(False)
sax[2].set_xlabel("time max since KN trigger")
sax[2].get_yaxis().set_visible(False)

sax[3].scatter(
    lcs.groupby(["sim"]).apply(lambda x: x.shape[0]),
    lcs.groupby(["sim"]).apply(lambda x: x["mag"].min()),
    s=1,
)
sax[3].set_xlabel("num photo (all)")
sax[3].set_ylabel("peak mag (all)")
sax[3].invert_yaxis()

for name, group in (
    lcs.groupby(["sim", "passband"])
    .apply(lambda x: (x["mjd"] - x["tc"])[x["mjd"].idxmin()])
    .groupby(level=1)
):
    hist, bins = np.histogram(group, bins=np.arange(0, 4.5, 0.2))
    sax[4].step(
        bins,
        np.pad(hist, (1, 0)) / hist.max(),
        alpha=0.5,
        color=band_colors[name],
        lw=2,
    )
hist, bins = np.histogram(
    lcs.groupby(["sim"]).apply(lambda x: x["mjd"].min() - x["tc"][x["mjd"].idxmin()]),
    bins=np.arange(0, 4.5, 0.2),
)
sax[4].step(bins, np.pad(hist, (1, 0)) / hist.max(), color="k", lw=0.5)

sax[4].set_xlabel("time first detection since KN trigger")
sax[4].get_yaxis().set_visible(False)

fig.savefig(telescope + "_" + type + "_stats.png", dpi=300)

fig = plt.figure(figsize=(10, 10))
fig.subplots_adjust(
    left=0.07, right=0.98, bottom=0.07, top=0.98, wspace=0.0, hspace=0.0
)
rows, cols = 6, 5
gs = gridspec.GridSpec(rows, cols)
sax = []
for r in range(rows):
    for c in range(cols):
        sax.append(plt.subplot(gs[cols * r + c]))

s_bool = {True: 4, False: 2}
i = 0
for sim, LC in lcs[
    lcs["sim"].isin(random.sample(lcs["sim"].unique().tolist(), rows * cols))
].groupby("sim"):
    for name, group in LC.groupby(["passband", "ToO"]):
        sax[i].errorbar(
            group["mjd"] - group["tc"],
            group["mag"],
            yerr=group["mag_err"],
            fmt="o",
            ms=s_bool[name[1]],
            c=band_colors[name[0]],
        )
    sax[i].set_xlabel("time since KN trigger")
    sax[i].set_xlim(0.0, 8.0)
    sax[i].set_ylim(22.0, 16.0)
    if i % cols == 0:
        sax[i].set_ylabel("mag")
    else:
        sax[i].set_yticklabels([])
    i += 1

fig.savefig(telescope + "_" + type + "_sample.png", dpi=300)
plt.close()
