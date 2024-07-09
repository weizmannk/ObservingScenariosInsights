from scipy import stats
from astropy.table import Table
import numpy as np

n_samples = 1000000

# Atrophysique Binaries Black Hole  
# minimal and  max spin
bh_astro_spin_min = -0.99
bh_astro_spin_max = +0.99
bh_mass_min = 5.0
bh_mass_max = 50.0

mass_distn = stats.pareto(b=1.3)
spin_distn = stats.uniform(bh_astro_spin_min, bh_astro_spin_max - bh_astro_spin_min)

# censor the mass distribution, remove or ovoid a non BBH-astro mass 
def draw_masses(n_samples):
    nbad = n_samples
    mass = np.empty(n_samples)
    bad = np.ones(n_samples, dtype=bool)
    while nbad > 0:
        mass[bad] = mass_distn.rvs(nbad)
        bad = (mass < bh_mass_min) | (mass > bh_mass_max)
        nbad = np.sum(bad)
    return mass


# black hole masses
mass1 = draw_masses(n_samples)
mass2 = draw_masses(n_samples)


# swap masses to ensure that mass1 >= mass2 
swap = mass1 < mass2
mass1[swap], mass2[swap] = mass2[swap].copy(), mass1[swap].copy()

# We could simply use this one swap
#mass1, mass2 = np.maximum(mass1, mass2), np.minimum(mass1, mass2)

# black hole spin 
spin1z = spin_distn.rvs(n_samples)
spin2z = spin_distn.rvs(n_samples)

# save data on .h5 file 

Table({
    'mass1' : mass1,
    'mass2' : mass2,
    'spin1z': spin1z,
    'spin2z': spin2z
}).write(
    "distribution_samples.h5", overwrite=True
)


# Histogram plot 

data = Table.read("distribution_samples.h5").to_pandas()

hist = data[['mass1', 'mass2', 'spin1z', 'spin2z']].hist(bins=1000, sharey=True)
plt.savefig("distribution_samples.png", dpi=200)


# In case where we have all populations on the same file 
# abd  we need to  slpit the disribbution as a diferent populations 

split_data = Table.read("farah.h5")


bns = split_data[(split_data['mass1'] <= 2.5)]

bbh = split_data[(split_data['mass2'] >= 5)]

nsbh = split_data[(split_data['mass1'] >= 5) & ( split_data['mass2'] <= 2.5) ]

bns.write('farah_bns.h5', overwrite=True)
bbh.write('farah_bbh.h5', overwrite=True)
nsbh.write('farah_nsbh.h5', overwrite=True)
