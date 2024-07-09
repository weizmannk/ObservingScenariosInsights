## Sqlite
from ligo.skymap.util import sqlite

## Fits files 
from ligo.skymap.io import read_sky_map

## XML
from gwpy.table import Table as gwpy_Table

##Read .fits files
# for more please see this,  https://lscsoft.docs.ligo.org/ligo.skymap/io/fits.html
fits_data =  read_sky_map('0.fits')


## Read sqlite , some examples 
with sqlite.open('events.sqlite', 'r') as db:
    ##print(sqlite.get_filename(db))
    # Get simulated rate from LIGO-LW process table
    (rate,), = db.execute('SELECT comment FROM process WHERE program = ?', ('bayestar-inject',))

     # Get simulated detector network from LIGO-LW process table
    (network,), = db.execute('SELECT ifos FROM process WHERE program = ?', ('bayestar-realize-coincs',))
    
    # Get number of Monte Carlo samples from LIGO-LW process_params table
    (nsamples,), = db.execute('SELECT value FROM process_params WHERE program = ? AND param = ?', ('bayestar-inject', '--nsamples'))
    



## For XML inside the 'events' folder 
# ``tablename=`` keyword argument. The following tables were found: 'coinc_definer', 'process_params', 'process', 'time_slide', 'coinc_event', 'coinc_event_map', 'sngl_inspiral'

#but most of parameters you need should be here, I think
xml_data = gwpy_Table.read('events.xml.gz', format="ligolw", tablename="sim_inspiral")


