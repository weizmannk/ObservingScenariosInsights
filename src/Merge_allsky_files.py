# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : July 2024
@description    : This script reads multiple data files, merges them into a single table, and saves 
                  the merged table in a specified format.
---------------------------------------------------------------------------------------------------
"""

from astropy.table import Table, vstack
from astropy.io.registry import IORegistryError
from pathlib import Path



# Define the paths to the input files and output directory
input_file_1 = Path("./runs/O4/farah/allsky_chunk1.dat")
input_file_2 = Path("./runs/O4/farah/allsky_chunk2.dat")
input_file_3 = Path("./runs/O4/farah/allsky_chunk3.dat")

output_file = Path("./runs/O4/farah/allsky_merged.dat")

# Read the input files
try:
    allsky_1 = Table.read(input_file_1, format="ascii.fast_tab")
    allsky_2 = Table.read(input_file_2, format="ascii.fast_tab")
    allsky_3 = Table.read(input_file_3, format="ascii.fast_tab")
except IORegistryError as e:
    print(f"Error reading input files: {e}")
    raise



# Merge the tables
try:
    allsky_merged = vstack([allsky_1, allsky_2, allsky_2], join_type='exact')
except TableMergeError as ex:
    print(f"Error merging tables: {ex}")
    raise

# Save the merged table
try:
    allsky_merged.write(output_file, format="ascii.tab", overwrite=True)
    print(f"Successfully saved merged table to {output_file}")
except IORegistryError as e:
    print(f"Error writing merged table: {e}")
    raise
