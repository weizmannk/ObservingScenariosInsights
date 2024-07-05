# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
---------------------------------------------------------------------------------------------------
@title          : chncky_events.py
@description    : This script splits an events.xml.gz file into smaller chunks to avoid
                  submitting more than 15,000 localization events at once to the cluster,
                  which can be too overwhelming. By splitting the events, it ensures that
                  each chunk is manageable for submission.

@original_author: Leo Psinger
@modified_by    : Ramodgwendé Weizmann KIENDREBEOGO
@contact        : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : June 2024
---------------------------------------------------------------------------------------------------
"""

# python chunky_events.py events.xml.gz chunk-event --chunk-size 5000 --max-events 20000


#!/usr/bin/env python
import os
from argparse import ArgumentParser
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from lalinspiral.thinca import InspiralCoincDef
from ligo.lw.ligolw import Document, LIGO_LW, LIGOLWContentHandler
from ligo.lw.lsctables import (
    CoincDefTable,
    CoincMapTable,
    CoincTable,
    ProcessParamsTable,
    ProcessTable,
    SnglInspiralTable,
    TimeSlideTable,
    New as lsctables_new,
    use_in as lsctables_use_in,
)
from ligo.lw.param import Param, use_in as param_use_in
from ligo.lw.utils import load_filename, write_filename
from tqdm.auto import tqdm
import logging

# Set up logging for the script
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


table_classes = [
    CoincDefTable,
    CoincMapTable,
    CoincTable,
    ProcessParamsTable,
    ProcessTable,
    SnglInspiralTable,
    TimeSlideTable,
]
table_classes_to_copy = [
    CoincDefTable,
    ProcessParamsTable,
    ProcessTable,
    TimeSlideTable,
]


# Define custom content handler for LIGO-LW XML files
@param_use_in
@lsctables_use_in
class ContentHandler(LIGOLWContentHandler):
    pass


# Function to parse command line arguments
def parser():
    parser = ArgumentParser(
        description="Split events.xml.gz file into multiple files with a specified number of events per file."
    )
    parser.add_argument("input", help="Path to the events.xml.gz file")
    parser.add_argument(
        "outdir", type=Path, help="Output directory to save the split files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=15000,
        help="Number of events per chunk (default: 50000)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=15000,
        help="Maximum number of events to process at once (default: 99999)",
    )
    return parser


# Function to split a list into chunks of specified size
def chunksplit(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# Function to process each chunk and create a new XML document
def process_chunk(
    chunk, coinc_def_id, coinc_map_dict, sngl_dict, snr_series_dict, tables
):
    new_xmldoc = Document()
    new_ligolw = new_xmldoc.appendChild(LIGO_LW())
    for table_cls in table_classes_to_copy:
        new_ligolw.appendChild(tables[table_cls])

    new_coinc_table = new_ligolw.appendChild(lsctables_new(CoincTable))
    new_coinc_map_table = new_ligolw.appendChild(lsctables_new(CoincMapTable))
    new_sngl_table = new_ligolw.appendChild(lsctables_new(SnglInspiralTable))

    for coinc in chunk:
        if coinc.coinc_def_id != coinc_def_id:
            continue
        new_coinc_table.append(coinc)
        coinc_map_entries = coinc_map_dict[coinc.coinc_event_id]
        new_coinc_map_table.extend(coinc_map_entries)
        new_sngl_table.extend(sngl_dict[row.event_id] for row in coinc_map_entries)
        for row in coinc_map_entries:
            new_ligolw.appendChild(snr_series_dict[row.event_id])

    return new_xmldoc


# Main function to process the input file and split events into chunks
def main(args=None):
    opts = parser().parse_args(args)

    # Read the input events.xml.gz file
    log.info('Reading "%s"', opts.input)
    xmldoc = load_filename(opts.input, contenthandler=ContentHandler)
    tables = {cls: cls.get_table(xmldoc) for cls in table_classes}

    # Index the events and create mappings for quick access
    log.info("Indexing events")
    coinc_def_id = tables[CoincDefTable].get_coinc_def_id(
        InspiralCoincDef.search, InspiralCoincDef.search_coinc_type
    )
    keyfunc = attrgetter("coinc_event_id")
    coinc_map_dict = {
        key: tuple(items)
        for key, items in groupby(
            sorted(tables[CoincMapTable], key=keyfunc), key=keyfunc
        )
    }
    sngl_dict = {row.event_id: row for row in tables[SnglInspiralTable]}
    snr_series_dict = {
        param.value: param.parentNode
        for param in Param.getParamsByName(xmldoc, "event_id")
    }

    # Prepare the output directory
    log.info('Writing new files to directory "%s"', opts.outdir)
    opts.outdir.mkdir(exist_ok=True)

    # Filter events based on coinc_def_id before splitting into chunks
    filtered_events = [
        event for event in tables[CoincTable] if event.coinc_def_id == coinc_def_id
    ]

    # Split events into chunks if the total number exceeds the max limit, or always create at least one chunk
    event_chunks = list(chunksplit(filtered_events, opts.chunk_size))

    log.info("Total filtered events: %d", len(filtered_events))

    if filtered_events:
        if len(filtered_events) > opts.max_events:
            for i, chunk in enumerate(
                tqdm(event_chunks, desc="Processing chunks", unit="chunk")
            ):

                # display the number of each chunk
                log.info(f"Chunk {i+1}: {len(chunk)} events")
                new_xmldoc = process_chunk(
                    chunk,
                    coinc_def_id,
                    coinc_map_dict,
                    sngl_dict,
                    snr_series_dict,
                    tables,
                )

                # Save the chunk to a new file in the output directory
                new_filename = str(opts.outdir / f"chunk_{i+1}.xml.gz")
                write_filename(new_xmldoc, new_filename)
                log.info('Created chunk file "%s"', new_filename)

        else:
            log.info(
                "Total events (%d) do not exceed the maximum limit (%d)",
                len(filtered_events),
                opts.max_events,
            )
            new_xmldoc = process_chunk(
                filtered_events,
                coinc_def_id,
                coinc_map_dict,
                sngl_dict,
                snr_series_dict,
                tables,
            )
            new_filename = str(opts.outdir / f"chunk_1.xml.gz")
            write_filename(new_xmldoc, new_filename)
            log.info('Created single chunk file "%s"', new_filename)


if __name__ == "__main__":
    main()
