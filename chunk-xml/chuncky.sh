#!/bin/bash
# Loop over all events.xml.gz files in the specified directories
for eventsfile in runs/*/*/events.xml.gz; do
    # Extract the directory of the events file to use as the output directory
    outdir=$(dirname "$eventsfile")/chunk-events
    mkdir -p "$outdir"  # Create the output directory if it doesn't exist

    # Run the Python script with the specified chunk size and max events
    python chuncky_events.py "$eventsfile" "$outdir" --chunk-size 15000 --max-events 15000
done
