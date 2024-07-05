
#!/bin/bash

# Iterate over each events file
for eventsfile in runs/*/*/events.sqlite
do
    # Create a temporary file to store intermediate results
    temp_allsky=$(mktemp)

    # Find .fits files and process them in batches using xargs
    find $(dirname $eventsfile)/allsky -name '*.fits' | sort -V | xargs -n 141880 | while read -r batch
    do
        ligo-skymap-stats -d "$eventsfile" -o "${temp_allsky}_part" $batch --cosmology --contour 20 50 90 -j
        cat "${temp_allsky}_part" >> "$temp_allsky"
        rm "${temp_allsky}_part"
    done

    # Move the temporary file to the final destination
    mv "$temp_allsky" "$(dirname $eventsfile)/allsky.dat"
done
