
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


for eventsfile in runs/*/*/events.sqlite
    do ligo-skymap-stats -d $eventsfile -o $(dirname $eventsfile)/allsky_tester.dat \
        $(find $(dirname $eventsfile)/allsky -name '*.fits' | sort -V) --cosmology --contour 20 50 90 -j
done



for eventsfile in runs/*/*/events.sqlite
do
    fits_list=$(dirname $eventsfile)/allsky_fits_list.txt
    find $(dirname $eventsfile)/allsky -name '*.fits' | sort -V > $fits_list
    
    ligo-skymap-stats -d $eventsfile -o $(dirname $eventsfile)/allsky_tester.dat \
        $(cat $fits_list) --cosmology --contour 20 50 90 -j
    
    rm $fits_list
done






batch_size=50000
chunk_number=1
fits_files=($(find "$output_dir/allsky" -name '*.fits' | sort -V))

total_files=${#fits_files[@]}
for (( i=0; i<total_files; i+=batch_size )); do
    last_index=$((i + batch_size - 1))
    if [ $last_index -ge $total_files ]; then
        last_index=$((total_files - 1))
    fi
    current_batch=("${fits_files[@]:i:last_index-i+1}")
    output_file="${output_dir}/allsky_chunk${chunk_number}.dat"
    echo "Processing chunk $chunk_number for $eventsfile with $((${last_index-i+1})) files"

    # Command to process the current batch
    ligo-skymap-stats -d "$eventsfile" -o "$output_file" \
        ${current_batch[@]} --cosmology --contour 20 50 90 -j

    ((chunk_number++))
done

