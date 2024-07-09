## For the progress bar, install : sudo apt-get install pv
 

#!/bin/bash

# Script to split or combine a ZIP file
# Usage:
# To split:  ./split_combine.sh split
# To combine: ./split_combine.sh combine


ACTION=$1
ZIP_FILE="runs.zip"
PART_PREFIX="runs_part_"
PART_SIZE="20G"

if [ "$ACTION" == "split" ]; then
    echo "Splitting $ZIP_FILE into parts of $PART_SIZE..."
    
    # Get the size of the ZIP file
    ZIP_FILE_SIZE=$(stat -c%s "$ZIP_FILE")
    
    # Use pv to show progress while splitting the file
    pv -s $ZIP_FILE_SIZE "$ZIP_FILE" | split -b $PART_SIZE - "$PART_PREFIX"
    
    echo "Splitting completed."
elif [ "$ACTION" == "combine" ]; then
    echo "Combining parts into $ZIP_FILE..."
    
    # Calculate the total size of the parts
    TOTAL_SIZE=$(du -cb ${PART_PREFIX}* | grep total | awk '{print $1}')
    
    # Use pv to show progress while combining the parts
    pv -s $TOTAL_SIZE ${PART_PREFIX}* > $ZIP_FILE
    
    echo "Combining completed."
    #echo "Unzipping $ZIP_FILE..."
    #unzip $ZIP_FILE
    #echo "Unzipping completed."
else
    echo "Usage: $0 [split|combine]"
fi

