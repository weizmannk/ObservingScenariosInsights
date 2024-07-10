# Observing Scenarios Insights

Observing Scenarios Insights is a comprehensive project designed to facilitate the download, processing, and analysis of astronomical data. The project interacts with the Zenodo API to download data files based on DOIs, splits large files into chunks, and provides tools to merge these chunks for further analysis.




Scripts

`BNS_NSBH_zenodo_download.py`
A tool for interacting with the Zenodo API, facilitating the download of files based on DOI. This script retrieves the latest version DOI associated with a provided permanent DOI and downloads the corresponding file from Zenodo.


`download_split_chunk_data.py` 
Similar to BNS_NSBH_zenodo_download.py, this script downloads data from Zenodo based on DOIs but is specifically designed to handle files in chunks. It ensures that all parts are downloaded before combining them using the split_combine.sh script.

`split_combine.sh`

A shell script to combine downloaded file chunks into a single file. Ensure all parts are present before running the combination process.