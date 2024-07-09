# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
---------------------------------------------------------------------------------------------------
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ObservingScenariosInsights.git
@createdOn      : July 2024
@description    : A tool for interacting with the Zenodo API, facilitating the download of files based on DOIs.
                  This class provides functionality to retrieve the latest version DOI associated with a provided
                  permanent DOI, and subsequently download the corresponding files from Zenodo in chunks.
                  After downloading, use the "split_combine.sh"  script to combine the chunks into a single file.
                  Install pv for progress bar: sudo apt-get install pv
                  
                  Make sure you have the all 6 files together  before  run the combination process :
                   ./split_combine.sh combine
---------------------------------------------------------------------------------------------------
"""

from tqdm.auto import tqdm
import re
import requests
from pathlib import Path

# If the link to the data is required manually, it can be accessed at :
# "runs_part_aa" and  "runs_part_ab"  : https://zenodo.org/doi/10.5281/zenodo.12693652
# "runs_part_ac" and  "runs_part_ad"  :  https://zenodo.org/doi/10.5281/zenodo.12694779
# "runs_part_ae" and  "runs_part_af" : https://zenodo.org/doi/10.5281/zenodo.12696695

# List of DOIs and corresponding file names
doi_file_map = {
    "12693652": ["runs_part_aa", "runs_part_ab"],
    "12694779": ["runs_part_ac", "runs_part_ad"],
    "12696695": ["runs_part_ae", "runs_part_af"]
}


# Create output directory to save the downloaded files
save_path =  Path("./data")
save_path.mkdir(parents=True, exist_ok=True)



class ZenodoDownloader:

    """A class to interact with the Zenodo API and download files based on DOI.

    :param permanent_doi: Permanent DOI for a Zenodo record
    :type permanent_doi: str

    :param file_name: Name of the file to be downloaded
    :type file_name: str

    :param save_path: Directory path to save the downloaded file
    :type save_path: str
    """

    def __init__(self, permanent_doi, file_name, save_path):
        self.permanent_doi = permanent_doi
        self.file_name = file_name
        self.save_path = save_path
        self.headers = {
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Host": "zenodo.org",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        self.latest_doi = self.get_latest_zenodo_doi()

    def get_latest_zenodo_doi(self):

        """Retrieves the latest version DOI associated with the provided permanent DOI.

        :return: Latest version DOI
        :rtype: str
        """
        r = requests.get(
            f"https://zenodo.org/record/{self.permanent_doi}",
            allow_redirects=True,
            headers=self.headers,
        )
        try:
            data = r.text.split("10.5281/zenodo.")[1]
            doi = re.findall(r"^\d+", data)[0]
        except Exception as e:
            raise ValueError(f"Could not find latest DOI: {str(e)}")
        return doi

    def download_zenodo_data(self):

        """Downloads the file from Zenodo based on the provided DOI and file name.

        :return: None
        """

        try:
            # Fetching the Zenodo record metadata using the DOI
            r = requests.get(
                f"https://zenodo.org/api/records/{self.latest_doi}",
                headers=self.headers,
            )
            r.raise_for_status()

            # Extracting file URL(s) from the Zenodo record metadata
            record_data = r.json()
            files = record_data["files"]

            # Select file to download; default to first file if specific name not found
            file_to_download = next(
                (f for f in files if f["key"] == self.file_name), files[0]
            )

            file_url = file_to_download["links"]["self"]
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Raise HTTP errors if any

            file_size = int(response.headers.get("content-length", 0))
            
            # Combine the save path and file name
            full_path = self.save_path / self.file_name

            with open(full_path, "wb") as file:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        progress.update(len(chunk))

            print(f"File downloaded successfully to '{self.save_path}'.")
        except requests.RequestException as e:
            print(f"Error: {e}")



for permanent_doi, file_names in doi_file_map.items():
    for file_name in file_names:
        zenodo_downloader = ZenodoDownloader(permanent_doi, file_name, save_path)
        zenodo_downloader.download_zenodo_data()