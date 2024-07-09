import requests
import json
from pathlib import Path
from tqdm import tqdm
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor

# Put your own token, from your Zenodo profile
ACCESS_TOKEN = 'zenodo-token'
# Paths to the files you want to upload
FILE_PATHS = ['./runs_part_aa', './runs_part_ab', './runs_part_ac']

# Replace with your deposition metadata
METADATA = {
    'metadata': {
        'title': 'Observing scenarios simulations for HLVK-Configuration for O4 Runs, using 20 million injections. This simulation led to 17,009 BNS useful for training PE of EM counterparts of GW. (July 2024 edition).',
        'upload_type': 'dataset',
        'description': (
            'We have conducted a simulation of the HLVK-configuration deployed during the ongoing O4 run. '
            'This project supports the training of kilonova regression with machine learning processes, requiring thousands of BNS to pass the threshold cutoff. '
            'Here we have 17,009 BNS passing the SNR threshold, along with 3,148 NSBH and 121,718 BBH, from 20 million CBCs injected. '
            'Upper-lower limit between NS and BH is 3 solar masses.\n\n'
            'Because of large file sizes, we have split them into three parts. '
            'After downloading them, you will need to combine them using the following process:\n\n'
            '1. Combine the parts:\n'
            '```sh\n'
            'cat runs_part_* > runs.zip\n'
            '```\n'
            '2. Verify the combined file:\n'
            '```sh\n'
            'ls -lh runs.zip\n'
            'file runs.zip\n'
            '```\n'
            '3. Unzip the combined file:\n'
            '```sh\n'
            'unzip runs.zip\n'
            '```\n'
        ),
        'creators': [
            {
                'name': 'R. Weizmann Kiendrebeogo',
                'affiliation': (
                    '1. Laboratoire de Physique et de Chimie de l\'Environnement, Université Joseph KI-ZERBO, Ouagadougou, Burkina Faso\n'
                    '2. Artemis, Observatoire de la Côte d\'Azur, Université Côte d\'Azur, Boulevard de l\'Observatoire, 06304 Nice, France'
                ),
                'orcid': 'https://orcid.org/0000-0002-9108-5059'
            },
            {
                'name': 'Michael W. Coughlin',
                'affiliation': 'School of Physics and Astronomy, University of Minnesota, Minneapolis, Minnesota 55455, USA',
                'orcid': 'https://orcid.org/0000-0002-8262-2924'
            },
            {
                'name': 'Natalya Pletskova',
            },
            {
                'name': 'Hannah Hoggard',
            }
        ],
        'references': [
            {
                'identifier': '10.3847/1538-4357/acfcb1',
                'relation': 'isReferencedBy',
                'scheme': 'doi'
            }
        ]
    }
}

# Zenodo API URL
ZENODO_URL = 'https://zenodo.org/api/deposit/depositions'

def create_deposition():
    headers = {"Content-Type": "application/json"}
    params = {'access_token': ACCESS_TOKEN}
    response = requests.post(ZENODO_URL, params=params, json={}, headers=headers)
    response.raise_for_status()
    return response.json()

def upload_file(deposition_id, file_path):
    file_size = Path(file_path).stat().st_size
    data = {'name': Path(file_path).name}

    with open(file_path, 'rb') as file:
        encoder = MultipartEncoder(fields={'file': (Path(file_path).name, file, 'application/octet-stream')})
        progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=Path(file_path).name)
        monitor = MultipartEncoderMonitor(encoder, lambda monitor: progress_bar.update(monitor.bytes_read - progress_bar.n))
        
        headers = {
            'Content-Type': monitor.content_type
        }
        params = {'access_token': ACCESS_TOKEN}
        response = requests.post(
            f"{ZENODO_URL}/{deposition_id}/files",
            params=params,
            data=monitor,
            headers=headers
        )
        progress_bar.close()

    response.raise_for_status()
    return response.json()

def update_metadata(deposition_id, metadata):
    params = {'access_token': ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    deposition_url = f"{ZENODO_URL}/{deposition_id}"
    response = requests.put(deposition_url, params=params, data=json.dumps(metadata), headers=headers)
    response.raise_for_status()
    return response.json()

def publish_deposition(deposition_id):
    params = {'access_token': ACCESS_TOKEN}
    deposition_url = f"{ZENODO_URL}/{deposition_id}/actions/publish"
    response = requests.post(deposition_url, params=params)
    response.raise_for_status()
    return response.json()

def main():
    # Step 1: Create a new deposition
    deposition = create_deposition()
    deposition_id = deposition['id']
    print(f"Created deposition: {deposition_id}")

    # Step 2: Upload files
    for file_path in FILE_PATHS:
        upload_response = upload_file(deposition_id, file_path)
        print(f"Uploaded file {file_path}: {upload_response}")

    # Step 3: Update metadata
    metadata_response = update_metadata(deposition_id, METADATA)
    print(f"Updated metadata: {metadata_response}")

    # Step 4: Publish deposition
    publish_response = publish_deposition(deposition_id)
    print(f"Published deposition: {publish_response}")

if __name__ == '__main__':
    main()


