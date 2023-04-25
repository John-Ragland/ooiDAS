import os
import requests
import shutil
import subprocess
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

url = "http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/"
connect_str = os.environ["AZURE_CONSTR_dasdata"]
container_name = "hdf5"

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client(container_name)

url_contents = requests.get(url).text
file_list = [line.split('">')[-1] for line in url_contents.split('\n')[1:-2]]

for filename in tqdm(file_list):
    local_file = os.path.join(os.getcwd(), filename)
    if not os.path.exists(local_file):
        file_url = os.path.join(url, filename)
        with requests.get(file_url, stream=True) as r:
            with open(local_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    blob_client = container_client.get_blob_client(filename)
    if not blob_client.exists():
        blob_client.upload_blob_from_path(local_file)
    os.remove(local_file)
