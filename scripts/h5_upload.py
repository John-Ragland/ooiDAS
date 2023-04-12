import os
import requests
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from tqdm import tqdm

# define variables
url = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'
local_path = 'temp/'
connection_string = os.environ['AZURE_CONSTR_dasdata']
container_name = 'dasdata'

# create blob service client and container client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

start_idx = 58
url_list = os.listdir(url)
url_list = url_list[start_idx:]

# loop through all files in URL
for filename in tqdm(url_list):
    # download file from URL
    file_url = url + filename
    file_content = requests.get(file_url).content

    # save file to local disk
    with open(os.path.join(local_path, filename), 'wb') as f:
        f.write(file_content)

    # upload file to Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
    with open(os.path.join(local_path, filename), 'rb') as data:
        blob_client.upload_blob(data)

    # delete file from local disk
    os.remove(os.path.join(local_path, filename))
 
print('All files downloaded, uploaded, and deleted.')
