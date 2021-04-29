import os
from google.cloud import storage

def explicit():
    # Explicitly use service account credentials by specifying the private key
    # file.
    storage_client = storage.Client.from_service_account_json(
        '/home/my-key.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)
    
def download_blob(date):
    """Downloads a blob from the bucket."""
    bucket_name = "where-collect.appspot.com"

    storage_client = storage.Client.from_service_account_json(
        '/home/my-key.json')
    
    prefix = f"data/{date}/"
    download_dir = f"data/realtime/{date}/"
    if not os.path.exists(download_dir):
            os.makedirs(download_dir)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    for blob in blobs:
        file_list = blob.name.split('/')
        blob.download_to_filename(download_dir + file_list[-1])
        print(f"Blob {blob.name} downloaded.")
