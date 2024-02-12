import requests
import fire
import subprocess
import json
from tqdm import tqdm
import os
def upload_folder(url, folder_path, json_data):
    # Create a session to persist the connection
    session = requests.Session()

    # Create a multi-part form-data request for files
    files = []
    print('uploading model data...')
    print("files to be uploaded")

    # Get the total number of files in the folder
    total_files = len([file_name for file_name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file_name))])

    # Iterate over the files in the folder
    for file_name in (os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Append the file to the 'files' list
            print(file_name)
            files.append(('file', (file_name, open(file_path, 'rb'))))

    # Send the multi-part form-data request with the progress bar
    with tqdm(total=total_files, unit='file') as progress_bar:
        response = session.post(url, files=files, data=json_data, stream=True)

        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                # Update the progress bar based on the received data chunks
                progress_bar.update(1)

    # Check the response
    if response.status_code == 200:
        print("Upload successful, now you can deploy the model for ineference.")
    else:
        print("Upload failed.")
