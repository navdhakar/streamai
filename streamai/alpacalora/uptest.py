import requests
import os
import json
from tqdm import tqdm

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
        print("Upload successful.")
    else:
        print("Upload failed.")

payload = {
    "secretkey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiclVna3l1aUVVV1lUejhpbGVIU0FyeHR4ZWtDMyIsImlhdCI6MTY4NzQwODE1OX0.NJFe6OafIKb55nS0JeKxEXYNTc442FXLm2uIpdTQPB8",
    "modelname": "alpaca-lora",
    "modelrepo": "somerepo",
    "modelbaseweights": "decapoda/llama",
    "trainingtype": "lora"
}

upload_folder("http://localhost:8080/upload-model", './modeldata', payload)
