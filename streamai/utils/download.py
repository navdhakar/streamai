import requests
import fire
import subprocess
import json
from tqdm import tqdm
import os
def download_file(url, local_filename):
    try:
        # Check if the URL is a local path or filename
        if os.path.exists(url):  # If it's a local path or filename
            print(f"Using local file: {url}")
            with open(url, 'rb') as infile:
                content = infile.read()
                with open(local_filename, 'wb') as outfile:
                    outfile.write(content)
            print(f"Content from {url} written to {local_filename}")
        else:  # If it's a URL
            print("Downloading file from URL...")
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print("Download finished")
    except Exception as e:
        print("Error:", e)
        print("URL is not correct. If it is a local file, please check the path provided.")
        return None
    return local_filename
