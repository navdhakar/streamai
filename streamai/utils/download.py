import requests
import fire
import subprocess
import json
from tqdm import tqdm
import os
def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    print("downloading train dataset...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    print("download finished")
    return local_filename