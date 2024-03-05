from streamai.alpacalora.finetune import train
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

def Trainmodel(
    model_name:str="",
    base_model:str="decapoda-research/llama-7b-hf",
    dataset_url:str=None,
    scrol_token:str=None,
    num_train_epochs:int=None,
    max_length:int=None
    ):
    output_dir_base = model_name if model_name else "./alpaca-lora-finetuned"
    output_dir = f"{output_dir_base}"
    download_file(dataset_url, "dataset.json")
    val_set_size = 50
    user_input = None
    payload = {
    "secretkey": scrol_token,
    "modelname": output_dir,
    "modelrepo": 'https://github.com/corporaai/alpaca-lora.git',
    "modelbaseweights": "decapoda/llama",
    "trainingtype": "lora"
        }
    if scrol_token:
        with open('dataset.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            val_set_size = int(len(json_object)*0.1)

            train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size, num_train_epochs=num_train_epochs, max_length=max_length)
            print("uploading finetuned model to storage")
            upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
            # print(res)
    else:
        print("scrol token missing, without it finetuned model will not be uploaded on scrol stoarage and you can't perform auto deploy on scrol.ai(have to manually deploy it for inference) but can deploy using loadmodel on your gpu cloud.")
        user_input = input('continue without scrol_token (y/n): ')
        if(user_input == 'y'):
            with open('dataset.json', 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
                val_set_size = int(len(json_object)*0.1)

                train(base_model=base_model, data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size, num_train_epochs=num_train_epochs, max_length=max_length)
                print("training completed.")
                print(f"fine tuning weights are present in dir {output_dir}")
        elif(user_input == 'n'):
                print('re-run the command with valid scrol_token')
        else:
            print("not a valid choice")

if __name__ == "__main__":
    fire.Fire(Trainmodel)
