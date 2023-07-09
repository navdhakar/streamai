from finetune import train
import requests
import fire
import subprocess
import json
def upload_folder(url, folder_path):
    # Iterate over the files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # Check if the current item is a file
        if os.path.isfile(file_path):
            # Open the file in binary mode
            with open(file_path, 'rb') as file:
                # Create the multipart form-data request
                files = {'file': (file_name, file)}
                response = requests.post(url, files=files)

                # Check the response
                if response.status_code == 200:
                    print(f"Uploaded {file_name} successfully.")
                else:
                    print(f"Failed to upload {file_name}.")

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

def train_model( 
    model_name:str="",
    user_id:str=None,
    dataset_url:str=None
    ):
    if(user_id):
        output_dir_base = model_name if model_name else "./alpaca-lora-finetuned"
        output_dir = f"{output_dir_base}{user_id}"
        download_file(dataset_url, "dataset.json")
        val_set_size = 50

        with open('dataset.json', 'r') as openfile:
            # Reading from json file
            json_object = json.load(openfile)
            val_set_size = int(len(json_object)*0.1)
            
        train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
        # print("uploading finetuned model to storage")
        # res = uplaod(url, output_dir)
        # print(res)
    else:
        print("please provide a valid user token")

if __name__ == "__main__":
    fire.Fire(train_model)
