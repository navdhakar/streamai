from os.path import basename
from streamai.mistral.mistral7b import train as TrainMistral7b
from streamai.mistral.mistral8x7b import train as TrainMistral8x7b
from streamai.utils.upload import upload_folder
from streamai.utils.download import download_file
import requests
import fire
import subprocess
import json
from tqdm import tqdm
import os
import subprocess, sys

available_models = [{"mistralai/Mistral-7B-v0.1":"a 7b parameter version of mistral"}, {"mistralai/Mixtral-8x7B-v0.1":"mistral model based on mixture of experts(moe) having 8 seperate 7b models"}, {"mistralai/Mixtral-8x7B-Instruct-v0.1":"mistral model based on mixture of experts(moe) having 8 seperate 7b models"}]
def Trainmodel(
    model_name:str="",
    base_model:str="mistralai/Mistral-7B-v0.1f",
    dataset_url:str=None,
    scrol_token:str=None,
    num_train_epochs:int=5,
    max_length:int=512,
    resume_checkpoint:str=None,
    batch_size:int=32
    ):
    output_dir_base = model_name if model_name else "./mistral7b-finetuned"
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
            if(base_model == 'mistral8x7b'):
                packages_to_install = ["flash-attn"]
                if packages_to_install:
                    subprocess.run([sys.executable, "-m", "pip", "install"] + packages_to_install)
                print("training mistral moe")
                # train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
                # print("uploading finetuned model to storage")
                # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
            if(base_model == 'mistralai/Mistral-7B-v0.1'):
                print('training mistral 7b model')
                TrainMistral7b(base_model="mistralai/Mistral-7B-v0.1", dataset_path='dataset.json', output_dir=f"{output_dir}", num_train_epochs=num_train_epochs, max_length=max_length, resume_checkpoint=resume_checkpoint, batch_size=batch_size)
                # print("uploading finetuned model to storage")
                # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
            else:
                print(f'this model ${base_model} is not available in library.')
                print(f'please select from the availabel model: ${available_models}')

            # print(res)
    else:
        print("scrol token missing, without it finetuned model will not be uploaded on scrol stoarage and you can't perform auto deploy on scrol.ai(have to manually deploy it for inference) but can deploy using loadmodel on your gpu cloud.")
        user_input = input('continue without scrol_token (y/n): ')
        if(user_input == 'y'):
            with open('dataset.json', 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
                val_set_size = int(len(json_object)*0.1)

                if(base_model == 'mistralai/Mixtral-8x7B-v0.1' or base_model=="mistralai/Mixtral-8x7B-Instruct-v0.1"):
                    packages_to_install = ["flash-attn"]
                    if packages_to_install:
                        subprocess.run([sys.executable, "-m", "pip", "install"] + packages_to_install)

                    print("training mistral moe")
                    TrainMistral8x7b(base_model=base_model, dataset_file='dataset.json', output_dir=f"{output_dir}", num_train_epochs=num_train_epochs, max_length=max_length, resume_checkpoint=resume_checkpoint, batch_size=batch_size)
                    # print("uploading finetuned model to storage")
                    # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
                    print("training completed.")
                    print(f"fine tuning weights are present in dir {output_dir}")
                if(base_model == 'mistralai/Mistral-7B-v0.1' or base_model == 'mistralai/Mistral-7B-Instruct-v0.1'):
                    print('training mistral 7b model')
                    TrainMistral7b(base_model=base_model, dataset_file='dataset.json', output_dir=f"{output_dir}", num_train_epochs=num_train_epochs, max_length=max_length, resume_checkpoint=resume_checkpoint, batch_size=batch_size)
                    print("training completed.")
                    print(f"fine tuning weights are present in dir {output_dir}")
                else:
                    print(f'this model ${base_model} is not available in library.')
                    print(f'please select from the availabel model: ${available_models}')

        elif(user_input == 'n'):
                print('re-run the command with valid scrol_token')
        else:
            print("not a valid choice")

if __name__ == "__main__":
    fire.Fire(Trainmodel)
