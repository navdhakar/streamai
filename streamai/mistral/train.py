from streamai.mistral.mistral8x7b import train
import requests
import fire
import subprocess
import json
from tqdm import tqdm
import os

available_models = [{"mistral7b":"a 7b parameter version of mistral"}, {"mistral8x7b":"mistral model based on mixture of experts(moe) having 8 seperate 7b models"}]
def Trainmodel( 
    model_name:str="",
    base_model:str="decapoda-research/llama-7b-hf",
    dataset_url:str=None,
    scrol_token:str=None
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
            if(base_model == 'mistral8x7b'){
                print("training mistral moe")
                # train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
                # print("uploading finetuned model to storage")
                # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
            }
            if(base_model == 'mistral7b'){
                print('training mistral 7b model')
                # train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
                # print("uploading finetuned model to storage")
                # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
            }
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

                if(base_model == 'mistral8x7b'){
                    print("training mistral moe")
                    # train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
                    # print("uploading finetuned model to storage")
                    # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
                    print("training completed.")
                    print(f"fine tuning weights are present in dir {output_dir}")
                }
                if(base_model == 'mistral7b'){
                    print('training mistral 7b model')
                    # train(base_model="decapoda-research/llama-7b-hf", data_path='dataset.json', output_dir=f"{output_dir}", val_set_size=val_set_size)
                    # print("uploading finetuned model to storage")
                    # upload_folder("https://scrol-internal-testing.onrender.com/upload-model", output_dir, payload)
                    print("training completed.")
                    print(f"fine tuning weights are present in dir {output_dir}")
                }
                else:
                    print(f'this model ${base_model} is not available in library.')
                    print(f'please select from the availabel model: ${available_models}')
                    
        elif(user_input == 'n'):
                print('re-run the command with valid scrol_token')
        else:
            print("not a valid choice")

if __name__ == "__main__":
    fire.Fire(Trainmodel)
