import subprocess, sys
class Autoalpacalora:
    def __init__(self, base_model, lora_weights):
        self.info = {
            "modelname":"alpacalora",
            "initialization_args":{
                "base_model":"path to base model (eg. decapoda/llama-7b)",
                "lora_weights":"path to trained lora weights dir."
            },
            "available_methods":{
                "loadmodels":{
                    "args":None,
                    "desc":"call this to load model immediately after object creation"
                },
                "train":{
                    "arg":["datasetpath"],
                    "desc":"training model directly with correct formatted dataset(dataset instruction WIP)"
                },
                "inference":{
                    "arg":["instruction*", "loadedmodel", "input", "temperature", "top_p", "top_k", "num_beams", "max_new_tokens", "stream_output(WIP)"]
                }
            }
        }
        self.load_8bit = False,
        self.base_model = base_model,
        self.lora_weights = lora_weights,
        self.prompt_template= "",  
        self.server_name = "0.0.0.0",  


       # require_install = ['accelerate', 'appdirs', 'loralib', 'bitsandbytes', 'black', 'black[jupyter]', 'datasets', 'fire', 'git+https://github.com/huggingface/peft.git', 'transformers>=4.28.0', 'sentencepiece', 'gradio', 'scipy', 'tqdm']
       # for package in require_install:
       #     subprocess.run([sys.executable, "-m", "pip", "install", package])
       #from serveai.models.alpacalora import Loadmodel, Evalmodel, AutoTrainalpacalora

    def loadmodel(self):
        self.model = Loadmodel(load_8bit = self.load_8bit, base_model = self.base_model, lora_weights = self.lora_weights)
        out
    def inference(
        self,
        instruction,
        model,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
    ):
        if model:
            self.output = Evalmodel(
                                instruction=instruction,
                                model=model,
                                input=None,
                                temperature=0.1,
                                top_p=0.75,
                                top_k=40,
                                num_beams=4,
                                max_new_tokens=128,
                                stream_output=False,
                            )    
            return output
        else:
            return "Please load the model first(modelinstance.loadmodel())."
    def train(self, base_model:str, data_path:str, output_dir:str):
        AutoTrainalpacalora(base_model=base_model, data_path=data_path, output_dir=output_dir)        
        return f"training model" 
