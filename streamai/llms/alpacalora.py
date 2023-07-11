import subprocess, sys
class Autoalpacalora:
    def __init__(self, base_model, lora_weights=""):
        self.info = {
            "modelname":"alpacalora",
            "initialization_args":{
                "base_model*":"path to base model (eg. decapoda/llama-7b)",
                "lora_weights(optional)":"path to trained lora weights dir."
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
                "inferenceIO":{
                    "arg":["instruction/prompt*"]
                },
                "setparameters":{
                    "arg":["input*", "temperature", "top_p", "top_k", "num_beams", "max_new_tokens", "stream_output(WIP)"]
                }
            }
        }
        self.load_8bit = False,
        self.base_model = base_model,
        self.lora_weights = lora_weights,
        self.prompt_template= "",  
        self.server_name = "0.0.0.0",
        self.model = None
        self.input = None
        self.temperature=0.1,
        self.top_p=0.75,
        self.top_k=40,
        self.num_beams=4,
        self.max_new_tokens=128,
        self.stream_output=False,


        #require_install = ['accelerate', 'appdirs', 'loralib', 'bitsandbytes', 'black', 'black[jupyter]', 'datasets', 'fire', 'git+https://github.com/huggingface/peft.git', 'transformers>=2.28.0', 'sentencepiece', 'gradio', 'scipy', 'tqdm']
        #for package in require_install:
        #    subprocess.run([sys.executable, "-m", "pip", "install", package])
        #from streamai.alpacalora import Loadmodel, Evalmodel, AutoTrainalpacalora

    def loadmodel(self):
        self.model = Loadmodel(load_8bit = self.load_8bit, base_model = self.base_model, lora_weights = self.lora_weights)
    def setparameters(
        self,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
    ):
        self.input = input
        self.temperature=temperature,
        self.top_p=top_p,
        self.top_k=top_k,
        self.num_beams=num_beams,
        self.max_new_tokens=max_new_tokens,
        self.stream_output=stream_output,
    def train(self, base_model:str, data_path:str, output_dir:str):
        #WIP
        #TODO: 
        #peft(lora) training, #raw training, training metrics, 
        #correct data path provide, 
        #saving trained output weights correcly so autoloader can load finetuned model easily,
        #chek if required can gpu specs support training
        AutoTrainalpacalora(base_model=base_model, data_path=data_path, output_dir=output_dir)        
        return f"training model" 
    def inferenceIO(self, prompt):
        if self.model:
            self.output = Evalmodel(
                                instruction=prompt,
                                model=self.model,
                                input=self.input,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                top_k=self.top_k,
                                num_beams=self.num_beams,
                                max_new_tokens=self.max_new_tokens,
                                stream_output=self.stream_output,
                            )    
            # this out put should always be a string.                
            return output
        else:
            return "Please load the model first(modelinstance.loadmodel())."
    def testinferenceIO(self, prompt):
        return f"your model output is {prompt}"
