import subprocess, sys
import pkg_resources
class AutoMistral:
    def __init__(self, base_model):
        require_install = ['trl', 'accelerate', 'appdirs', 'loralib', 'bitsandbytes', 'black', 'black[jupyter]', 'datasets', 'fire', 'git+https://github.com/huggingface/peft.git', 'transformers>=2.28.0', 'sentencepiece', 'gradio', 'scipy', 'tqdm', 'torch==2.2.0']
    
        installed_packages = [pkg.key for pkg in pkg_resources.working_set]
        packages_to_install = [package for package in require_install if package not in installed_packages]
    
        if packages_to_install:
            subprocess.run([sys.executable, "-m", "pip", "install"] + packages_to_install)
    
        from streamai.mistral import Loadmodel, Evalmodel, AutoTrainMistral
        self.info = {
            "modelname":"mistral",
            "initialization_args":{
                "base_model*":"path to base model (eg. mistralai/Mistral-7B-v0.1)",
            },
            "dataset":{
                "format":".json",
                "structure":[
                    {"instruction": "Give three tips for staying healthy.", "input": "", "output": "1.Eat a balanced diet . \n2. Exercise regularly.    \n3. Get enough sleep."},
                    {"instruction": "What are the three primary colors?", "input": "", "output": "The three primary colors are red, blue, and yellow."
                    }
                ],
            "demo_dataset":"https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b"
            },
            "available_methods":{
                "loadmodels":{
                    "args":["lora_wieghts"],
                    "desc":"call this to load model immediately after object creation"
                },
                "train":{
                    "arg":["dataset_url"],
                    "desc":"training model directly with correct formatted dataset(dataset instruction WIP)"
                },
                "inferenceIO":{
                    "arg":["instruction/prompt*"]
                },
                "setparameters":{
                    "arg":["input", "temperature", "top_p", "top_k", "num_beams", "max_new_tokens", "stream_output(WIP)"]
                }
            }
        }
        self.load_8bit = False
        self.base_model = base_model
        self.prompt_template= ""  
        self.server_name = "0.0.0.0"
        self.model = None
        self.input = None
        self.temperature=0.1
        self.top_p=0.75
        self.top_k=40
        self.num_beams=4
        self.max_new_tokens=128
        self.stream_output=False

    def loadmodel(self, model_name:str=""):
        from streamai.mistral import Loadmodel
        self.model = Loadmodel(load_8bit = self.load_8bit, base_model = self.base_model, lora_weights = model_name)
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
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k
        self.num_beams=num_beams
        self.max_new_tokens=max_new_tokens
        self.stream_output=stream_output
    def train(self, base_model:str="mistralai/Mistral-7B-v0.1", dataset_url:str=None, model_name:str="mistralfinetuned"):
        from streamai.mistral import AutoTrainMistral

        #WIP
        #TODO: 
        #peft(lora) training, #raw training, training metrics, 
        #correct data path provide, 
        #saving trained output weights correcly so autoloader can load finetuned model easily,
        #chek if required can gpu specs support training
        if dataset_url:
            AutoTrainMistral(base_model=base_model, dataset_url=dataset_url, model_name=model_name)        
        else:
            return f"please provide url for your dataset." 
    def inferenceIO(self, prompt):
        from streamai.mistral import Evalmodel
        if self.model:
            self.generation = ""
            for output in Evalmodel(
                                instruction=prompt,
                                model=self.model,
                                base_model=self.base_model,
                                input=self.input,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                top_k=self.top_k,
                                num_beams=self.num_beams,
                                max_new_tokens=self.max_new_tokens,
                                stream_output=self.stream_output,
            ):
                self.generation = self.generation + output
            # this out put should always be a string.                
            return self.generation
        else:
            return "Please load the model first(modelinstance.loadmodel())."
