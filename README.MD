## 🌩️ Stream AI easily
### train, finetune any open source model and deploy as api in less than 10 lines of python.
- sometime need to replace cudua lib in bitsandbytes with cuda lib 117 version(bnb bug).
```bash
cp venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
```

#### run base model without creating api.
```py
from streamai.app import endpointIO
from streamai.llms import AutoMistral
import json
modelinstance = AutoMistral(base_model="mistralai/Mistral-7B-v0.1")
modelinstance.loadmodel()
response = modelinstance.inferenceIO(prompt="what is python?")
print(response)
```

#### deploy your custom model with api.
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def custom_model_IO(input:str):
    output = customodelinference(input) #depend on your inference function, just need to return string output from it.
    return f"this is output of {output}"

custom_model = endpointIO(custom_model_IO)
custom_model.run() #this will create a server api endpoint for your model, at http://0.0.0.0:8000 see terminal logs for more info about endpoints
```
#### deploy from available base model as api.
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def testiofunc(inpt:str):
    output = modelinstance.inferenceIO(prompt=input)
modelinstance = Autoalpacalora("decapoda-research/llama-7b-hf")
modelinstance.loadmodel() #required for deployment of model as api, not required during finetuning.
alpaca_model = endpointIO(modelinstance.inferenceIO)
alpaca_model.run()
```
#### finetune any available model available in llms and then deploy it.
```py
from streamai.app import endpointIO
from streamai.llms import AutoMistral
import json
modelinstance = AutoMistral(base_model="mistralai/Mistral-7B-v0.1")

# let's see what is the structure of dataset that is required for training
print(json.dumps(modelinstance.info['dataset'], indent = 1))

#let's provide the dataset url
modelinstance.train(dataset_url="https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json", model_name="mistral7btest")
```
#### serve your finetuned model as api.
```py
from streamai.app import endpointIO
from streamai.llms import AutoMistral
import json
modelinstance = AutoMistral(base_model="mistralai/Mistral-7B-v0.1")

#load fine tuned model using directory in which finetuned model is stored.
modelinstance.loadmodel(finetuned_weights_dir="mistral7btest")
#provide lora weights if model is finetuned, it is directory that will be logged after traininig is done.

# now run the model as api endpoint.
finetunedmodel = endpointIO(modelinstance.inferenceIO)
finetunedmodel.run()
```
todo:
- [ ] add input for num_train_epochs for finetuning.
- [ ] add input train test split.
- [ ] add both generation and training test cases for mistral.
- [ ] add nginx reverse proxy.
- [ ] add tests cases.
- [ ] add endpoint for info about deployed model
