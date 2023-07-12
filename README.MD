## 🌩️ Stream AI easily
### streamai let's you serve any AI agents/model(currently llms only) easily as api and integrate in frontend quickly.
- sometime need to replace cudua lib in bitsandbytes with cuda lib 117 version(bnb bug).
```bash
cp venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda117.so venv/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so
```

#### deploy from custom model script
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def custom_model_IO(input:str):
    output = customodelinference(input) #depend on your inference function, just need to return string output from it.
    return f"this is output of {output}"
    
custom_model = endpointIO(custom_model_IO)
custom_model.run() #this will create a server api endpoint for your model, at http://0.0.0.0:8000 see terminal logs for more info about endpoints
```
#### deploy from inbuild models libs(you can also finetune inbuilt models)
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
#### finetune the model and then deploy it.
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora

modelinstance = Autoalpacalora(base_model="decapoda-research/llama-7b-hf")
modelinstance.train(dataset_url="https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b") # when finetuning
#then load fine tuned model using lora weights
modelinstance.loadmodel(lora_weights="alpacafinetuned") #provide lora weights if model is finetuned, it is directory that will be logged after training is finished and you can paste it here as lora weights.
finetunedmodel = endpointIO(modelinstance.inferenceIO)
finetunedmodel.run()
```
todo:
- [x] test alpaca lora with deployment
- [x] fix dynamic installation issue(temp fix)
- [x] fix(api endpoint start but does not work on cloud probably firewall issue) 
- [x] test finetuning of alpacalora.
- [ ] add nginx reverse proxy.
- [ ] add tests cases.
- [ ] add endpoint for info about deployed model


