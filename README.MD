## 🌩️ Stream AI easily
### streamai let's you serve any AI agents/model(currently llms only) easily as api and integrate in fontend quickly.

#### deploy from custom model script
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def custom_model_IO(input:str):
    output = customodelinference(input) #depend on your inference function, just need to return string output from it.
    return f"this is output of {output}"
    
model1 = endpointIO(custom_model_IO)
model1.run() #this will create a server api endpoint for your model, at http://0.0.0.0:8000 see terminal logs for more info about endpoints
```
#### deploy from inbuild models libs(you can also finetun inbuilt models)
```py
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
    
modelinstance = Autoalpacalora("decapoda/llama-7b", "./scroltest")
#modelinstance.loadmodel() #required for deployment of model as api, not required during finetuning.
#modelinstance.setparameters(input="use this as context", max_tokens=128, top_p=12, top_k=40) #optional, look into .info['available_methods']['setparameters'] for more details.
print(modelinstance.info['available_methods'])
model1 = endpointIO(modelinstance.testinferenceIO) #still some testing to do in actual alpaca inferneceIO.
model1.run()
```
todo:
- [ ] test alpaca lora with deployment
- [ ] fix dynamic installation issue
- [ ] add endpoint for info about deployed model


