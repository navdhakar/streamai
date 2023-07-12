from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
import json
modelinstance = Autoalpacalora(base_model="decapoda-research/llama-7b-hf")

# let's see what is the structure of dataset that is required for training
print(json.dumps(modelinstance.info['dataset'], indent = 1))

#let's provide the dataset url
modelinstance.train(dataset_url="https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b") # here it is using dummy dataset available on this link.

#then load fine tuned model using lora weights
modelinstance.loadmodel(lora_weights="alpacafinetuned") 
#provide lora weights if model is finetuned, it is directory that will be logged after traininig is done.

# now run the model as api endpoint.
finetunedmodel = endpointIO(modelinstance.inferenceIO)
finetunedmodel.run()
