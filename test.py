
from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
modelinstance = Autoalpacalora("decapoda-research/llama-7b-hf")
#modelinstance.loadmodel() #required for deployment of model as api, not required during finetuning.
#modelinstance.setparameters(input="use this as context", max_tokens=128, top_p=12, top_k=40) #optional, look into .info['available_methods']['setparameters'] for more details.
#print(modelinstance.info['available_methods'])
output = modelinstance.inferenceIO(prompt="what is llama?"):
print(output)
#model1 = endpointIO(modelinstance.testinferenceIO)
#model1.run()
