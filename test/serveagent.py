from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def testiofunc(inpt:str):
    output = modelinstance.inferenceIO(prompt=input)
modelinstance = Autoalpacalora("decapoda-research/llama-7b-hf")
modelinstance.loadmodel() #required for deployment of model as api, not required during finetuning.
model1 = endpointIO(modelinstance.inferenceIO)
model1.run()
