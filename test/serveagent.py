from streamai.app import endpointIO
from streamai.llms import Autoalpacalora
def testiofunc(inpt:str):
    output = modelinstance.inferenceIO(prompt=input)
modelinstance = Autoalpacalora("decapoda-research/llama-7b-hf")
modelinstance.loadmodel() #required for deployment of model as api, not required during finetuning.
#if api endpoint is not accepting request, please check your server or vpc firewall.
#it seems to work with anychange in some clusters but need ufw firewall modification in some.(last tested on lamda lab rtx 6000)
model1 = endpointIO(modelinstance.inferenceIO)
model1.run()
