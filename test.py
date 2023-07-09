
from serveai.app import endpointIO
from serveai.models import Autoalpacalora
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
newtraininst = Autoalpacalora("decapoda/llama-7b", "./scroltest")
print(newtraininst.info['available_methods'])
model1 = endpointIO(newtraininst.testIO)
model1.run()
