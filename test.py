
from serveai.app import endpointIO
from serveai.models import Autoalpacalora
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
newtraininstance = Autoalpacalora("decapoda/llama-7b", "./scroltest")
print(newtraininstance.info['available_methods'])
#model1 = endpointIO("some sort", testiofunc)
#model1.run()
