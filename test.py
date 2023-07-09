
from serveai.app import endpointIO
from serveai.models import ATalpacalora
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
newtraininstance = ATalpacalora("string")
print(newtraininstance.train())
#model1 = endpointIO("some sort", testiofunc)
#model1.run()
