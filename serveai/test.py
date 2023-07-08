
from app import endpointIO
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
model1 = endpointIO("some sort", testiofunc)
model1.run()
