
from serveai.app import agent
def testiofunc(inpt:str):
    return f"this is output of {inpt}"
    
model1 = agent("some sort", testiofunc)
model1.run()
