from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import string
import uvicorn
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Data(BaseModel):
    input: str
    description: Union[str, None] = None

endpoint_secret = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
            
class endpointIO:
    def __init__(self, iofunc):
        self.iopipe = iofunc

    def run(self, port=8080, workers=4):
        print("Starting AI services")
        print(f"Inference endpoint -> http://0.0.0.0:{port}/{endpoint_secret}/inference")
        print(f"endpoint secret {endpoint_secret}")
        @app.post(f"/{endpoint_secret}/inference")
        async def infernece(data: Data):
            output = self.iopipe(data.input)
            return {"return": f"{output}"}
        uvicorn.run(app, host="0.0.0.0", port=8080)
