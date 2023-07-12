from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import subprocess
import random, string
from streamai.llms import Autoalpacalora
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
endpointsecret = ''.join(random.choices(string.ascii_letters + string.digits, k=20))
class endpointIO:
    def __init__(self, iofunc):
        self.iopipe = iofunc 

    def run(self, port=8080):
        print("starting AI services")
        print(f"inference endpoint -> http://0.0.0.0:8000/{endpointsecret}/inference")
        @app.post(f"/xFlwFmy8qUlNUMibglik/inference")
        async def root(data:Data):
            output = self.iopipe(data.input)
            return {"return": f"{output}"}
        gunicorn_command = [
            "gunicorn",
            "main:app",
            "--workers",
            "4",
            "--worker-class",
            "uvicorn.workers.UvicornWorker",
            "--bind",
            f"0.0.0.0:{port}"
        ]
        subprocess.run(gunicorn_command)

