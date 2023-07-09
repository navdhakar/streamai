import subprocess, sys
class ATalpacalora:
    def __init__(self, input):
        self.input = input
        require_install = ['accelerate', 'appdirs', 'loralib', 'bitsandbytes', 'black', 'black[jupyter]', 'datasets', 'fire', 'git+https://github.com/huggingface/peft.git', 'transformers>=4.28.0', 'sentencepiece', 'gradio', 'scipy', 'tqdm']
        for package in require_install:
            subprocess.run([sys.executable, "-m", "pip", "install", package])
        from serveai.models.alpacalora import AutoTrainalpacalora
    def deploy(self):
        return f"training model"
