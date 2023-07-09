import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='serveai',
      version='0.0.2',
      description='serve ai agents as api easily.',
      author='Navdeep Dhakar',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['serveai'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['fastapi', 'uvicorn', 'fire'],
      extras_require={
        'llm': ['accelerate', 'appdirs', 'loralib', 'bitsandbytes', 'black', 'black[jupyter]', 'datasets', 'git+https://github.com/huggingface/peft.git', 'transformers>=4.28.0', 'sentencepiece', 'gradio', 'scipy', 'tqdm'],
           },
      python_requires='>=3.6',
      include_package_data=True)
