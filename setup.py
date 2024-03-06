import os
from setuptools import setup, find_packages

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='streamai',
      version='0.1.2',
      description='finetune and deploy latest open source llms on cloud in less than 10 line of code.',
      author='Navdeep Dhakar',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['fastapi', 'uvicorn', 'fire'],
      python_requires='>=3.6',
      packages=find_packages(),
      include_package_data=True)
