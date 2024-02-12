from streamai.app import endpointIO
from streamai.llms import AutoMistral
import json
modelinstance = AutoMistral(base_model="mistralai/Mistral-7B-v0.1b")

# let's see what is the structure of dataset that is required for training
print(json.dumps(modelinstance.info['dataset'], indent = 1))

#let's provide the dataset url
modelinstance.train(dataset_url="https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b", model_name="mistral7btest") # here it is using dummy dataset available on this link.
