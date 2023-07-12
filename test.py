from streamai.app import endpointIO
from streamai.llms import Autoalpacalora

modelinstance = Autoalpacalora(base_model="decapoda-research/llama-7b-hf")
modelinstance.train(dataset_url="https://firebasestorage.googleapis.com/v0/b/pdf-analysis-saas.appspot.com/o/Other%2Fdataset.json?alt=media&token=28abd658-a308-4050-b631-54bab9b63a6b") # when finetuning
#then load fine tuned model using lora weights
modelinstance.loadmodel(lora_weights="alpacafinetuned") #provide lora weights if model is finetuned, it is directory that will be logged after training is finished and you can paste it here as lora weights.
finetunedmodel = endpointIO(modelinstance.inferenceIO)
finetunedmodel.run()
