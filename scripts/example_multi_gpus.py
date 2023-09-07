import torch
import torch.nn as nn
import torchvision
from concurrent.futures import ThreadPoolExecutor, wait
import time
import random 

# Dummy models to be evaluated
dense1 = torchvision.models.densenet121(pretrained=False)
dense2 = torchvision.models.densenet121(pretrained=False)
dense3 = torchvision.models.densenet121(pretrained=False)
rest1 = torchvision.models.resnet101(pretrained=False)
rest2 = torchvision.models.resnet101(pretrained=False)
rest3 = torchvision.models.resnet101(pretrained=False)
rest4 = torchvision.models.resnet101(pretrained=False)
models = [dense1, dense2, dense3, rest1, rest2, rest3, rest4]
models_names = [f"model_{i}" for i in range(len(models))]

# Dummy inputs
inputs = [torch.ones((200,3,32,32)) for i in range(len(models))]

# Target function
def train(model, name, value, epoch, device, val1, val2, val3):
    model = model.to(device)
    value = value.to(device)
    print(f"started model {name} with {epoch} @ {device}")
    ti = time.perf_counter()
    for _ in range(epoch):
        ret = model(value)
    to = time.perf_counter()
    print(f"finished model {name} with {epoch} @ {device}")
    
    elapsed_time = to-ti
    return elapsed_time, name, str(val1+val2+val3)

output = []

# Create a list containing definitions for each model
runners_params = []
random.seed(2023)
for idx, (model_name, model, input) in enumerate(zip(models_names, models, inputs)):
    # epochs    
    epochs = random.randint(30,50)        
    # args specific for each model 
    kwargs = {"model": model, "name": model_name, "value": input, "epoch": epochs}
    runners_params.append(kwargs)   

# Using GPU Manager object to parallelize inference (one model per GPU)
from src.gpu_manager import MultiGPUManager
kwargs = {"val1": 0.1, "val2": 0.01, "val3": 0.001}
manager = MultiGPUManager()
responses = manager.run(target_fn=train, iterable_items=runners_params, kwargs=kwargs, spawn=False)

for response in responses:
    print(response)
    