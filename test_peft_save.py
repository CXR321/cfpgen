import torch
import torch.nn as nn
import re
from peft import LoraConfig, get_peft_model

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

model = DummyModel()
config = LoraConfig(target_modules=["fc1"])
peft_model = get_peft_model(model, config)

# mimic the unfreeze logic
lora_target_module = "fc1"
for name, param in peft_model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
    elif re.search(lora_target_module, name):
        param.requires_grad = False
    else:
        param.requires_grad = True

# Check state_dict
sd = peft_model.state_dict()
print("Keys in state_dict:")
for k in sd.keys():
    print("  ", k)

