import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

model = DummyModel()
config = LoraConfig(target_modules=["fc1"])
peft_model = get_peft_model(model, config)

for name, param in peft_model.named_parameters():
    print(name, param.requires_grad)

