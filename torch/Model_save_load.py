## https://pytorch.org/tutorials/beginner/saving_loading_models.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class TheModoleClass(nn.Module):
	def __init__(self):
		super(TheModoleClass, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc2(x)
		return x 

model = TheModoleClass()

optimizer = optim.SGD(model.parameters(),\
 lr = 0.001, momentum=0.9)


print("Model's state dict:")
for param_tensor in model.state_dict():
	print(param_tensor, "\t",
		model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
	print(var_name, "\t",
		optimizer.state_dict()[var_name])



## Saving & Loading Model for Inference

PATH = os.getcwd() + '\\checkpoints\\model.pth'
# print('Creating folder')
# os.makedirs(PATH, mode=0o777, exist_ok=True)
torch.save(model.state_dict(), PATH)

model = TheModoleClass()
model.load_state_dict(torch.load(PATH))
print('Model loaded')
model.eval()

## Save/Load Entire Model
PATH_entire_model = os.getcwd() + '\\checkpoints\\entire_model.pth'
torch.save(model, PATH_entire_model)
print('saved entire model')
model = torch.load(PATH_entire_model)
model.eval()


## Saving & Loading a General Checkpoint for Inference and/or Resuming Training

