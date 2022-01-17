import torch
import torch.nn as nn
import torch.nn.functional as F


class DEBNet(nn.Module):

	def __init__(self, num_features, num_classes):
		super(DEBNet, self).__init__()
		self.fc1 = nn.Linear(num_features, min(num_features * 5, 1000))
		self.fc2 = nn.Linear(min(num_features * 5, 1000), 100)
		self.fc3 = nn.Linear(100, num_classes)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.dropout1(x)
		x = F.relu(self.fc2(x))
		x = self.dropout2(x)
		x = self.fc3(x)
		return x
