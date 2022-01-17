import os

import numpy as np
import torch
from tqdm import tqdm
import datasets
import torch.nn.functional as F
import torch.optim as optim
from utils import mkdir_p, to_torch_tensor
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import torchvision.datasets as torch_datasets
from constants import *
from torch.utils.data import TensorDataset, DataLoader

from neural_models.MNet import Net
from neural_models.DEBNet import DEBNet
from sklearn.cluster import KMeans

#######################################################################################


torch.manual_seed(MANUAL_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################

def train(model, train_loader, optimizer, epoch):
	model.train()
	loss = torch.zeros(1)
	data_loader = tqdm(
		train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}', desc='Train Epoch: {}'.format(epoch),
		unit='batch'
	)
	for data, target in data_loader:
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
	print('\n Training epoch {}:  Loss: {:.4f}\n'.format(epoch, loss.item()))


def test(model, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	data_loader = tqdm(
		test_loader, leave=True, bar_format='{l_bar}{bar:24}{r_bar}', desc="Test",
		unit='batch'
	)
	with torch.no_grad():
		for data, target in data_loader:
			data = torch.round(data)
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def train_mnist():
	train_kwargs = {'batch_size': 100}
	test_kwargs = {'batch_size': 100}
	if torch.cuda.is_available():
		cuda_kwargs = {'num_workers': 1,
					   'pin_memory': True,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	dataset1 = torch_datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
	dataset2 = torch_datasets.MNIST(DATA_DIR, train=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

	# Training settings
	model = Net().to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test(model, test_loader)

	mkdir_p(MNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(MNIST_NET_PATH, MNIST_NET_FILE))


def generate_debd_labels(dataset_name, train_x, valid_x, test_x):
	kmeans = KMeans(n_clusters=NUM_CLUSTERS,
					verbose=0,
					max_iter=100,
					n_init=3).fit(train_x.reshape(train_x.shape[0], -1))

	train_labels = kmeans.predict(train_x)
	valid_labels = kmeans.predict(valid_x)
	test_labels = kmeans.predict(test_x)

	del kmeans
	torch.cuda.empty_cache()

	return train_labels, valid_labels, test_labels


def train_debd(dataset_name):

	print("--------------------------------------------------------------------")
	print(" Training Neural Network for {}". format(dataset_name))
	print("--------------------------------------------------------------------")

	train_kwargs = {'batch_size': 100}
	test_kwargs = {'batch_size': 100}
	if torch.cuda.is_available():
		cuda_kwargs = {'num_workers': 1,
					   'shuffle': True}
		train_kwargs.update(cuda_kwargs)
		test_kwargs.update(cuda_kwargs)

	train_x, valid_x, test_x = datasets.load_debd(dataset_name)
	train_labels, valid_labels, test_labels = generate_debd_labels(dataset_name, train_x, valid_x, test_x)

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = to_torch_tensor(train_x, valid_x, test_x, train_labels, valid_labels, test_labels)

	data_train = TensorDataset(train_x, train_labels)
	data_test = TensorDataset(test_x, test_labels)
	data_valid = TensorDataset(valid_x, valid_labels)

	train_loader = torch.utils.data.DataLoader(data_train, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(data_test, **test_kwargs)

	# Training settings
	model = DEBNet(train_x.shape[1], 10).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                 eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test(model, test_loader)

	mkdir_p(DEBD_NET_PATH)
	torch.save(model.state_dict(), os.path.join(DEBD_NET_PATH, "{}.pt".format(dataset_name)))

	del model, train_x, valid_x, test_x, train_labels, valid_labels, test_labels, data_train, data_valid, data_test
	torch.cuda.empty_cache()


def train_debd_datasets():
	for dataset_name in DEBD_DATASETS:
		train_debd(dataset_name)


if __name__ == '__main__':
	# train_mnist()
	train_debd_datasets()
