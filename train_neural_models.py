import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as torch_datasets
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset
from torchvision import transforms
from tqdm import tqdm

import datasets
from constants import *
from neural_models.DEBNet import DEBNet
from neural_models.MNet import MNet
from neural_models.BMNet import BMNet
from utils import mkdir_p, to_torch_tensor, predict_labels_mnist

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


def test_neural(model, test_loader):
	model.eval()
	test_loss = 0
	correct = 0
	data_loader = tqdm(
		test_loader, leave=True, bar_format='{l_bar}{bar:24}{r_bar}', desc="Test",
		unit='batch'
	)
	with torch.no_grad():
		for data, target in data_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	print('\nNeural Network Test Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))


def train_mnist():
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	dataset1 = torch_datasets.MNIST(DATA_DIRECTORY, train=True, download=True, transform=transform)
	dataset2 = torch_datasets.MNIST(DATA_DIRECTORY, train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset2, batch_size=100, shuffle=True)

	# Training settings
	model = MNet(10).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(MNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(MNIST_NET_PATH, MNIST_NET_FILE))


def train_fashion_mnist():
	transform = transforms.Compose([
		transforms.ToTensor()
	])

	dataset1 = torch_datasets.FashionMNIST(DATA_DIRECTORY, train=True, download=True, transform=transform)
	dataset2 = torch_datasets.FashionMNIST(DATA_DIRECTORY, train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset2, batch_size=100, shuffle=True)

	# Training settings
	model = MNet(10).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(FASHION_MNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(FASHION_MNIST_NET_PATH, FASHION_MNIST_NET_FILE))


def train_kmnist():
	transform = transforms.Compose([
		transforms.ToTensor()
	])

	dataset1 = torch_datasets.KMNIST(DATA_DIRECTORY, train=True, download=True, transform=transform)
	dataset2 = torch_datasets.KMNIST(DATA_DIRECTORY, train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset2, batch_size=100, shuffle=True)

	# Training settings
	model = MNet(10).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(KMNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(KMNIST_NET_PATH, KMNIST_NET_FILE))


def train_emnist():
	transform = transforms.Compose([
		transforms.ToTensor()
	])

	dataset1 = torch_datasets.EMNIST(DATA_DIRECTORY, train=True, download=True, transform=transform)
	dataset2 = torch_datasets.EMNIST(DATA_DIRECTORY, train=False, transform=transform)

	train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset2, batch_size=100, shuffle=True)

	# Training settings
	model = MNet(62).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(EMNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(EMNIST_NET_PATH, EMNIST_NET_FILE))


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
	print(" Training Neural Network for {}".format(dataset_name))
	print("--------------------------------------------------------------------")

	train_x, valid_x, test_x = datasets.load_debd(dataset_name)
	train_labels, valid_labels, test_labels = generate_debd_labels(dataset_name, train_x, valid_x, test_x)

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = to_torch_tensor(train_x, valid_x, test_x,
																						train_labels, valid_labels,
																						test_labels)

	data_train = TensorDataset(train_x, train_labels)
	data_test = TensorDataset(test_x, test_labels)
	data_valid = TensorDataset(valid_x, valid_labels)

	train_loader = torch.utils.data.DataLoader(data_train, batch_size=100, shuffle=True)
	test_loader = torch.utils.data.DataLoader(data_test, batch_size=100, shuffle=True)

	# Training settings
	model = DEBNet(train_x.shape[1], 10).to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999),
						   eps=1e-08, weight_decay=0, amsgrad=False)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(DEBD_NET_PATH)
	torch.save(model.state_dict(), os.path.join(DEBD_NET_PATH, "{}.pt".format(dataset_name)))

	del model, train_x, valid_x, test_x, train_labels, valid_labels, test_labels, data_train, data_valid, data_test
	torch.cuda.empty_cache()


def train_debd_datasets():
	for dataset_name in DEBD_DATASETS:
		train_debd(dataset_name)


def train_binary_mnist():
	train_x, valid_x, test_x = datasets.load_binarized_mnist_dataset()

	train_x = train_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
	valid_x = valid_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
	test_x = test_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))

	train_labels = predict_labels_mnist(train_x)
	valid_labels = predict_labels_mnist(valid_x)
	test_labels = predict_labels_mnist(test_x)

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = to_torch_tensor(train_x, valid_x, test_x,
																						train_labels, valid_labels,
																						test_labels)

	data_train = TensorDataset(train_x, train_labels)
	data_test = TensorDataset(test_x, test_labels)

	train_loader = torch.utils.data.DataLoader(data_train, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
	test_loader = torch.utils.data.DataLoader(data_test, shuffle=True, batch_size=EVAL_BATCH_SIZE)

	# Training settings
	model = BMNet().to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	for epoch in range(TRAIN_NEURAL_NET_MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test_neural(model, test_loader)

	mkdir_p(BINARY_MNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(BINARY_MNIST_NET_PATH, BINARY_MNIST_NET_FILE))


if __name__ == '__main__':
	# train_mnist()
	# train_debd_datasets()
	# train_binary_mnist()
	# train_fashion_mnist()
	train_kmnist()
	train_emnist()
