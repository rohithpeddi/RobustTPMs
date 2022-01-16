import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from utils import mkdir_p
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from neural_models.Net import Net

#######################################################################################

MAX_NUM_EPOCHS = 15

DATA_DIR = "data/"
MNIST_NET_PATH = "checkpoints/mnist/"
MNIST_NET_FILE = "mnist_cnn.pt"

MANUAL_SEED = 999

torch.manual_seed(MANUAL_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################

def train(model, train_loader, optimizer, epoch):
	model.train()
	loss = torch.zeros(1)
	data_loader = tqdm(
		train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}', desc='Train Epoch: {}'.format(epoch), unit='batch'
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

	dataset1 = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
	dataset2 = datasets.MNIST(DATA_DIR, train=False, transform=transform)
	train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
	test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

	# Training settings
	model = Net().to(device)
	optimizer = optim.Adadelta(model.parameters(), lr=1)

	scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
	for epoch in range(MAX_NUM_EPOCHS):
		train(model, train_loader, optimizer, epoch)
		test(model, test_loader)
		scheduler.step()

	mkdir_p(MNIST_NET_PATH)
	torch.save(model.state_dict(), os.path.join(MNIST_NET_PATH, MNIST_NET_FILE))


if __name__ == '__main__':
	train_mnist()
