import torch
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from neural_models.Net import Net

####################################################################################

CHECKPOINT_DIRECTORY = "../../checkpoints"

MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"
MNIST_HEIGHT = 28
MNIST_WIDTH = 28
MNIST_CHANNELS = 1

MNIST_MODEL_DIRECTORY = "checkpoints/mnist"

device = 'cuda' if torch.cuda.is_available() else 'cpu'


####################################################################################

def load_neural_network(dataset_name):
	net = None
	if dataset_name == MNIST:
		net = Net().to(device)
		net.load_state_dict(torch.load(os.path.join(MNIST_MODEL_DIRECTORY, "mnist_cnn.pt")))
		net.eval()
	return net


def fgsm_attack(inputs, epsilon, data_grad):
	sign_data_grad = data_grad.sign()
	perturbed_inputs = inputs + epsilon * sign_data_grad
	perturbed_inputs = torch.clamp(perturbed_inputs, -1, 1)
	return perturbed_inputs


def generate_adv_sample_mnist(inputs, net, epsilon=0.05, target=None):
	inputs = inputs.clone()
	inputs = inputs.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
	target = (target.reshape(-1)).to(torch.int64)
	inputs.requires_grad = True

	output = net.forward(inputs)
	loss = F.nll_loss(output, target.reshape(-1))
	net.zero_grad()
	loss.backward()
	data_grad = inputs.grad.data
	perturbed_inputs = fgsm_attack(inputs, epsilon, data_grad)
	perturbed_inputs = perturbed_inputs.reshape((-1, MNIST_HEIGHT * MNIST_WIDTH))
	perturbed_inputs = perturbed_inputs.detach()
	return perturbed_inputs


def generate_adv_dataset(inputs, target, dataset_name, epsilon=0.05, combine=False):
	adv_inputs = inputs.clone()
	adv_target = target.clone()
	original_N = inputs.shape[0]
	dataset = TensorDataset(inputs, target)
	data_loader = DataLoader(dataset, batch_size=100, shuffle=True)

	net = load_neural_network(dataset_name)
	for batch_inputs, batch_target in data_loader:
		if dataset_name == MNIST:
			perturbed_inputs = generate_adv_sample_mnist(batch_inputs, net, epsilon, batch_target)
			adv_inputs = torch.cat((adv_inputs, perturbed_inputs))
			adv_target = torch.cat((adv_target, batch_target))
	if combine:
		return adv_inputs, adv_target
	else:
		return adv_inputs[original_N:, :], adv_target[original_N:, :]
