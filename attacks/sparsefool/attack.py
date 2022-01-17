import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from attacks.sparsefool.sparsefool import sparsefool
from neural_models.MNet import Net
from constants import *

####################################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


####################################################################################


def sparsefool_attack(data, net, target=None):
	ub = torch.ones_like(data, device=device, dtype=torch.int8)
	lb = -torch.ones_like(data, device=device, dtype=torch.int8)
	x_adv, r, pred_label, fool_label, loops = sparsefool(data, net, lb, ub, device=device)
	return x_adv


def load_neural_network(dataset_name):
	net = None
	if dataset_name == MNIST:
		net = Net().to(device)
		net.load_state_dict(torch.load(os.path.join(MNIST_NET_DIRECTORY, "mnist_cnn.pt")))
		net.eval()
	return net


def generate_adv_sample_mnist(inputs, net, target=None):
	inputs = inputs.clone()
	inputs = inputs.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
	target = (target.reshape(-1)).to(torch.int64)

	perturbed_inputs = sparsefool_attack(inputs, net, target)
	perturbed_inputs = perturbed_inputs.reshape((-1, MNIST_HEIGHT * MNIST_WIDTH))
	perturbed_inputs = perturbed_inputs.detach()
	return perturbed_inputs


def generate_adv_dataset(inputs, target, dataset_name, combine=False):
	adv_inputs = inputs.clone()
	adv_target = target.clone()
	original_N = inputs.shape[0]

	dataset = TensorDataset(inputs, target)
	data_loader = DataLoader(dataset, batch_size=DEFAULT_ATTACK_BATCH_SIZE, shuffle=True)

	net = load_neural_network(dataset_name)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)

	for batch_inputs, batch_target in data_loader:
		if dataset_name == MNIST:
			perturbed_inputs = generate_adv_sample_mnist(batch_inputs, net, batch_target)
			adv_inputs = torch.cat((adv_inputs, perturbed_inputs))
			adv_target = torch.cat((adv_target, batch_target))
	if combine:
		return adv_inputs, adv_target
	else:
		return adv_inputs[original_N:, :], adv_target[original_N:, :]
