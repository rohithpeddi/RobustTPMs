import os

import torch
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
from attacks.sparsefool.sparsefool import sparsefool
from constants import *
from neural_models.MNet import MNet
from neural_models.BMNet import BMNet
from neural_models.DEBNet import DEBNet

####################################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


####################################################################################


def sparsefool_attack(dataset_name, data, net, target=None):
	ub, lb = None, None
	if dataset_name == MNIST:
		ub = torch.ones_like(data, device=torch.device(device), dtype=torch.int8)
		lb = -torch.ones_like(data, device=torch.device(device), dtype=torch.int8)
	elif dataset_name == BINARY_MNIST or dataset_name in DEBD_DATASETS:
		ub = torch.ones_like(data, device=torch.device(device), dtype=torch.int8)
		lb = torch.zeros_like(data, device=torch.device(device), dtype=torch.int8)
	x_adv, r, pred_label, fool_label, loops = sparsefool(data, net, lb, ub, device=torch.device(device))
	return x_adv


def generate_adv_sample(dataset_name, inputs, net, target=None):
	inputs = inputs.detach().clone()
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		inputs = inputs.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
	target = (target.reshape(-1)).to(torch.int64)

	perturbed_inputs = sparsefool_attack(dataset_name, inputs, net, target)
	perturbed_inputs = perturbed_inputs.detach()

	if dataset_name == BINARY_MNIST:
		perturbed_inputs = perturbed_inputs.reshape((-1, MNIST_HEIGHT * MNIST_WIDTH))
		inputs = inputs.reshape((-1, MNIST_HEIGHT * MNIST_WIDTH))
		if torch.sum(torch.abs(inputs - perturbed_inputs)) > BINARY_MNIST_HAMMING_THRESHOLD:
			return 0, None
		perturbed_inputs[perturbed_inputs < BINARY_MNIST_THRESHOLD] = 0
		perturbed_inputs[perturbed_inputs > BINARY_MNIST_THRESHOLD] = 1
		return 1, perturbed_inputs
	elif dataset_name in DEBD_DATASETS:
		if torch.sum(torch.abs(inputs - perturbed_inputs)) > BINARY_DEBD_HAMMING_THRESHOLD:
			return 0, None
		perturbed_inputs[perturbed_inputs < BINARY_DEBD_THRESHOLD] = 0
		perturbed_inputs[perturbed_inputs > BINARY_DEBD_THRESHOLD] = 1
		return 1, perturbed_inputs
	elif dataset_name == MNIST:
		perturbed_inputs = perturbed_inputs.reshape((-1, MNIST_HEIGHT * MNIST_WIDTH))
		return 1, perturbed_inputs


def generate_adv_dataset(einet, dataset_name, inputs, labels, combine=False, batched=True):
	adv_inputs = inputs.detach().clone()
	adv_target = labels.detach().clone()
	original_N = inputs.shape[0]

	dataset = TensorDataset(inputs, labels)
	data_loader = DataLoader(dataset, batch_size=DEFAULT_SPARSEFOOL_ATTACK_BATCH_SIZE, shuffle=True)

	net = load_neural_network(dataset_name, inputs)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)

	for batch_inputs, batch_target in data_loader:
		flag, perturbed_inputs = generate_adv_sample(dataset_name, batch_inputs, net, batch_target)
		if flag == 1:
			adv_inputs = torch.cat((adv_inputs, perturbed_inputs))
			adv_target = torch.cat((adv_target, batch_target))
	if combine:
		return adv_inputs
	else:
		return adv_inputs[original_N:, :]


def load_neural_network(dataset_name, data=None):
	net = None
	if dataset_name == MNIST:
		net = MNet().to(device)
		net.load_state_dict(torch.load(os.path.join(MNIST_NET_DIRECTORY, MNIST_NET_FILE)))
	elif dataset_name == BINARY_MNIST:
		net = BMNet().to(device)
		# net.load_state_dict(torch.load(os.path.join(MNIST_NET_DIRECTORY, MNIST_NET_FILE)))
		net.load_state_dict(torch.load(os.path.join(BINARY_MNIST_NET_PATH, BINARY_MNIST_NET_FILE)))
	elif dataset_name in DEBD_DATASETS:
		net = DEBNet(data.shape[1], 10).to(device)
		net.load_state_dict(torch.load(os.path.join(DEBD_NET_PATH, "{}.pt".format(dataset_name))))
	net.eval()
	return net
