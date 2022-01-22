import torch
import random
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import *

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def generate_perturbed_samples(inputs, k=10):
	inputs = inputs.reshape((1, -1))
	num_dims = inputs.shape[1]
	perturbed_set = []
	dim_idx = random.sample(range(0, num_dims), k)
	for dimension in dim_idx:
		perturbed_samples = inputs.clone().detach()
		perturbed_samples[:, dimension] = 1 - perturbed_samples[:, dimension]
		perturbed_set.append(perturbed_samples)
	perturbed_set = torch.cat(perturbed_set)
	return perturbed_set


def generate_adversarial_sample_batched(einet, inputs, k=10):
	batch_size, num_dims = inputs.shape
	iteration_inputs = inputs.clone().detach()
	for iteration in range(BINARY_DEBD_HAMMING_THRESHOLD):
		batched_perturbed_set = []
		for batch_idx in range(batch_size):
			batch_input = iteration_inputs[batch_idx, :]
			perturbed_input_set = generate_perturbed_samples(batch_input, k)
			batched_perturbed_set.append(perturbed_input_set)
		batched_perturbed_set = torch.cat(batched_perturbed_set)

		outputs = (einet(batched_perturbed_set)).clone().detach()
		arg_min_idx = []
		for batch_idx in range(batch_size):
			batch_input_min_idx = torch.argmin(outputs[batch_idx * k:min((batch_idx + 1) * k, outputs.shape[0])])
			arg_min_idx.append(batch_idx * k + batch_input_min_idx)
		iteration_inputs = batched_perturbed_set[arg_min_idx, :]
	adv_sample_batched = iteration_inputs
	return adv_sample_batched


def generate_adversarial_sample(einet, inputs, k=10):
	iteration_inputs = inputs.clone().detach()
	for iteration in range(BINARY_DEBD_HAMMING_THRESHOLD):
		num_dims = iteration_inputs.shape[1]
		perturbed_set = []
		for dimension in range(num_dims):
			perturbed_sample = iteration_inputs.clone().detach()
			perturbed_sample[:, dimension] = 1 - perturbed_sample[:, dimension]
			perturbed_set.append(perturbed_sample)
		perturbed_set = torch.cat(perturbed_set)

		outputs = (einet(perturbed_set)).clone().detach()
		min_idx = torch.argmin(outputs)
		iteration_inputs = perturbed_set[min_idx, :]
	adv_sample = iteration_inputs
	return adv_sample


def generate_adv_dataset(einet, dataset_name, inputs, labels, combine=True, batched=False, k=10):
	adv_inputs = inputs.detach().clone()
	original_N, num_dims = inputs.shape

	k = min(max(10, int(0.1*num_dims)), 50)
	batch_size = max(1, int(1000 / k)) if batched else 1

	dataset = TensorDataset(inputs)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)
	perturbed_inputs = []
	for inputs in data_loader:
		if batched:
			adv_sample = generate_adversarial_sample_batched(einet, inputs[0], k=k)
		else:
			adv_sample = generate_adversarial_sample(einet, inputs[0], k=k)
		perturbed_inputs.append(adv_sample)
	perturbed_inputs = torch.cat(perturbed_inputs)
	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
