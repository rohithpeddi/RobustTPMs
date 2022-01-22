import numpy as np
import torch

from tqdm import tqdm
from attacks.evolutionary.differential_evolution import differential_evolution
from constants import *
from torch.utils.data import TensorDataset, DataLoader

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def perturb_inputs(perturbations, X):
	if perturbations.ndim < 2:
		perturbations = np.array([perturbations])
	batch = len(perturbations)
	perturbed_inputs = X.repeat(batch, 1)
	perturbations = perturbations.astype(int)
	count = 0
	for x in perturbations:
		pixels = np.split(x, len(x) / 2)
		for pixel in pixels:
			x_pos, value = pixel
			perturbed_inputs[count, x_pos] = value
		count += 1
	return perturbed_inputs


def prediction_function(perturbations, X, einet):
	perturbed_inputs = perturb_inputs(perturbations, X.clone())
	outputs = einet(perturbed_inputs)
	return outputs.detach().cpu().numpy()


def attack_success(perturbation, X, einet):
	# attack_input = perturb_inputs(perturbation, X.clone())
	# output = einet(attack_input)
	return False


def attack(inputs, einet, pixels=BINARY_DEBD_HAMMING_THRESHOLD, maxiter=75, popsize=400):
	num_dim = inputs.shape[1]

	bounds = [(0, num_dim), (0, 1)] * pixels
	popmul = max(1, int(popsize / len(bounds)))

	predict_fn = lambda xs: prediction_function(xs, inputs, einet)
	callback_fn = lambda x, convergence: attack_success(x, inputs, einet)

	inits = np.zeros([popmul * len(bounds), len(bounds)])
	for init in inits:
		for i in range(pixels):
			init[i * 2 + 0] = int(np.random.random() * num_dim)
			init[i * 2 + 1] = 0 if np.random.random() < 0.5 else 1

	attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,
										   recombination=1, atol=-1, callback=callback_fn, polish=False, init=inits)

	attack_input = perturb_inputs(attack_result.x, inputs)

	return attack_input


def generate_adversarial_sample(einet, train_x):
	popsize = min(train_x.shape[1], 400)
	maxiter = min(int(train_x.shape[1]/2), 5)
	attacked_input = attack(train_x, einet, pixels=BINARY_DEBD_HAMMING_THRESHOLD,
							maxiter=maxiter,
							popsize=popsize)

	return attacked_input


def generate_adv_dataset(einet, dataset_name, inputs, combine=True):
	adv_inputs = inputs.detach().clone()
	original_N = inputs.shape[0]

	dataset = TensorDataset(inputs)
	data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

	data_loader = tqdm(
		data_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adv samples for {}'.format(dataset_name), unit='batch'
	)
	perturbed_inputs = []
	for inputs in data_loader:
		adv_sample = generate_adversarial_sample(einet, inputs[0])
		perturbed_inputs.append(adv_sample)
	perturbed_inputs = torch.cat(perturbed_inputs)
	if combine:
		return torch.cat((adv_inputs, perturbed_inputs))
	else:
		return perturbed_inputs
