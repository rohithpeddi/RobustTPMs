import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import datasets
from EinsumNetwork import EinsumNetwork, Graph
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from constants import *
from deeprob.torch.callbacks import EarlyStopping
from train_neural_models import generate_debd_labels
from utils import mkdir_p
from utils import predict_labels_mnist
from attacks.localsearch import attack as local_search_attack
from attacks.localrestrictedsearch import attack as local_restricted_search_attack
from attacks.sparsefool import attack as sparsefool_attack

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def generate_exponential_family_args(exponential_family, dataset_name):
	exponential_family_args = None
	if exponential_family == BinomialArray:
		exponential_family_args = {'N': 255}
	elif exponential_family == CategoricalArray:
		if dataset_name == BINARY_MNIST or dataset_name in DEBD_DATASETS:
			exponential_family_args = {'K': 2}
		else:
			exponential_family_args = {'K': 256}
	elif exponential_family == NormalArray:
		exponential_family_args = {'min_var': 1e-6, 'max_var': 0.1}
	return exponential_family_args


def load_dataset(dataset_name):
	train_x, train_labels, test_x, test_labels, valid_x, valid_labels = None, None, None, None, None, None
	if dataset_name == FASHION_MNIST:
		train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
	elif dataset_name == MNIST:
		train_x, train_labels, test_x, test_labels = datasets.load_mnist()
		train_x /= 255.
		test_x /= 255.
		train_x -= 0.1307
		test_x -= 0.1307
		train_x /= 0.3081
		test_x /= 0.3081
		valid_x = train_x[-10000:, :]
		train_x = train_x[:-10000, :]
		valid_labels = train_labels[-10000:]
		train_labels = train_labels[:-10000]

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))
	elif dataset_name == BINARY_MNIST:
		train_x, valid_x, test_x = datasets.load_binarized_mnist_dataset()
		train_labels = predict_labels_mnist(train_x)
		valid_labels = predict_labels_mnist(valid_x)
		test_labels = predict_labels_mnist(test_x)

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))
	elif dataset_name in DEBD_DATASETS:
		train_x, test_x, valid_x = datasets.load_debd(dataset_name)
		train_labels, valid_labels, test_labels = generate_debd_labels(dataset_name, train_x, valid_x, test_x)

		train_x = torch.tensor(train_x, dtype=torch.float32, device=torch.device(device))
		valid_x = torch.tensor(valid_x, dtype=torch.float32, device=torch.device(device))
		test_x = torch.tensor(test_x, dtype=torch.float32, device=torch.device(device))

	train_labels = torch.from_numpy(train_labels.reshape(-1, 1)).to(torch.device(device))
	valid_labels = torch.from_numpy(valid_labels.reshape(-1, 1)).to(torch.device(device))
	test_labels = torch.from_numpy(test_labels.reshape(-1, 1)).to(torch.device(device))

	return train_x, valid_x, test_x, train_labels, valid_labels, test_labels


def load_structure(run_id, structure, dataset_name, structure_args):
	RUN_STRUCTURE_DIRECTORY = os.path.join("run_{}".format(run_id), STRUCTURE_DIRECTORY)
	mkdir_p(RUN_STRUCTURE_DIRECTORY)
	graph = None
	if structure == POON_DOMINGOS:
		height = structure_args[HEIGHT]
		width = structure_args[WIDTH]
		pd_num_pieces = structure_args[PD_NUM_PIECES]

		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY, "_".join([structure, dataset_name]) + ".pc")
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			pd_delta = [[height / d, width / d] for d in pd_num_pieces]
			graph = Graph.poon_domingos_structure(shape=(height, width), delta=pd_delta)
			Graph.write_gpickle(graph, file_name)
	else:
		# Structure - Binary Trees
		num_var = structure_args[NUM_VAR]
		depth = structure_args[DEPTH]
		num_repetitions = structure_args[NUM_REPETITIONS]

		mkdir_p(STRUCTURE_DIRECTORY)
		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY,
								 "{}_{}_{}.pc".format(structure, dataset_name, num_repetitions))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)
			Graph.write_gpickle(graph, file_name)
	return graph


def load_einet(run_id, structure, dataset_name, einet_args):
	RUN_STRUCTURE_DIRECTORY = os.path.join("run_{}".format(run_id), STRUCTURE_DIRECTORY)
	mkdir_p(RUN_STRUCTURE_DIRECTORY)
	args, graph = None, None
	if structure == POON_DOMINGOS:
		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY, "{}_{}.pc".format(structure, dataset_name))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			AssertionError("Graph for the corresponding structure is not stored, generate graph first")

	else:
		# Structure - Binary Trees
		file_name = os.path.join(RUN_STRUCTURE_DIRECTORY,
								 "{}_{}_{}.pc".format(structure, dataset_name, einet_args[NUM_REPETITIONS]))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			AssertionError("Graph for the corresponding structure is not stored, generate graph first")

	args = EinsumNetwork.Args(
		num_var=einet_args[NUM_VAR],
		num_dims=1,
		num_classes=GENERATIVE_NUM_CLASSES,
		num_sums=einet_args[NUM_SUMS],
		num_input_distributions=einet_args[NUM_INPUT_DISTRIBUTIONS],
		exponential_family=einet_args[EXPONENTIAL_FAMILY],
		exponential_family_args=einet_args[EXPONENTIAL_FAMILY_ARGS],
		online_em_frequency=einet_args[ONLINE_EM_FREQUENCY],
		online_em_stepsize=einet_args[ONLINE_EM_STEPSIZE])

	einet = EinsumNetwork.EinsumNetwork(graph, args)
	einet.initialize()
	einet.to(device)
	return einet


def epoch_einet_train(train_dataloader, einet, epoch, dataset_name, weight=1):
	train_dataloader = tqdm(
		train_dataloader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Training epoch : {}, for dataset : {}'.format(epoch, dataset_name),
		unit='batch'
	)
	einet.train()
	for inputs in train_dataloader:
		outputs = einet.forward(inputs[0])
		ll_sample = weight * EinsumNetwork.log_likelihoods(outputs)
		log_likelihood = ll_sample.sum()

		objective = log_likelihood
		objective.backward()

		einet.em_process_batch()
	einet.em_update()


def evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=0):
	# Evaluate
	einet.eval()
	train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=EVAL_BATCH_SIZE)
	valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=EVAL_BATCH_SIZE)
	test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=EVAL_BATCH_SIZE)
	print("[{}] train LL {} valid LL {} test LL {}".format(epoch_count, train_ll / train_x.shape[0],
														   valid_ll / valid_x.shape[0], test_ll / test_x.shape[0]))
	return train_ll / train_x.shape[0], valid_ll / valid_x.shape[0], test_ll / test_x.shape[0]


def save_model(run_id, einet, dataset_name, structure, einet_args, is_adv, attack_type=CLEAN):
	RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), EINET_MODEL_DIRECTORY)
	RUN_ATTACK_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, attack_type)
	mkdir_p(RUN_ATTACK_DIRECTORY)

	file_name = None
	if is_adv:
		file_name = os.path.join(RUN_ATTACK_DIRECTORY,
								 "{}_{}_{}_{}_{}_adv.mdl".format(structure, dataset_name,
																 einet_args[NUM_SUMS],
																 einet_args[NUM_INPUT_DISTRIBUTIONS],
																 DEFAULT_NUM_REPETITIONS))
	else:
		file_name = os.path.join(RUN_ATTACK_DIRECTORY,
								 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
															 einet_args[NUM_INPUT_DISTRIBUTIONS],
															 einet_args[NUM_REPETITIONS]))

	torch.save(einet, file_name)
	return


def train_einet(run_id, structure, dataset_name, einet, train_x, valid_x, test_x, einet_args,
				attack_type=CLEAN, batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=False):
	patience = 2 if is_adv else DEFAULT_EINET_PATIENCE

	early_stopping = EarlyStopping(einet, patience=patience, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)

	train_dataset = TensorDataset(train_x)
	NUM_EPOCHS = MAX_NUM_EPOCHS
	# NUM_EPOCHS = 1
	for epoch_count in range(NUM_EPOCHS):
		train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
		epoch_einet_train(train_dataloader, einet, epoch_count, dataset_name, weight=1)
		train_ll, valid_ll, test_ll = evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)
		early_stopping(-valid_ll, epoch_count)
		if early_stopping.should_stop:
			print("Early Stopping... {}".format(early_stopping))
			break
		if is_adv:
			print("Fetching adversarial data, training epoch {}".format(epoch_count))
			train_dataset = fetch_adv_data(einet, dataset_name, train_x, None, attack_type, combine=True)

	save_model(run_id, einet, dataset_name, structure, einet_args, is_adv, attack_type)

	return einet


def fetch_attack_method(attack_type):
	if attack_type == LOCAL_RESTRICTED_SEARCH:
		return local_restricted_search_attack
	elif attack_type == LOCAL_SEARCH:
		return local_search_attack
	elif attack_type == NEURAL_NET:
		return sparsefool_attack


def fetch_adv_data(einet, dataset_name, inputs, labels, attack_type, combine=True):
	attack = fetch_attack_method(attack_type)
	adv_data = attack.generate_adv_dataset(einet, dataset_name, inputs, labels, combine=combine, batched=True)
	adv_data = TensorDataset(adv_data)
	return adv_data


def get_stats(likelihoods):
	mean_ll = (torch.mean(likelihoods)).cpu().item()
	stddev_ll = (2.0 * torch.std(likelihoods) / np.sqrt(len(likelihoods))).cpu().item()
	return mean_ll, stddev_ll


def test_einet(dataset_name, einet, test_x, test_labels, attack_type=None, batch_size=1, is_adv=False):
	einet.eval()
	if is_adv:
		test_x = fetch_adv_data(einet, dataset_name, test_x, test_labels, attack_type, combine=False).tensors[0]
	test_lls = EinsumNetwork.fetch_likelihoods_for_data(einet, test_x, batch_size=batch_size)
	mean_ll, stddev_ll = get_stats(test_lls)
	return mean_ll, stddev_ll, test_x


def test_conditional_einet(dataset_name, einet, evidence_percentage, test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	einet.eval()
	test_lls = EinsumNetwork.fetch_conditional_likelihoods_for_data(einet, test_x, marginalize_idx=marginalize_idx,
																	batch_size=batch_size)
	mean_ll, stddev_ll = get_stats(test_lls)
	return mean_ll, stddev_ll
