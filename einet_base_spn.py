import os
import torch
import numpy as np
import datasets
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from utils import mkdir_p, save_image_stack
from EinsumNetwork import EinsumNetwork, Graph
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray
from attacks.fgsm import attack as fgsm_attack

from constants import *
from utils import predict_labels_mnist
from attacks.sparsefool import attack as sparsefool_attack
from train_neural_models import generate_debd_labels

from deeprob.torch.callbacks import EarlyStopping

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


def load_structure(structure, dataset_name, structure_args):
	graph = None
	if structure == POON_DOMINGOS:
		height = structure_args[HEIGHT]
		width = structure_args[WIDTH]
		pd_num_pieces = structure_args[PD_NUM_PIECES]

		mkdir_p(STRUCTURE_DIRECTORY)
		file_name = os.path.join(STRUCTURE_DIRECTORY, "_".join([structure, dataset_name]) + ".pc")
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
		file_name = os.path.join(STRUCTURE_DIRECTORY, "{}_{}_{}.pc".format(structure, dataset_name, num_repetitions))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			graph = Graph.random_binary_trees(num_var=num_var, depth=depth, num_repetitions=num_repetitions)
			Graph.write_gpickle(graph, file_name)
	return graph


def load_einet(structure, dataset_name, einet_args):
	args, graph = None, None
	if structure == POON_DOMINGOS:
		file_name = os.path.join(STRUCTURE_DIRECTORY, "{}_{}.pc".format(structure, dataset_name))
		if os.path.exists(file_name):
			graph = Graph.read_gpickle(file_name)
		else:
			AssertionError("Graph for the corresponding structure is not stored, generate graph first")

	else:
		# Structure - Binary Trees
		mkdir_p(STRUCTURE_DIRECTORY)
		file_name = os.path.join(STRUCTURE_DIRECTORY,
								 "{}_{}_{}.pc".format(structure, dataset_name, einet_args['num_repetitions']))
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


def load_pretrained_einet(structure, dataset_name, einet_args):
	einet = None
	mkdir_p(MODEL_DIRECTORY)

	file_name = os.path.join(MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args['num_sums'],
														 einet_args['num_input_distributions'],
														 einet_args['num_repetitions']))
	if os.path.exists(file_name):
		torch.load(einet, file_name)
	else:
		AssertionError("Einet for the corresponding structure is not stored, train first")

	einet.initialize()
	einet.to(device)
	return einet


def generate_samples(einet, structure, dataset_name, einet_args):
	einet.eval()
	DATASET_SAMPLES_DIR = os.path.join(SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		samples = einet.sample(num_samples=25).cpu().numpy()
		samples = samples.reshape((-1, 28, 28))
		file_name = "{}_{}_{}_{}_{}_sample.png".format(structure, dataset_name, einet_args['num_sums'],
													   einet_args['num_input_distributions'],
													   einet_args['num_repetitions'])
		save_image_stack(samples, 5, 5, os.path.join(DATASET_SAMPLES_DIR, file_name), margin_gray_val=0.)


def generate_adv_samples(einet, structure, dataset_name, einet_args, epsilon):
	einet.eval()
	DATASET_SAMPLES_DIR = os.path.join(SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		samples = einet.sample(num_samples=25).cpu().numpy()
		samples = samples.reshape((-1, 28, 28))

		file_name = "{}_{}_{}_{}_{}_{}_adv_sample.png".format(structure, dataset_name, einet_args['num_sums'],
															  einet_args['num_input_distributions'],
															  einet_args['num_repetitions'], epsilon)
		save_image_stack(samples, 5, 5, os.path.join(DATASET_SAMPLES_DIR, file_name), margin_gray_val=0.)


def generate_conditional_samples(einet, structure, dataset_name, einet_args, test_x):
	einet.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_CONDITIONAL_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:

		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))
		keep_idx = [i for i in range(MNIST_WIDTH * MNIST_HEIGHT) if i not in marginalize_idx]
		einet.set_marginalization_idx(marginalize_idx)

		num_samples = 10
		samples = None
		for k in range(num_samples):
			if samples is None:
				samples = einet.sample(x=test_x[0:25, :]).cpu().numpy()
			else:
				samples += einet.sample(x=test_x[0:25, :]).cpu().numpy()
		samples /= num_samples
		samples = samples.squeeze()

		samples = samples.reshape((-1, 28, 28))
		sample_reconstruction_file = "{}_{}_{}_{}_{}_sample_reconstruction.png".format(structure, dataset_name,
																					   einet_args['num_sums'],
																					   einet_args[
																						   'num_input_distributions'],
																					   einet_args['num_repetitions'])
		save_image_stack(samples, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, sample_reconstruction_file),
						 margin_gray_val=0.)

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_{}_ground_truth.png".format(structure, dataset_name,
																	 einet_args['num_sums'],
																	 einet_args[
																		 'num_input_distributions'],
																	 einet_args['num_repetitions'])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		mpe_reconstruction = einet.mpe(x=test_x[0:25, :]).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_{}_mpe_reconstruction.png".format(structure, dataset_name,
																				 einet_args['num_sums'],
																				 einet_args[
																					 'num_input_distributions'],
																				 einet_args['num_repetitions'])
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def generate_conditional_adv_samples(einet, structure, dataset_name, einet_args, test_x, epsilon):
	einet.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_CONDITIONAL_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:

		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))
		keep_idx = [i for i in range(MNIST_WIDTH * MNIST_HEIGHT) if i not in marginalize_idx]
		einet.set_marginalization_idx(marginalize_idx)

		num_samples = 10
		samples = None
		for k in range(num_samples):
			if samples is None:
				samples = einet.sample(x=test_x[0:25, :]).cpu().numpy()
			else:
				samples += einet.sample(x=test_x[0:25, :]).cpu().numpy()
		samples /= num_samples
		samples = samples.squeeze()

		samples = samples.reshape((-1, 28, 28))

		sample_reconstruction_file = "{}_{}_{}_{}_{}_{}_adv_sample_reconstruction.png".format(structure, dataset_name,
																							  einet_args['num_sums'],
																							  einet_args[
																								  'num_input_distributions'],
																							  einet_args[
																								  'num_repetitions'],
																							  epsilon)
		save_image_stack(samples, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, sample_reconstruction_file),
						 margin_gray_val=0.)

		mpe_reconstruction = einet.mpe(x=test_x[0:25, :]).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_{}_{}_adv_mpe_reconstruction.png".format(structure, dataset_name,
																						einet_args['num_sums'],
																						einet_args[
																							'num_input_distributions'],
																						einet_args['num_repetitions'],
																						epsilon)
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def fetch_num_epochs(dataset_name):
	NUM_EPOCHS = EINET_MAX_NUM_EPOCHS

	if dataset_name in ['bnetflix', 'kosarek', 'msweb', 'tretail', 'plants', 'accidents', 'nltcs', 'msnbc', 'kdd',
						'pumsb_star', 'jester']:
		NUM_EPOCHS = 30
	elif dataset_name in ['baudio']:
		NUM_EPOCHS = 40
	elif dataset_name in ['c20ng', 'dna']:
		NUM_EPOCHS = 100
	elif dataset_name in ['ad', 'cr52', 'cwebkb', 'tmovie', 'book']:
		NUM_EPOCHS = 150
	elif dataset_name in ['bbc', ]:
		NUM_EPOCHS = 150
	return NUM_EPOCHS


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


def train_einet(structure, dataset_name, einet, train_x, train_labels, valid_x, valid_labels, test_x, test_labels,
				einet_args, batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=False):
	if is_adv:
		train_dataset = fetch_adv_data(einet, dataset_name, train_x, train_labels, adv_type=TRAIN_DATASET)
	else:
		train_dataset = TensorDataset(train_x)

	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

	early_stopping = EarlyStopping(einet, patience=DEFAULT_EINET_PATIENCE, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)

	# NUM_EPOCHS = fetch_num_epochs(dataset_name)
	NUM_EPOCHS = 1
	for epoch_count in range(NUM_EPOCHS):
		epoch_einet_train(train_dataloader, einet, epoch_count, dataset_name, weight=1)
		train_ll, valid_ll, test_ll = evaluate_lls(einet, train_x, valid_x, test_x, epoch_count=epoch_count)
		early_stopping(-valid_ll, epoch_count)
		if early_stopping.should_stop:
			print("Early Stopping... {}".format(early_stopping))
			break

	mkdir_p(EINET_MODEL_DIRECTORY)

	file_name = None
	if is_adv:
		file_name = os.path.join(EINET_MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}_adv.mdl".format(structure, dataset_name,
																 einet_args[NUM_SUMS],
																 einet_args[NUM_INPUT_DISTRIBUTIONS],
																 DEFAULT_NUM_REPETITIONS))
	else:
		file_name = os.path.join(EINET_MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
															 einet_args[NUM_INPUT_DISTRIBUTIONS],
															 einet_args[NUM_REPETITIONS]))

	torch.save(einet, file_name)

	return einet


def train_weighted_einet(structure, dataset_name, einet, train_x, train_labels, orig_train_size, valid_x, valid_labels,
						 test_x,
						 test_labels, einet_args, batch_size=DEFAULT_TRAIN_BATCH_SIZE):
	einet.train()

	original_train_data = train_x[0:orig_train_size, :]
	augmented_train_data = train_x[original_train_data, :]

	original_train_dataset = TensorDataset(original_train_data)
	original_train_dataloader = DataLoader(original_train_dataset, batch_size, shuffle=True)

	augmented_train_dataset = TensorDataset(augmented_train_data)
	augmented_train_dataloader = DataLoader(augmented_train_dataset, batch_size, shuffle=True)

	for epoch_count in range(EINET_MAX_NUM_EPOCHS):
		epoch_einet_train(original_train_dataloader, einet, epoch_count, dataset_name, weight=1)
		epoch_einet_train(augmented_train_dataloader, einet, epoch_count, dataset_name,
						  weight=AUGMENTED_DATA_WEIGHT_PARAMETER)

		# Evaluate
		einet.eval()
		train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=EVAL_BATCH_SIZE)
		valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=EVAL_BATCH_SIZE)
		test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=EVAL_BATCH_SIZE)
		print("[{}] train LL {} valid LL {} test LL {}".format(epoch_count, train_ll / train_x.shape[0],
															   valid_ll / valid_x.shape[0], test_ll / test_x.shape[0]))

	mkdir_p(WEIGHTED_EINET_MODEL_DIRECTORY)

	file_name = os.path.join(WEIGHTED_EINET_MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}.mdl".format(structure, dataset_name, einet_args[NUM_SUMS],
														 einet_args[NUM_INPUT_DISTRIBUTIONS],
														 einet_args[NUM_REPETITIONS]))

	torch.save(einet, file_name)

	return einet


def fetch_adv_data(einet, dataset_name, inputs, labels, adv_type=TRAIN_DATASET):
	attack, DATA_DIRECTORY = None, None
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/{}/augmented/sparsefool".format(dataset_name)
	elif dataset_name in DEBD_DATASETS:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/DEBD/datasets/{}/augmented/sparsefool/{}".format(dataset_name,
																				BINARY_DEBD_HAMMING_THRESHOLD)

	test_file_path = os.path.join(DATA_DIRECTORY, "{}.pt".format(adv_type))
	if os.path.exists(test_file_path):
		data_test = torch.load(test_file_path)
	else:
		combine = True if adv_type == TRAIN_DATASET or adv_type == VALID_DATASET else False
		inputs, labels = attack.generate_adv_dataset(dataset_name, inputs, labels, combine=combine)
		data_test = TensorDataset(inputs)
		mkdir_p(DATA_DIRECTORY)
		torch.save(data_test, test_file_path)
	return data_test


def test_einet(dataset_name, einet, test_x, test_labels, einet_args, batch_size=1, is_adv=False):
	einet.eval()
	if is_adv:
		test_x = fetch_adv_data(einet, dataset_name, test_x, test_labels, adv_type=TEST_DATASET).tensors[0]
	test_lls = EinsumNetwork.fetch_likelihoods_for_data(einet, test_x, batch_size=batch_size)
	mean_ll = (torch.mean(test_lls)).cpu().item()
	stddev_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
	return mean_ll, stddev_ll


def test_conditional_einet(dataset_name, einet, evidence_percentage, einet_args, test_x, test_labels,
						   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=False):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	einet.eval()

	if is_adv:
		test_x = fetch_adv_data(einet, dataset_name, test_x, test_labels, adv_type=TEST_DATASET).tensors[0]

	test_lls = EinsumNetwork.fetch_conditional_likelihoods_for_data(einet, test_x, marginalize_idx=marginalize_idx,
																	batch_size=batch_size)
	mean_ll = (torch.mean(test_lls)).cpu().item()
	stddev_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
	return mean_ll, stddev_ll
