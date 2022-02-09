import os
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset

import datasets
import deeprob.spn.models as spn_models
from EinsumNetwork import Graph
from attacks.sparsefool import attack as sparsefool_attack
from constants import *
from deeprob.torch.routines import train_model, test_model
from train_neural_models import generate_debd_labels
from utils import mkdir_p, save_image_stack, predict_labels_mnist

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def load_dataset(dataset_name):
	train_x, train_labels, test_x, test_labels, valid_x, valid_labels = None, None, None, None, None, None
	if dataset_name == FASHION_MNIST:
		train_x, train_labels, test_x, test_labels = datasets.load_fashion_mnist()
		train_x /= 255.
		test_x /= 255.
		train_x -= 0.28604
		test_x -= 0.28604
		train_x /= 0.3530
		test_x /= 0.3530
		valid_x = train_x[-10000:, :]
		train_x = train_x[:-10000, :]
		valid_labels = train_labels[-10000:]
		train_labels = train_labels[:-10000]

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))

		train_labels = ((torch.from_numpy(train_labels)).type(torch.int64)).to(torch.device(device))
		valid_labels = ((torch.from_numpy(valid_labels)).type(torch.int64)).to(torch.device(device))
		test_labels = ((torch.from_numpy(test_labels)).type(torch.int64)).to(torch.device(device))

		train_x = train_x.reshape((-1, FASHION_MNIST_CHANNELS, FASHION_MNIST_HEIGHT, FASHION_MNIST_WIDTH))
		valid_x = valid_x.reshape((-1, FASHION_MNIST_CHANNELS, FASHION_MNIST_HEIGHT, FASHION_MNIST_WIDTH))
		test_x = test_x.reshape((-1, FASHION_MNIST_CHANNELS, FASHION_MNIST_HEIGHT, FASHION_MNIST_WIDTH))
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

		train_labels = ((torch.from_numpy(train_labels)).type(torch.int64)).to(torch.device(device))
		valid_labels = ((torch.from_numpy(valid_labels)).type(torch.int64)).to(torch.device(device))
		test_labels = ((torch.from_numpy(test_labels)).type(torch.int64)).to(torch.device(device))

		train_x = train_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
		valid_x = valid_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))
		test_x = test_x.reshape((-1, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH))

	elif dataset_name == CIFAR_10:
		train_x, train_labels, test_x, test_labels = datasets.load_cifar_10()
		train_x /= 255.
		test_x /= 255.

		valid_x = train_x[-10000:, :]
		train_x = train_x[:-10000, :]
		valid_labels = train_labels[-10000:]
		train_labels = train_labels[:-10000]

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))

		train_labels = ((torch.from_numpy(train_labels)).type(torch.int64)).to(torch.device(device))
		valid_labels = ((torch.from_numpy(valid_labels)).type(torch.int64)).to(torch.device(device))
		test_labels = ((torch.from_numpy(test_labels)).type(torch.int64)).to(torch.device(device))

		train_x = train_x.reshape((-1, CIFAR_10_CHANNELS, CIFAR_10_HEIGHT, CIFAR_10_WIDTH))
		valid_x = valid_x.reshape((-1, CIFAR_10_CHANNELS, CIFAR_10_HEIGHT, CIFAR_10_WIDTH))
		test_x = test_x.reshape((-1, CIFAR_10_CHANNELS, CIFAR_10_HEIGHT, CIFAR_10_WIDTH))

	elif dataset_name == BINARY_MNIST:
		train_x, valid_x, test_x = datasets.load_binarized_mnist_dataset()
		train_labels = predict_labels_mnist(train_x)
		valid_labels = predict_labels_mnist(valid_x)
		test_labels = predict_labels_mnist(test_x)

		train_x = torch.from_numpy(train_x).to(torch.device(device))
		valid_x = torch.from_numpy(valid_x).to(torch.device(device))
		test_x = torch.from_numpy(test_x).to(torch.device(device))

		train_labels = torch.from_numpy(train_labels.reshape(-1, 1)).to(torch.device(device))
		valid_labels = torch.from_numpy(valid_labels.reshape(-1, 1)).to(torch.device(device))
		test_labels = torch.from_numpy(test_labels.reshape(-1, 1)).to(torch.device(device))

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
	# RUN_STRUCTURE_DIRECTORY = os.path.join("run_{}".format(run_id), STRUCTURE_DIRECTORY)
	RUN_STRUCTURE_DIRECTORY = os.path.join("", STRUCTURE_DIRECTORY)
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


def load_spn(dataset_name, spn_args):
	if dataset_name in [MNIST, FASHION_MNIST, CIFAR_10]:
		dgcspn = spn_models.DgcSpn(
			spn_args[N_FEATURES],
			out_classes=spn_args[OUT_CLASSES],  # The number of classes
			n_batch=spn_args[BATCHED_LEAVES],  # The number of batched leaves
			sum_channels=spn_args[SUM_CHANNELS],  # The sum layers number of channels
			depthwise=True,  # Use depthwise convolutions at every product layer
			n_pooling=spn_args[NUM_POOLING],  # Then number of initial pooling product layers
			in_dropout=spn_args[IN_DROPOUT],  # The probabilistic dropout rate to use at leaves layer
			sum_dropout=spn_args[SUM_DROPOUT],  # The probabilistic dropout rate to use at sum layers
			uniform_loc=(-1.5, 1.5)  # Initialize Gaussian locations uniformly
		)
		dgcspn.to(device)
		return dgcspn
	elif dataset_name in DEBD_DATASETS or dataset_name == BINARY_MNIST:
		ratspn = spn_models.BernoulliRatSpn(
			in_features=spn_args[N_FEATURES],
			out_classes=spn_args[OUT_CLASSES],
			rg_depth=spn_args[DEPTH],  # The region graph's depth
			rg_repetitions=spn_args[NUM_REPETITIONS],  # The region graph's number of repetitions
			rg_batch=spn_args[NUM_INPUT_DISTRIBUTIONS],  # The region graph's number of batched leaves
			rg_sum=spn_args[NUM_SUMS],  # The region graph's number of sum nodes per region
			in_dropout=DEFAULT_LEAF_DROPOUT,  # The probabilistic dropout rate to use at leaves layer
			sum_dropout=DEFAULT_SUM_DROPOUT
		)
		ratspn.to(device)
		return ratspn


def get_model_file_path(run_id, dataset_name, ratspn_args):
	file_path = None
	if dataset_name == MNIST:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), MNIST_MODEL_DIRECTORY)
		mkdir_p(RUN_MODEL_DIRECTORY)
		file_path = os.path.join(RUN_MODEL_DIRECTORY, "dgcspn_{}.pt".format(dataset_name))
	if dataset_name == CIFAR_10:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), CIFAR_10_MODEL_DIRECTORY)
		mkdir_p(RUN_MODEL_DIRECTORY)
		file_path = os.path.join(RUN_MODEL_DIRECTORY, "dgcspn_{}.pt".format(dataset_name))
	elif dataset_name in DEBD_DATASETS:
		DEBD_DATASET_MODEL_DIRECTORY = DEBD_MODEL_DIRECTORY + "/{}".format(dataset_name)
		mkdir_p(DEBD_DATASET_MODEL_DIRECTORY)
		file_path = os.path.join(DEBD_DATASET_MODEL_DIRECTORY,
								 "{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
														 ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														 ratspn_args[NUM_REPETITIONS]))
	elif dataset_name == BINARY_MNIST:
		mkdir_p(BINARY_MNIST_MODEL_DIRECTORY)
		file_path = os.path.join(BINARY_MNIST_MODEL_DIRECTORY,
								 "{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
														 ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														 ratspn_args[NUM_REPETITIONS]))

	return file_path


def load_pretrained_ratspn(run_id, dataset_name, spn_args):
	ratspn = load_spn(dataset_name, spn_args)
	file_path = get_model_file_path(run_id, dataset_name, spn_args)
	if os.path.exists(file_path):
		ratspn.load_state_dict(torch.load(file_path))
	else:
		print("Ratspn is not stored, train first")
		return None
	ratspn.to(device)
	return ratspn


def generate_samples(ratspn, dataset_name, ratspn_args):
	ratspn.eval()
	DATASET_SAMPLES_DIR = os.path.join(SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		samples = ratspn.sample(n_samples=25).cpu().numpy()
		samples = samples.reshape((-1, 28, 28))
		file_name = "{}_{}_{}_{}_sample.png".format(dataset_name, ratspn_args[NUM_SUMS],
													ratspn_args[NUM_INPUT_DISTRIBUTIONS],
													ratspn_args[NUM_REPETITIONS])
		save_image_stack(samples, 5, 5, os.path.join(DATASET_SAMPLES_DIR, file_name), margin_gray_val=0.)


def generate_adv_samples(ratspn, dataset_name, ratspn_args, epsilon=0.05):
	ratspn.eval()
	DATASET_SAMPLES_DIR = os.path.join(SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		samples = ratspn.sample(n_samples=25).cpu().numpy()
		samples = samples.reshape((-1, 28, 28))
		file_name = "{}_{}_{}_{}_{}_adv_sample.png".format(dataset_name, ratspn_args[NUM_SUMS],
														   ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														   ratspn_args[NUM_REPETITIONS], epsilon)
		save_image_stack(samples, 5, 5, os.path.join(DATASET_SAMPLES_DIR, file_name), margin_gray_val=0.)


def generate_conditional_samples(ratspn, dataset_name, ratspn_args, test_x):
	ratspn.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_CONDITIONAL_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_ground_truth.png".format(dataset_name, ratspn_args[NUM_SUMS],
																  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																  ratspn_args[NUM_REPETITIONS])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		test_batch_x = (test_x[0:25, :]).clone()
		test_batch_x[:, marginalize_idx] = np.nan

		mpe_reconstruction = ratspn.mpe(x=test_batch_x).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_mpe_reconstruction.png".format(dataset_name, ratspn_args[NUM_SUMS],
																			  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																			  ratspn_args[NUM_REPETITIONS])
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def test_conditional_likelihood(ratspn, dataset_name, evidence_percentage, ratspn_args, test_x):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	ratspn.eval()
	data_test = TensorDataset(test_x)

	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=CONDITIONAL,
								 batch_size=DEFAULT_EVAL_BATCH_SIZE,
								 device=device,
								 marginalize_idx=marginalize_idx)
	return mean_ll, std_ll


def generate_conditional_adv_samples(ratspn, dataset_name, ratspn_args, test_x, epsilon=0.05):
	ratspn.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_CONDITIONAL_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_ground_truth.png".format(dataset_name, ratspn_args[NUM_SUMS],
																  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																  ratspn_args[NUM_REPETITIONS])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		test_batch_x = test_x[0:25, :].clone()
		test_batch_x[:, marginalize_idx] = np.nan

		mpe_reconstruction = ratspn.mpe(x=test_batch_x).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_{}_adv_mpe_reconstruction.png".format(dataset_name,
																					 ratspn_args[NUM_SUMS],
																					 ratspn_args[
																						 NUM_INPUT_DISTRIBUTIONS],
																					 ratspn_args[
																						 NUM_REPETITIONS],
																					 epsilon)
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def train_spn(run_id, dataset_name, spn, train_x, valid_x, test_x, spn_args, train_labels=None, valid_labels=None, test_labels=None):
	spn.train()

	file_path = get_model_file_path(run_id, dataset_name, spn_args)

	data_train = TensorDataset(train_x, train_labels)
	data_valid = TensorDataset(valid_x, valid_labels)
	data_test = TensorDataset(test_x, test_labels)

	train_model(model=spn,
				data_train=data_train,
				data_valid=data_valid,
				data_test=data_test,
				setting=DISCRIMINATIVE,
				lr=spn_args[LEARNING_RATE],
				batch_size=spn_args[BATCH_SIZE],
				epochs=spn_args[NUM_EPOCHS],
				patience=spn_args[PATIENCE],
				device=torch.device(device))

	torch.save(spn.state_dict(), file_path)
	return spn


def train_adv_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x, valid_labels, test_x, test_labels,
					 ratspn_args, batch_size=100, epsilon=0.05):
	ratspn.train()

	if dataset_name in DEBD_DATASETS:
		RATSPN_MODEL_DIRECTORY = os.path.join(DEBD_MODEL_DIRECTORY,
											  dataset_name + "/{}".format(BINARY_DEBD_HAMMING_THRESHOLD))
	else:
		RATSPN_MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, dataset_name)
	mkdir_p(RATSPN_MODEL_DIRECTORY)

	file_path = os.path.join(RATSPN_MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}_adv.pt".format(dataset_name, ratspn_args[NUM_SUMS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															ratspn_args[NUM_REPETITIONS], epsilon))

	attack, DATA_DIRECTORY = None, None
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/{}/augmented/sparsefool".format(dataset_name)
	elif dataset_name in DEBD_DATASETS:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/DEBD/datasets/{}/augmented/sparsefool/{}".format(dataset_name,
																				BINARY_DEBD_HAMMING_THRESHOLD)

	train_file_path = os.path.join(DATA_DIRECTORY, "train_dataset.pt")
	valid_file_path = os.path.join(DATA_DIRECTORY, "valid_dataset.pt")

	if os.path.exists(train_file_path) and os.path.exists(valid_file_path):
		data_train = torch.load(train_file_path)
		data_valid = torch.load(valid_file_path)
	else:
		mkdir_p(DATA_DIRECTORY)
		original_N = train_x.shape[0]
		train_x, train_labels = attack.generate_adv_dataset(dataset_name, train_x, train_labels, combine=True)
		valid_x, valid_labels = attack.generate_adv_dataset(dataset_name, valid_x, valid_labels, combine=True)
		augmented_N = train_x.shape[0]

		print("Generated {} adversarial samples".format(augmented_N - original_N))

		data_train = TensorDataset(train_x)
		data_valid = TensorDataset(valid_x)

		torch.save(data_train, train_file_path)
		torch.save(data_valid, valid_file_path)

	batch_size = DEFAULT_TRAIN_BATCH_SIZE
	if ratspn_args[NUM_INPUT_DISTRIBUTIONS] == 50:
		batch_size = 50

	train_model(model=ratspn,
				data_train=data_train,
				data_valid=data_valid,
				setting=GENERATIVE,
				lr=DEFAULT_LEARNING_RATE,
				batch_size=batch_size,
				epochs=MAX_NUM_EPOCHS,
				patience=DEFAULT_PATIENCE,
				device=device)

	torch.save(ratspn.state_dict(), file_path)
	return ratspn


def test_spn(spn, test_x, spn_args, test_labels):
	spn.eval()

	data_test = TensorDataset(test_x, test_labels)
	nll, metrics = test_model(model=spn,
							  data_test=data_test,
							  setting=DISCRIMINATIVE,
							  batch_size=spn_args[BATCH_SIZE],
							  device=torch.device(device))
	return nll, metrics


def fetch_adv_test_data(ratspn, dataset_name, test_x, test_labels):
	attack, DATA_DIRECTORY = None, None
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/{}/augmented/sparsefool".format(dataset_name)
	elif dataset_name in DEBD_DATASETS:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/DEBD/datasets/{}/augmented/sparsefool/{}".format(dataset_name,
																				BINARY_DEBD_HAMMING_THRESHOLD)

	test_file_path = os.path.join(DATA_DIRECTORY, "test_dataset.pt")
	if os.path.exists(test_file_path):
		data_test = torch.load(test_file_path)
	else:
		test_x, test_labels = attack.generate_adv_dataset(dataset_name, test_x, test_labels, combine=False)
		data_test = TensorDataset(test_x)
		mkdir_p(DATA_DIRECTORY)
		torch.save(data_test, test_file_path)
	return data_test


def test_adv_spn(dataset_name, ratspn, test_x, test_labels, batch_size=100, epsilon=0.05):
	ratspn.eval()

	data_test = fetch_adv_test_data(ratspn, dataset_name, test_x, test_labels)

	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=GENERATIVE,
								 batch_size=batch_size,
								 device=device)

	return mean_ll, std_ll


def test_conditional_adv_likelihood(ratspn, dataset_name, evidence_percentage, ratspn_args, test_x, test_labels):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	ratspn.eval()
	data_test = fetch_adv_test_data(ratspn, dataset_name, test_x, test_labels)

	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=CONDITIONAL,
								 batch_size=DEFAULT_EVAL_BATCH_SIZE,
								 device=device,
								 marginalize_idx=marginalize_idx)
	return mean_ll, std_ll
