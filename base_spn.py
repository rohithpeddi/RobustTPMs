import torch
import os
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

import datasets
import deeprob.spn.models as spn
from attacks.fgsm import attack as fgsm_attack
from attacks.sparsefool import attack as sparsefool_attack
from deeprob.torch.routines import train_model, test_model
from utils import mkdir_p, save_image_stack, predict_labels_mnist
from constants import *

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def load_debd_dataset(dataset_name):
	train_x, test_x, valid_x = datasets.load_debd(dataset_name)

	train_x = torch.tensor(train_x, dtype=torch.float32, device=torch.device(device))
	valid_x = torch.tensor(valid_x, dtype=torch.float32, device=torch.device(device))
	test_x = torch.tensor(test_x, dtype=torch.float32, device=torch.device(device))

	return train_x, valid_x, test_x


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

	return train_x, valid_x, test_x, train_labels, valid_labels, test_labels


def load_ratspn(dataset_name, ratspn_args):
	ratspn = None
	if dataset_name == MNIST:
		ratspn = spn.GaussianRatSpn(
			ratspn_args[N_FEATURES],
			out_classes=ratspn_args[OUT_CLASSES],  # The number of classes
			rg_depth=ratspn_args[DEPTH],  # The region graph's depth
			rg_repetitions=ratspn_args[NUM_REPETITIONS],  # The region graph's number of repetitions
			rg_batch=ratspn_args[NUM_INPUT_DISTRIBUTIONS],  # The region graph's number of batched leaves
			rg_sum=ratspn_args[NUM_SUMS],  # The region graph's number of sum nodes per region
			in_dropout=DEFAULT_LEAF_DROPOUT,  # The probabilistic dropout rate to use at leaves layer
			sum_dropout=DEFAULT_SUM_DROPOUT  # The probabilistic dropout rate to use at sum nodes
		)
	elif dataset_name in DEBD_DATASETS or dataset_name == BINARY_MNIST:
		ratspn = spn.BernoulliRatSpn(
			in_features=ratspn_args[N_FEATURES],
			out_classes=ratspn_args[OUT_CLASSES],
			rg_depth=ratspn_args[DEPTH],  # The region graph's depth
			rg_repetitions=ratspn_args[NUM_REPETITIONS],  # The region graph's number of repetitions
			rg_batch=ratspn_args[NUM_INPUT_DISTRIBUTIONS],  # The region graph's number of batched leaves
			rg_sum=ratspn_args[NUM_SUMS],  # The region graph's number of sum nodes per region
			in_dropout=DEFAULT_LEAF_DROPOUT,  # The probabilistic dropout rate to use at leaves layer
			sum_dropout=DEFAULT_SUM_DROPOUT
		)
	ratspn.to(device)
	return ratspn


def get_model_file_path(dataset_name, ratspn_args):
	file_path = None
	if dataset_name == MNIST:
		mkdir_p(MNIST_MODEL_DIRECTORY)
		file_path = os.path.join(MNIST_MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															ratspn_args[NUM_REPETITIONS]))
	elif dataset_name in DEBD_DATASETS:
		mkdir_p(DEBD_MODEL_DIRECTORY)
		file_path = os.path.join(DEBD_MODEL_DIRECTORY,
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


def load_pretrained_ratspn(dataset_name, ratspn_args):
	ratspn = load_ratspn(dataset_name, ratspn_args)
	file_path = get_model_file_path(dataset_name, ratspn_args)
	if os.path.exists(file_path):
		ratspn.load_state_dict(torch.load(file_path))
	else:
		AssertionError("Ratspn is not stored, train first")
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


def train_clean_ratspn(dataset_name, ratspn, train_x, valid_x, test_x, ratspn_args, train_labels=None,
					   valid_labels=None, test_labels=None, batch_size=100):
	ratspn.train()

	file_path = get_model_file_path(dataset_name, ratspn_args)
	data_train = TensorDataset(train_x)
	data_valid = TensorDataset(valid_x)

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


def train_adv_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x, valid_labels, test_x, test_labels,
					 ratspn_args, batch_size=100, epsilon=0.05):
	ratspn.train()
	RATSPN_MODEL_DIRECTORY = os.path.join(MODEL_DIRECTORY, dataset_name)
	mkdir_p(RATSPN_MODEL_DIRECTORY)

	file_path = os.path.join(RATSPN_MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}_adv.pt".format(dataset_name, ratspn_args[NUM_SUMS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															DEFAULT_NUM_REPETITIONS, epsilon))

	attack, DATA_DIRECTORY = None, None
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/{}/augmented/sparsefool".format(dataset_name)

	train_file_path = os.path.join(DATA_DIRECTORY, "train_dataset.pt")
	valid_file_path = os.path.join(DATA_DIRECTORY, "valid_dataset.pt")

	if os.path.exists(train_file_path) and os.path.exists(valid_file_path):
		data_train = torch.load(train_file_path)
		data_valid = torch.load(valid_file_path)
	else:
		mkdir_p(DATA_DIRECTORY)
		train_x, train_labels = attack.generate_adv_dataset(dataset_name, train_x, train_labels, combine=True)
		valid_x, valid_labels = attack.generate_adv_dataset(dataset_name, valid_x, valid_labels, combine=True)

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


def test_clean_spn(dataset_name, ratspn, test_x, test_labels=None, batch_size=100):
	ratspn.eval()
	data_test = TensorDataset(test_x)
	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=GENERATIVE,
								 batch_size=batch_size,
								 device=device)
	return mean_ll, std_ll


def fetch_adv_test_data(ratspn, dataset_name, test_x, test_labels):
	attack, DATA_DIRECTORY = None, None
	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		attack = sparsefool_attack
		DATA_DIRECTORY = "data/{}/augmented/sparsefool".format(dataset_name)

	test_file_path = os.path.join(DATA_DIRECTORY, "test_dataset.pt")
	if os.path.exists(test_file_path):
		data_test = torch.load(test_file_path)
	else:
		test_x, test_labels = attack.generate_adv_dataset(dataset_name, test_x, test_labels, combine=False)
		data_test = TensorDataset(test_x)
		mkdir_p(DATA_DIRECTORY)
		torch.save(data_test, test_file_path)
	return data_test


def test_adv_spn(ratspn, dataset_name, test_x, test_labels, batch_size=100, epsilon=0.05):
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
