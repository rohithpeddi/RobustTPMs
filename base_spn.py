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
from utils import mkdir_p, save_image_stack
from constants import *

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def load_debd_dataset(dataset_name):
	train_x, test_x, valid_x = datasets.load_debd(dataset_name)

	train_x = torch.from_numpy(train_x).to(torch.device(device))
	valid_x = torch.from_numpy(valid_x).to(torch.device(device))
	test_x = torch.from_numpy(test_x).to(torch.device(device))

	return train_x, valid_x, test_x


def load_dataset(dataset_name):
	# TODO: Binary MNIST should not be here
	if dataset_name == BINARY_MNIST:
		return datasets.load_binarized_mnist_dataset()

	train_x, train_labels, test_x, test_labels = None, None, None, None
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
	elif dataset_name in DEBD_DATASETS:
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


def load_pretrained_ratspn(dataset_name, ratspn_args):
	ratspn = load_ratspn(dataset_name, ratspn_args)

	file_name = None
	if dataset_name == MNIST:
		mkdir_p(MODEL_DIRECTORY)
		file_name = os.path.join(MODEL_DIRECTORY,
								 "{}_{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															ratspn_args[NUM_REPETITIONS]))
	elif dataset_name in DEBD_DATASETS:
		mkdir_p(DEBD_MODEL_DIRECTORY)
		file_name = os.path.join(DEBD_MODEL_DIRECTORY,
								 "{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
														 ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														 ratspn_args[NUM_REPETITIONS]))

	if os.path.exists(file_name):
		ratspn.load_state_dict(torch.load(file_name))
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
		keep_idx = [i for i in range(MNIST_WIDTH * MNIST_HEIGHT) if i not in marginalize_idx]

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_ground_truth.png".format(dataset_name, ratspn_args[NUM_SUMS],
																  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																  ratspn_args[NUM_REPETITIONS])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		test_batch_x = test_x[0:25, :]
		test_batch_x[:, marginalize_idx] = np.nan

		mpe_reconstruction = ratspn.mpe(x=test_batch_x).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_mpe_reconstruction.png".format(dataset_name, ratspn_args[NUM_SUMS],
																			  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																			  ratspn_args[NUM_REPETITIONS])
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def evaluate_conditional_likelihood(ratspn, dataset_name, ratspn_args, test_x):
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = random.sample(range(num_dims), int(0.5 * num_dims))

		conditional_test_x = test_x.clone()
		conditional_test_x[:, marginalize_idx] = np.nan



def generate_conditional_adv_samples(ratspn, dataset_name, ratspn_args, test_x, epsilon=0.05):
	ratspn.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	mkdir_p(DATASET_CONDITIONAL_SAMPLES_DIR)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))
		keep_idx = [i for i in range(MNIST_WIDTH * MNIST_HEIGHT) if i not in marginalize_idx]

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_ground_truth.png".format(dataset_name, ratspn_args[NUM_SUMS],
																  ratspn_args[NUM_INPUT_DISTRIBUTIONS],
																  ratspn_args[NUM_REPETITIONS])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		test_batch_x = test_x[0:25, :]
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


def train_clean_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x, valid_labels, test_x, test_labels,
					   ratspn_args, batch_size=100):
	ratspn.train()
	mkdir_p(MODEL_DIRECTORY)
	file_path = os.path.join(MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}.pt".format(dataset_name, ratspn_args[NUM_SUMS],
														ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														ratspn_args[NUM_INPUT_DISTRIBUTIONS],
														DEFAULT_NUM_REPETITIONS))
	data_train = TensorDataset(train_x, train_labels)
	data_valid = TensorDataset(valid_x, valid_labels)

	train_model(model=ratspn,
				data_train=data_train,
				data_valid=data_valid,
				setting=GENERATIVE,
				lr=DEFAULT_LEARNING_RATE,
				batch_size=DEFAULT_TRAIN_BATCH_SIZE,
				epochs=MAX_NUM_EPOCHS,
				patience=DEFAULT_PATIENCE,
				device=device)

	torch.save(ratspn.state_dict(), file_path)
	return ratspn


def train_adv_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x, valid_labels, test_x, test_labels,
					 ratspn_args, batch_size=100, epsilon=0.05):
	ratspn.train()
	mkdir_p(MODEL_DIRECTORY)
	file_path = os.path.join(MODEL_DIRECTORY,
							 "{}_{}_{}_{}_{}_adv.pt".format(dataset_name, ratspn_args[NUM_SUMS],
															ratspn_args[NUM_INPUT_DISTRIBUTIONS],
															DEFAULT_NUM_REPETITIONS, epsilon))
	attack = None
	if dataset_name == MNIST:
		attack = sparsefool_attack

	train_file_path = os.path.join(DATA_MNIST_ADV_SPARSEFOOL, "train_dataset.pt")
	valid_file_path = os.path.join(DATA_MNIST_ADV_SPARSEFOOL, "valid_dataset.pt")

	if os.path.exists(train_file_path) and os.path.exists(valid_file_path):
		data_train = torch.load(train_file_path)
		data_valid = torch.load(valid_file_path)
	else:
		mkdir_p(DATA_MNIST_ADV_SPARSEFOOL)
		train_x, train_labels = attack.generate_adv_dataset(train_x, train_labels, dataset_name, combine=True)
		valid_x, valid_labels = attack.generate_adv_dataset(valid_x, valid_labels, dataset_name, combine=True)

		data_train = TensorDataset(train_x, train_labels)
		data_valid = TensorDataset(valid_x, valid_labels)

		torch.save(data_train, train_file_path)
		torch.save(data_valid, valid_file_path)

	train_model(model=ratspn,
				data_train=data_train,
				data_valid=data_valid,
				setting=GENERATIVE,
				lr=DEFAULT_LEARNING_RATE,
				batch_size=DEFAULT_TRAIN_BATCH_SIZE,
				epochs=MAX_NUM_EPOCHS,
				patience=DEFAULT_PATIENCE,
				device=device)

	torch.save(ratspn.state_dict(), file_path)
	return ratspn


def test_clean_spn(ratspn, test_x, test_labels, batch_size=100):
	ratspn.eval()
	data_test = TensorDataset(test_x, test_labels)

	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=GENERATIVE,
								 batch_size=batch_size,
								 device=device)

	return mean_ll, std_ll


def test_adv_spn(ratspn, dataset_name, test_x, test_labels, batch_size=100, epsilon=0.05):
	ratspn.eval()

	attack = None
	if dataset_name == MNIST:
		attack = sparsefool_attack

	test_file_path = os.path.join(DATA_MNIST_ADV_SPARSEFOOL, "test_dataset.pt")
	if os.path.exists(test_file_path):
		data_test = torch.load(test_file_path)
	else:
		test_x, test_labels = attack.generate_adv_dataset(test_x, test_labels, dataset_name, combine=False)
		data_test = TensorDataset(test_x, test_labels)
		mkdir_p(DATA_MNIST_ADV_SPARSEFOOL)
		torch.save(data_test, test_file_path)

	mean_ll, std_ll = test_model(model=ratspn,
								 data_test=data_test,
								 setting=GENERATIVE,
								 batch_size=batch_size,
								 device=device)

	return mean_ll, std_ll
