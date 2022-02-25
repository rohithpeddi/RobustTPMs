import os
import numpy as np
import torch
import datasets
import torch.optim as optim
import deeprob.spn.models as spn_models

from constants import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from deeprob.torch.routines import train_model, test_model
from train_neural_models import generate_debd_labels

from deeprob.torch.callbacks import EarlyStopping
from deeprob.torch.metrics import RunningAverageMetric
from utils import mkdir_p, save_image_stack, predict_labels_mnist

from attacks.localrestrictedsearch import attack as local_restricted_search_attack
from attacks.localsearch import attack as local_search_attack
from attacks.sparsefool import attack as sparsefool_attack
from attacks.weakermodel import attack as weaker_attack
import torchattacks

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def load_dataset(dataset_name, model_type=None):
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

		if model_type == GAUSSIAN_RATSPN:
			train_x = train_x.reshape(-1, 784)
			valid_x = valid_x.reshape(-1, 784)
			test_x = test_x.reshape(-1, 784)

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

		if model_type == GAUSSIAN_RATSPN:
			train_x = train_x.reshape(-1, 784)
			valid_x = valid_x.reshape(-1, 784)
			test_x = test_x.reshape(-1, 784)

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


def load_spn(dataset_name, spn_args):
	if dataset_name in [MNIST, FASHION_MNIST, CIFAR_10]:
		if spn_args[MODEL_TYPE] == DGCSPN:
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
		elif spn_args[MODEL_TYPE] == GAUSSIAN_RATSPN:
			ratspn = spn_models.GaussianRatSpn(
				spn_args[N_FEATURES],
				out_classes=spn_args[OUT_CLASSES],  # The number of classes
				rg_depth=spn_args[DEPTH],  # The region graph's depth
				rg_repetitions=spn_args[NUM_REPETITIONS],  # The region graph's number of repetitions
				rg_batch=spn_args[NUM_INPUT_DISTRIBUTIONS],  # The region graph's number of batched leaves
				rg_sum=spn_args[NUM_SUMS],  # The region graph's number of sum nodes per region
				in_dropout=DEFAULT_LEAF_DROPOUT,  # The probabilistic dropout rate to use at leaves layer
				sum_dropout=DEFAULT_SUM_DROPOUT  # The probabilistic dropout rate to use at sum nodes
			)
			ratspn.to(device)
			return ratspn
	elif dataset_name in DEBD_DATASETS or dataset_name in [BINARY_MNIST]:
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


def get_model_file_path(run_id, dataset_name, spn_args, train_attack_type, perturbations):
	file_path, RUN_MODEL_DIRECTORY = None, None
	if dataset_name == MNIST:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), MNIST_MODEL_DIRECTORY)
	if dataset_name == FASHION_MNIST:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), FASHION_MNIST_MODEL_DIRECTORY)
	elif dataset_name == BINARY_MNIST:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), BINARY_MNIST_MODEL_DIRECTORY)
	elif dataset_name == CIFAR_10:
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), CIFAR_10_MODEL_DIRECTORY)
	elif dataset_name in DEBD_DATASETS:
		DEBD_DATASET_MODEL_DIRECTORY = DEBD_MODEL_DIRECTORY + "/{}".format(dataset_name)
		RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), DEBD_DATASET_MODEL_DIRECTORY)
	RUN_MODEL_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, "{}".format(train_attack_type))
	mkdir_p(RUN_MODEL_DIRECTORY)

	if spn_args[MODEL_TYPE] == DGCSPN:
		file_path = os.path.join(RUN_MODEL_DIRECTORY, "dgcspn_{}_{}.pt".format(dataset_name, perturbations*255))
	elif spn_args[MODEL_TYPE] == GAUSSIAN_RATSPN:
		file_path = os.path.join(RUN_MODEL_DIRECTORY, "ratspn_{}_{}.pt".format(dataset_name, perturbations*255))

	if dataset_name in DEBD_DATASETS:
		file_path = os.path.join(RUN_MODEL_DIRECTORY, "{}_{}.pt".format(dataset_name, 0 if train_attack_type == CLEAN else perturbations))

	return file_path


def load_pretrained_spn(run_id, dataset_name, spn_args, train_attack_type, perturbations):
	trained_spn = load_spn(dataset_name, spn_args)
	file_path = get_model_file_path(run_id, dataset_name, spn_args, train_attack_type, perturbations)
	if os.path.exists(file_path):
		trained_spn.load_state_dict(torch.load(file_path))
	else:
		print("Ratspn is not stored, train first")
		return None
	trained_spn.to(device)
	return trained_spn


def generate_samples(run_id, trained_spn, dataset_name, spn_args):
	trained_spn.eval()
	DATASET_SAMPLES_DIR = os.path.join(SAMPLES_DIRECTORY, dataset_name)
	RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), DATASET_SAMPLES_DIR)
	RUN_SAMPLES_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, "{}".format(spn_args[MODEL_TYPE]))
	mkdir_p(RUN_SAMPLES_DIRECTORY)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		samples = trained_spn.sample(n_samples=25).cpu().numpy()
		samples = samples.reshape((-1, 28, 28))
		file_name = "{}_{}_{}_{}_{}_sample.png".format(spn_args[ATTACK_TYPE], dataset_name, spn_args[NUM_SUMS],
													   spn_args[NUM_INPUT_DISTRIBUTIONS],
													   spn_args[NUM_REPETITIONS])
		save_image_stack(samples, 5, 5, os.path.join(RUN_SAMPLES_DIRECTORY, file_name), margin_gray_val=0.)


def generate_conditional_samples(run_id, ratspn, dataset_name, spn_args, test_x):
	ratspn.eval()
	DATASET_CONDITIONAL_SAMPLES_DIR = os.path.join(CONDITIONAL_SAMPLES_DIRECTORY, dataset_name)
	RUN_MODEL_DIRECTORY = os.path.join("run_{}".format(run_id), DATASET_CONDITIONAL_SAMPLES_DIR)
	RUN_CONDITIONAL_SAMPLES_DIRECTORY = os.path.join(RUN_MODEL_DIRECTORY, "{}".format(spn_args[MODEL_TYPE]))
	mkdir_p(RUN_CONDITIONAL_SAMPLES_DIRECTORY)

	if dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT / 2), :].reshape(-1))

		# ground truth
		ground_truth = test_x[0:25, :].cpu().numpy()
		ground_truth = ground_truth.reshape((-1, 28, 28))
		ground_truth_file = "{}_{}_{}_{}_ground_truth.png".format(dataset_name, spn_args[NUM_SUMS],
																  spn_args[NUM_INPUT_DISTRIBUTIONS],
																  spn_args[NUM_REPETITIONS])
		save_image_stack(ground_truth, 5, 5, os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, ground_truth_file),
						 margin_gray_val=0.)

		test_batch_x = (test_x[0:25, :]).clone()
		test_batch_x[:, marginalize_idx] = np.nan

		mpe_reconstruction = ratspn.mpe(x=test_batch_x).cpu().numpy()
		mpe_reconstruction = mpe_reconstruction.squeeze()
		mpe_reconstruction = mpe_reconstruction.reshape((-1, 28, 28))
		mpe_reconstruction_file = "{}_{}_{}_{}_{}_mpe_reconstruction.png".format(spn_args[ATTACK_TYPE], dataset_name,
																				 spn_args[NUM_SUMS],
																				 spn_args[NUM_INPUT_DISTRIBUTIONS],
																				 spn_args[NUM_REPETITIONS])
		save_image_stack(mpe_reconstruction, 5, 5,
						 os.path.join(DATASET_CONDITIONAL_SAMPLES_DIR, mpe_reconstruction_file), margin_gray_val=0.)


def fetch_adv_data_batched(attack, data, attack_type, labels):
	test_loader = DataLoader(TensorDataset(data, labels), shuffle=True, batch_size=100)
	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Generating adversarial samples using {}'.format(attack_type), unit='batch'
	)

	adv_data = []
	for inputs, labels in data_loader:
		inputs = (inputs.to(device)).clone()
		adv_inputs = attack(inputs, labels)
		adv_data.append(adv_inputs)

	return torch.cat(adv_data)


def fetch_adv_data(trained_spn, dataset_name, train_x, inputs, labels, perturbations, attack_type, combine=False):
	if attack_type == FGSM:
		attack = torchattacks.FGSM(trained_spn, perturbations)
		adv_inputs = fetch_adv_data_batched(attack, inputs, attack_type, labels)
	elif attack_type == PGD:
		attack = torchattacks.PGD(trained_spn, eps=perturbations, alpha=1 / 255, steps=40, random_start=True)
		adv_inputs = fetch_adv_data_batched(attack, inputs, attack_type, labels)
	elif attack_type == SPARSEFOOL:
		attack = torchattacks.SparseFool(trained_spn, steps=20, lam=3, overshoot=0.02)
		adv_inputs = fetch_adv_data_batched(attack, inputs, attack_type, labels)
	elif attack_type in [RESTRICTED_LOCAL_SEARCH, LOCAL_SEARCH, WEAKER_MODEL]:
		attack = None
		if attack_type == RESTRICTED_LOCAL_SEARCH:
			attack = local_restricted_search_attack
		elif attack_type == LOCAL_SEARCH:
			attack = local_search_attack
		elif attack_type == WEAKER_MODEL:
			attack = weaker_attack
		adv_inputs = attack.generate_adv_dataset(trained_spn, dataset_name, inputs, labels, perturbations,
											   combine=combine, batched=True, train_data=train_x)
	if combine and attack_type in [FGSM, PGD]:
		adv_inputs = torch.cat(inputs, adv_inputs)

	# print("Difference between generated adversarial images {}, number of items changed {}".format(
	# 	torch.sum(torch.abs(adv_inputs - inputs)), torch.sum(torch.abs(adv_inputs - inputs) > 1e-5)))

	return adv_inputs


def train_generative_spn(run_id, dataset_name, spn_model, train_x, valid_x, test_x, spn_args, perturbations,
						 is_adv=False, train_labels=None, valid_labels=None, test_labels=None, train_attack_type=None):
	spn_model.train()
	file_path = get_model_file_path(run_id, dataset_name, spn_args, train_attack_type, perturbations)

	train_loader = DataLoader(TensorDataset(train_x), spn_args[BATCH_SIZE], shuffle=True)
	valid_loader = DataLoader(TensorDataset(valid_x), spn_args[BATCH_SIZE], shuffle=True)
	test_loader = DataLoader(TensorDataset(test_x), spn_args[BATCH_SIZE], shuffle=True)

	# Instantiate the optimizer
	optimizer = optim.SGD(spn_model.parameters(), lr=spn_args[LEARNING_RATE], momentum=0.9)

	print("Training with learning rate {}".format(spn_args[LEARNING_RATE]))

	# Instantiate the early stopping callback
	patience = 1 if is_adv else DEFAULT_SPN_PATIENCE
	early_stopping = EarlyStopping(spn_model, patience=patience, filepath=EARLY_STOPPING_FILE,
								   delta=EARLY_STOPPING_DELTA)

	# Instantiate the running average metrics
	running_train_loss = RunningAverageMetric()
	running_valid_loss = RunningAverageMetric()

	EPOCHS = spn_args[NUM_EPOCHS]

	for epoch in range(1, EPOCHS + 1):
		# Reset the metrics
		running_train_loss.reset()
		running_valid_loss.reset()

		if epoch % 5 == 0 and is_adv:
			adv_train_dataset = fetch_adv_data(spn_model, dataset_name, train_x, test_x, test_labels, perturbations,
											   train_attack_type, combine=False)
			adv_train_loader = DataLoader(TensorDataset(adv_train_dataset), spn_args[BATCH_SIZE], shuffle=True)
			epoch_train_loader = adv_train_loader
		else:
			epoch_train_loader = train_loader

		data_loader = tqdm(
			epoch_train_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Train Epoch {}/{}'.format(epoch, EPOCHS), unit='batch'
		)

		# Training phase
		for inputs in data_loader:
			inputs = inputs[0].to(device)
			optimizer.zero_grad()
			outputs = spn_model(inputs)
			loss = spn_model.loss(outputs)
			loss.backward()
			optimizer.step()
			spn_model.apply_constraints()
			running_train_loss(loss.item(), num_samples=inputs.shape[0])

		data_loader = tqdm(
			valid_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
			desc='Valid Epoch {}/{}'.format(epoch, EPOCHS), unit='batch'
		)

		# Make sure the model is set to evaluation mode
		spn_model.eval()

		# Validation phase
		with torch.no_grad():
			for inputs in data_loader:
				inputs = inputs[0].to(device)
				outputs = spn_model(inputs)
				loss = spn_model.loss(outputs)
				running_valid_loss(loss.item(), num_samples=inputs.shape[0])

		# Get the average train and validation losses and print it
		train_loss = running_train_loss.average()
		valid_loss = running_valid_loss.average()
		print("Epoch {}/{} - train_loss: {:.4f}, valid_loss: {:.4f}".format(epoch, EPOCHS, train_loss, valid_loss))

		# Check if training should stop according to early stopping

		# early_stopping(valid_loss, epoch)
		# if early_stopping.should_stop:
		# 	print("Early Stopping... {}".format(early_stopping))
		# 	break

	# Load the best parameters state according to early stopping
	spn_model.load_state_dict(early_stopping.get_best_state())
	torch.save(spn_model.state_dict(), file_path)
	return spn_model


def test_spn(dataset_name, trained_spn, data_spn, train_x, test_x, test_labels, perturbations, spn_args,
			 attack_type=None, is_adv=False):
	trained_spn.eval()

	if is_adv:
		test_x = fetch_adv_data(trained_spn, dataset_name, train_x, test_x, test_labels, perturbations, attack_type,
								combine=False)

	data_test = TensorDataset(test_x, test_labels)
	mean_ll, std_ll = test_model(model=trained_spn,
								 data_test=data_test,
								 setting=spn_args[TRAIN_SETTING],
								 batch_size=spn_args[BATCH_SIZE],
								 device=torch.device(device))
	return mean_ll, std_ll, test_x


def test_conditional_spn(trained_spn, dataset_name, evidence_percentage, spn_args, test_x):
	marginalize_idx = None
	if dataset_name in DEBD_DATASETS:
		test_N, num_dims = test_x.shape
		marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
	elif dataset_name == MNIST or dataset_name == BINARY_MNIST:
		image_scope = np.array(range(MNIST_HEIGHT * MNIST_WIDTH)).reshape(MNIST_HEIGHT, MNIST_WIDTH)
		marginalize_idx = list(image_scope[0:round(MNIST_HEIGHT * (1 - evidence_percentage)), :].reshape(-1))

	trained_spn.eval()
	data_test = TensorDataset(test_x)

	mean_ll, std_ll = test_model(model=trained_spn,
								 data_test=data_test,
								 setting=CONDITIONAL,
								 batch_size=DEFAULT_EVAL_BATCH_SIZE,
								 device=torch.device(device),
								 marginalize_idx=marginalize_idx)
	return mean_ll, std_ll
