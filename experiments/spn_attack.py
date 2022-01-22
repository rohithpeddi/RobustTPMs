import json
import os

import numpy as np
import sklearn.metrics
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn import metrics

import deeprob.spn.models as spn
import utils
from constants import *
from deeprob.torch.metrics import RunningAverageMetric
from deeprob.torch.routines import train_model
from deeprob.torch.transforms import Flatten

import torchattacks

"""
1. Load the dataset
2. Train a RATSPN discriminative on MNIST
3. Take gradients and create new test points like FGSM attack 
4. Find the test accuracy
5. Use the FGSM attack in training
"""

n_features = 784
n_classes = 10
# Set the preprocessing transformation
transform_spn = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,)),
	Flatten()
])

transform_nn = transforms.Compose([
	transforms.ToTensor(),
])

# Setup the datasets
data_train_spn = datasets.MNIST(DATA_DIRECTORY, train=True, transform=transform_spn, download=True)
data_test_spn = datasets.MNIST(DATA_DIRECTORY, train=False, transform=transform_spn, download=True)
n_val = int(0.1 * len(data_train_spn))
n_train = len(data_train_spn) - n_val
data_train_spn, data_val_spn = data.random_split(data_train_spn, [n_train, n_val])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_rat_SPN(data_train, data_val, n_features, n_classes):
	# Instantiate a RAT-SPN model with Gaussian leaves
	ratspn = spn.GaussianRatSpn(
		n_features,
		out_classes=n_classes,  # The number of classes
		rg_depth=3,  # The region graph's depth
		rg_repetitions=8,  # The region graph's number of repetitions
		rg_batch=16,  # The region graph's number of batched leaves
		rg_sum=16,  # The region graph's number of sum nodes per region
		in_dropout=0.2,  # The probabilistic dropout rate to use at leaves layer
		sum_dropout=0.2  # The probabilistic dropout rate to use at sum nodes
	)

	# Train the model using discriminative setting, i.e. by minimizing the categorical cross-entropy
	train_model(
		ratspn, data_train, data_val, setting='discriminative',
		lr=1e-2, batch_size=100, epochs=100, patience=5, checkpoint='checkpoint-ratspn-mnist.pt', device=device
	)
	utils.mkdir_p(EXPERIMENTS_DIRECTORY)
	torch.save(ratspn.state_dict(), os.path.join(EXPERIMENTS_DIRECTORY, 'ratspn-mnist.pt'))
	return ratspn


def fgsm_attack(image, epsilon, data_grad):
	# Collect the element-wise sign of the data gradient
	sign_data_grad = data_grad.sign()
	# Create the perturbed image by adjusting each pixel of the input image
	perturbed_image = image + epsilon * sign_data_grad
	# Adding clipping to maintain [0,1] range
	perturbed_image = torch.clamp(perturbed_image, 0, 1)
	# Return the perturbed image
	return perturbed_image


def test_rat_SPN_fgsm(ratspn, data_test):
	# Test the model, plotting the test negative log-likelihood and some classification metrics

	test_loader = data.DataLoader(data_test, batch_size=DEFAULT_EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
								  num_workers=1)

	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Test', unit='batch'
	)

	# Make sure the model is set to evaluation mode
	ratspn.eval()

	y_true = []
	y_pred = []
	running_loss = RunningAverageMetric()
	with torch.no_grad():
		for inputs, targets in data_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = ratspn(inputs)
			loss = ratspn.loss(outputs, y=targets)
			running_loss(loss.item(), num_samples=inputs.shape[0])
			predictions = torch.argmax(outputs, dim=1)
			y_pred.extend(predictions.cpu().tolist())
			y_true.extend(targets.cpu().tolist())

	y_pred = np.array(y_pred)
	y_true = np.array(y_true)

	N = y_pred.shape[0]
	accuracy = (y_pred == y_true).sum() / N

	nll = running_loss.average()
	print('Clean Test NLL: {:.4f}'.format(nll))
	print('Accuracy {}'.format(accuracy))

	for epsilon in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8]:

		y_true = []
		y_pred = []

		with torch.no_grad():
			for inputs, targets in data_loader:
				with torch.enable_grad():
					inputs, targets = inputs.to(device), targets.to(device)
					inputs.requires_grad = True
					outputs = ratspn(inputs)
					loss = ratspn.loss(outputs, y=targets)

					ratspn.zero_grad()
					loss.backward()

					data_grad = inputs.grad.data

				perturbed_inputs = fgsm_attack(inputs, epsilon=epsilon, data_grad=data_grad)

				outputs = ratspn(perturbed_inputs)
				loss = ratspn.loss(outputs, y=targets)
				running_loss(loss.item(), num_samples=inputs.shape[0])
				predictions = torch.argmax(outputs, dim=1)
				y_pred.extend(predictions.cpu().tolist())
				y_true.extend(targets.cpu().tolist())

		y_pred = np.array(y_pred)
		y_true = np.array(y_true)

		N = y_pred.shape[0]
		accuracy = (y_pred == y_true).sum() / N

		nll = running_loss.average()
		print('Clean Test NLL: {:.4f}'.format(nll))
		print('Epsilon {}, Accuracy {}'.format(epsilon, accuracy))


def test_rat_SPN(ratspn, data_test):
	# Test the model, plotting the test negative log-likelihood and some classification metrics
	test_loader = data.DataLoader(data_test, batch_size=DEFAULT_EVAL_BATCH_SIZE, shuffle=False, drop_last=False,
								  num_workers=1)

	y_true = []
	y_pred = []
	running_loss = RunningAverageMetric()
	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Test', unit='batch'
	)

	with torch.no_grad():
		for inputs, targets in data_loader:
			inputs, targets = inputs.to(device), targets.to(device)

			with torch.enable_grad():
				atk = torchattacks.PGD(ratspn, eps=8 / 255, alpha=2 / 255, steps=4)
				perturbed_inputs = atk(inputs, targets)

			outputs = ratspn(perturbed_inputs)
			loss = ratspn.loss(outputs, y=targets)
			running_loss(loss.item(), num_samples=inputs.shape[0])
			predictions = torch.argmax(outputs, dim=1)
			y_pred.extend(predictions.cpu().tolist())
			y_true.extend(targets.cpu().tolist())

	y_pred = np.array(y_pred)
	y_true = np.array(y_true)

	N = y_pred.shape[0]
	accuracy = (y_pred == y_true).sum() / N

	nll = running_loss.average()
	print('Clean Test NLL: {:.4f}'.format(nll))
	print('Accuracy {}'.format(accuracy))


def main():
	# train_rat_SPN(data_train_spn, data_val_spn, n_features, n_classes)

	ratspn = spn.GaussianRatSpn(
		n_features,
		out_classes=n_classes,  # The number of classes
		rg_depth=3,  # The region graph's depth
		rg_repetitions=8,  # The region graph's number of repetitions
		rg_batch=16,  # The region graph's number of batched leaves
		rg_sum=16,  # The region graph's number of sum nodes per region
		in_dropout=0.2,  # The probabilistic dropout rate to use at leaves layer
		sum_dropout=0.2  # The probabilistic dropout rate to use at sum nodes
	)
	utils.mkdir_p(EXPERIMENTS_DIRECTORY)
	ratspn.load_state_dict(torch.load(os.path.join(EXPERIMENTS_DIRECTORY, 'ratspn-mnist.pt')))
	ratspn.to(device)

	test_rat_SPN_fgsm(ratspn, data_test_spn)
	# test_rat_SPN(ratspn, data_test_spn)


if __name__ == '__main__':
	main()
