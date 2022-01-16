import os

import torch

import base_spn as SPN
from utils import mkdir_p
from attacks.fgsm import attack
from constants import *
from attacks.sparsefool import attack as sparsefool_attack

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def test_standard_spn_continuous():
	for dataset_name in CONTINUOUS_DATASETS:

		print_message = dataset_name
		evaluation_message(print_message)

		# Load data as tensors
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		if dataset_name == MNIST:

			for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
				print_message += (" " + "number of distributions {}".format(num_distributions))
				evaluation_message(print_message)

				ratspn_args = dict()
				ratspn_args[N_FEATURES] = MNIST_HEIGHT*MNIST_WIDTH
				ratspn_args[OUT_CLASSES] = NUM_CLASSES
				ratspn_args[DEPTH] = DEFAULT_DEPTH
				ratspn_args[NUM_SUMS] = num_distributions
				ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
				ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

				print_message += (" " + "loading ratspn")
				evaluation_message(print_message)

				ratspn = SPN.load_ratspn(dataset_name, ratspn_args)

				print_message += (" " + "training einet")
				evaluation_message(print_message)

				trained_ratspn = SPN.train_clean_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x,
														valid_labels, test_x, test_labels,
														ratspn_args, batch_size=TRAIN_BATCH_SIZE)

				mean_ll, std_ll = SPN.test_clean_spn(trained_ratspn, test_x, test_labels, batch_size=EVAL_BATCH_SIZE)
				print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}". format(mean_ll, std_ll))

				print_message += (" " + "generating samples")
				evaluation_message(print_message)

				SPN.generate_samples(trained_ratspn, dataset_name, ratspn_args)

				SPN.generate_conditional_samples(trained_ratspn, dataset_name, ratspn_args, test_x)


def test_standard_spn_adv_test_data():
	for dataset_name in CONTINUOUS_DATASETS:

		print_message = dataset_name
		evaluation_message(print_message)

		# Load data as tensors
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		if dataset_name == MNIST:

			# test_x, test_labels = attack.generate_adv_dataset(test_x, test_labels, dataset_name)
			test_x, test_labels = sparsefool_attack.generate_adv_dataset(test_x, test_labels, dataset_name)

			for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
				print_message += (" " + "number of distributions {}".format(num_distributions))
				evaluation_message(print_message)

				ratspn_args = dict()
				ratspn_args[N_FEATURES] = MNIST_HEIGHT*MNIST_WIDTH
				ratspn_args[OUT_CLASSES] = NUM_CLASSES
				ratspn_args[DEPTH] = DEFAULT_DEPTH
				ratspn_args[NUM_SUMS] = num_distributions
				ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
				ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

				print_message += (" " + "loading pretrained ratspn")
				evaluation_message(print_message)

				trained_ratspn = SPN.load_pretrained_ratspn(dataset_name, ratspn_args)

				mean_ll, std_ll = SPN.test_clean_spn(trained_ratspn, test_x, test_labels, batch_size=EVAL_BATCH_SIZE)
				print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}". format(mean_ll, std_ll))


if __name__ == '__main__':
	test_standard_spn_adv_test_data()
