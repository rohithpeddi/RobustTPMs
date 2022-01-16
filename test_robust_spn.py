import os
from typing import List

import torch

import base_spn as SPN
from constants import *
from utils import mkdir_p
from attacks.fgsm import attack

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def test_adv_spn_continuous():
	for dataset_name in CONTINUOUS_DATASETS:

		print_message = dataset_name
		evaluation_message(print_message)

		# Load data as tensors
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		if dataset_name == MNIST:

			attack_type = "sparsefool"
			if attack_type == "sparsefool":
				EPSILON_LIST = [0]

			for epsilon in EPSILON_LIST:

				for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
					print_message += (" " + "number of distributions {}".format(num_distributions))
					evaluation_message(print_message)

					ratspn_args = dict()
					ratspn_args[N_FEATURES] = MNIST_HEIGHT * MNIST_WIDTH
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

					trained_ratspn = SPN.train_adv_ratspn(dataset_name, ratspn, train_x, train_labels, valid_x,
														  valid_labels, test_x, test_labels,
														  ratspn_args, batch_size=TRAIN_BATCH_SIZE, epsilon=epsilon)

					mean_ll, std_ll = SPN.test_clean_spn(trained_ratspn, test_x, test_labels,
														 batch_size=EVAL_BATCH_SIZE)
					print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}".format(mean_ll, std_ll))

					print_message += (" " + "generating samples")
					evaluation_message(print_message)

					SPN.generate_adv_samples(trained_ratspn, dataset_name, ratspn_args, epsilon=epsilon)
					SPN.generate_conditional_adv_samples(trained_ratspn, dataset_name, ratspn_args, test_x,
														 epsilon=epsilon)

					mean_ll, std_ll = SPN.test_adv_spn(trained_ratspn, dataset_name, test_x, test_labels,
													   batch_size=EVAL_BATCH_SIZE, epsilon=epsilon)
					print("Adv Test - Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))


# def test_standard_spn_adv_test_data():
# 	for dataset_name in CONTINUOUS_DATASETS:
#
# 		print_message = dataset_name
# 		evaluation_message(print_message)
#
# 		# Load data as tensors
# 		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)
#
# 		if dataset_name == MNIST:
#
# 			test_x, test_labels = attack.generate_adv_dataset(test_x, test_labels, dataset_name)
#
# 			for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
# 				print_message += (" " + "number of distributions {}".format(num_distributions))
# 				evaluation_message(print_message)
#
# 				ratspn_args = dict()
# 				ratspn_args[N_FEATURES] = MNIST_HEIGHT * MNIST_WIDTH
# 				ratspn_args[OUT_CLASSES] = NUM_CLASSES
# 				ratspn_args[DEPTH] = DEFAULT_DEPTH
# 				ratspn_args[NUM_SUMS] = num_distributions
# 				ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
# 				ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
#
# 				print_message += (" " + "loading pretrained ratspn")
# 				evaluation_message(print_message)
#
# 				trained_ratspn = SPN.load_pretrained_ratspn(dataset_name, ratspn_args)
#
# 				mean_ll, std_ll = SPN.test_clean_spn(trained_ratspn, test_x, test_labels, batch_size=EVAL_BATCH_SIZE)
# 				print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}".format(mean_ll, std_ll))


if __name__ == '__main__':
	test_adv_spn_continuous()
