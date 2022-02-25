import os

import torch

import base_spn as SPN
from utils import mkdir_p
from attacks.fgsm import attack
from constants import *
from attacks.sparsefool import attack as sparsefool_attack
from utils import pretty_print_dictionary, dictionary_to_file

############################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################


def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("#  " + message)
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
				ratspn_args[N_FEATURES] = MNIST_HEIGHT * MNIST_WIDTH
				ratspn_args[OUT_CLASSES] = DEFAULT_GENERATIVE_NUM_CLASSES
				ratspn_args[DEPTH] = DEFAULT_DEPTH
				ratspn_args[NUM_SUMS] = num_distributions
				ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
				ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

				print_message += (" " + "loading ratspn")
				evaluation_message(print_message)

				ratspn = SPN.load_spn(dataset_name, ratspn_args)

				print_message += (" " + "training einet")
				evaluation_message(print_message)

				trained_ratspn = SPN.train_generative_spn(1, dataset_name, ratspn, train_x, train_labels, valid_x, valid_labels,
														  test_x, test_labels, ratspn_args)

				mean_ll, std_ll = SPN.test_spn(trained_ratspn, test_x, test_labels, batch_size=EVAL_BATCH_SIZE)
				print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}".format(mean_ll, std_ll))

				print_message += (" " + "generating samples")
				evaluation_message(print_message)

				SPN.generate_samples(trained_ratspn, dataset_name, ratspn_args)

				SPN.generate_conditional_samples(132, trained_ratspn, dataset_name, ratspn_args, test_x)


def test_standard_spn_adv_test_data():
	for dataset_name in CONTINUOUS_DATASETS:

		print_message = dataset_name
		evaluation_message(print_message)

		# Load data as tensors
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		if dataset_name == MNIST:

			# test_x, test_labels = attack.generate_adv_dataset(test_x, test_labels, dataset_name)
			test_x, test_labels = sparsefool_attack.generate_adv_dataset(dataset_name, test_x, test_labels)

			for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
				print_message += (" " + "number of distributions {}".format(num_distributions))
				evaluation_message(print_message)

				ratspn_args = dict()
				ratspn_args[N_FEATURES] = MNIST_HEIGHT * MNIST_WIDTH
				ratspn_args[OUT_CLASSES] = DEFAULT_GENERATIVE_NUM_CLASSES
				ratspn_args[DEPTH] = DEFAULT_DEPTH
				ratspn_args[NUM_SUMS] = num_distributions
				ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
				ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

				print_message += (" " + "loading pretrained ratspn")
				evaluation_message(print_message)

				trained_ratspn = SPN.load_pretrained_spn(1, dataset_name, ratspn_args,, perturbations

				mean_ll, std_ll = SPN.test_spn(trained_ratspn, test_x, test_labels, batch_size=EVAL_BATCH_SIZE)
				print("Mean LogLikelihood : {}, Standard deviation of log-likelihood : {}".format(mean_ll, std_ll))


def test_standard_spn_discrete(specific_datasets=None):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	results = dict()
	for dataset_name in specific_datasets:
		evaluation_message("Dataset : {}".format(dataset_name))

		dataset_results = dict()

		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:

			dataset_distribution_results = dict()

			evaluation_message("Number of distributions {}".format(num_distributions))

			ratspn_args = dict()
			ratspn_args[N_FEATURES] = train_x.shape[1]
			ratspn_args[OUT_CLASSES] = DEFAULT_GENERATIVE_NUM_CLASSES
			ratspn_args[DEPTH] = DEFAULT_DEPTH
			ratspn_args[NUM_SUMS] = num_distributions
			ratspn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
			ratspn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

			evaluation_message("Loading ratspn")

			ratspn = SPN.load_spn(dataset_name, ratspn_args)

			evaluation_message("Training ratspn")

			trained_ratspn = SPN.train_generative_spn(1, dataset_name, ratspn, train_x, valid_x, test_x, ratspn_args)

			mean_ll, std_ll = SPN.test_spn(dataset_name, trained_ratspn, test_x, batch_size=EVAL_BATCH_SIZE)
			evaluation_message("Clean Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Clean Mean LL'] = mean_ll
			dataset_distribution_results['Clean Std LL'] = std_ll

			mean_ll, std_ll = SPN.test_adv_spn(dataset_name, trained_ratspn, test_x, test_labels,
											   batch_size=EVAL_BATCH_SIZE, epsilon=0)
			evaluation_message("Adv Test - Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Adv Mean LL'] = mean_ll
			dataset_distribution_results['Adv Std LL'] = std_ll

			if dataset_name == BINARY_MNIST:
				evaluation_message("Generating samples")
				SPN.generate_samples(trained_ratspn, dataset_name, ratspn_args)

				evaluation_message("Generating conditional samples")
				SPN.generate_conditional_samples(132, trained_ratspn, dataset_name, ratspn_args, test_x)

			for evidence_percentage in EVIDENCE_PERCENTAGES:
				dataset_distribution_evidence_results = dict()
				mean_ll, std_ll = SPN.test_conditional_spn(trained_ratspn, dataset_name, evidence_percentage,
														   ratspn_args, test_x)
				evaluation_message(
					"Clean Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage, mean_ll,
																						std_ll))
				dataset_distribution_evidence_results['Clean Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Clean Std LL'] = std_ll

				mean_ll, std_ll = SPN.test_conditional_adv_likelihood(trained_ratspn, dataset_name, evidence_percentage,
																	  ratspn_args, test_x, test_labels)
				evaluation_message(
					"Adv Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage, mean_ll,
																					  std_ll))
				dataset_distribution_evidence_results['Adv Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Adv Std LL'] = std_ll

				dataset_distribution_results[evidence_percentage] = dataset_distribution_evidence_results
			dataset_results[num_distributions] = dataset_distribution_results
		results[dataset_name] = dataset_results
		dictionary_to_file(dataset_name, dataset_results, 1, is_adv=False)
		pretty_print_dictionary(dataset_results)
	pretty_print_dictionary(results)


if __name__ == '__main__':
	test_standard_spn_discrete(DEBD_DATASETS)
	# test_standard_spn_discrete(['plants'])
