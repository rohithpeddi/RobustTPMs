import os
import torch
import einet_base_spn as SPN
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray

from utils import mkdir_p, pretty_print_dictionary, dictionary_to_file
from constants import *

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

			exponential_family = NormalArray
			exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)

			for structure in STRUCTURES:
				print_message += (" " + structure)
				evaluation_message(print_message)

				graph = None
				if structure == POON_DOMINGOS:
					structure_args = dict()
					structure_args['height'] = MNIST_HEIGHT
					structure_args['width'] = MNIST_WIDTH
					structure_args['pd_num_pieces'] = PD_NUM_PIECES
					graph = SPN.load_structure(1, structure, dataset_name, structure_args)
				else:
					structure_args = dict()
					structure_args['num_var'] = train_x.shape[1]
					structure_args['depth'] = DEFAULT_DEPTH
					structure_args['num_repetitions'] = DEFAULT_NUM_REPETITIONS
					graph = SPN.load_structure(1, structure, dataset_name, structure_args)

				for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:
					print_message += (" " + "number of distributions {}".format(num_distributions))
					evaluation_message(print_message)

					einet_args = dict()
					einet_args['num_var'] = train_x.shape[1]
					einet_args['num_sums'] = num_distributions
					einet_args['num_input_distributions'] = num_distributions
					einet_args['exponential_family'] = exponential_family
					einet_args['exponential_family_args'] = exponential_family_args
					einet_args['online_em_frequency'] = DEFAULT_ONLINE_EM_FREQUENCY
					einet_args['online_em_stepsize'] = DEFAULT_ONLINE_EM_STEPSIZE
					einet_args['num_repetitions'] = DEFAULT_NUM_REPETITIONS

					print_message += (" " + "loading einet")
					evaluation_message(print_message)

					einet = SPN.load_einet(structure, dataset_name, einet_args)

					print_message += (" " + "training einet")
					evaluation_message(print_message)

					trained_einet = SPN.train_clean_einet(structure, dataset_name, einet, train_x, train_labels,
														  valid_x,
														  valid_labels, test_x, test_labels, einet_args,
														  batch_size=DEFAULT_TRAIN_BATCH_SIZE)

					print_message += (" " + "saving trained einet")
					evaluation_message(print_message)

					mkdir_p(MODEL_DIRECTORY)
					file_name = os.path.join(MODEL_DIRECTORY, "_".join(
						[structure, dataset_name, str(num_distributions), str(num_distributions),
						 str(DEFAULT_NUM_REPETITIONS)]) + ".mdl")
					torch.save(trained_einet, file_name)

					print_message += (" " + "generating samples")
					evaluation_message(print_message)

					SPN.generate_samples(trained_einet, structure, dataset_name, einet_args)

					SPN.generate_conditional_samples(einet, structure, dataset_name, einet_args, test_x)


def test_standard_spn_discrete(specific_datasets=None, is_adv=False):
	if specific_datasets is None:
		specific_datasets = DISCRETE_DATASETS
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	results = dict()
	for dataset_name in specific_datasets:
		evaluation_message("Dataset : {}".format(dataset_name))

		dataset_results = dict()

		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		exponential_family, exponential_family_args, structures = None, None, None
		if dataset_name in DISCRETE_DATASETS:
			structures = [BINARY_TREES]
			exponential_family = CategoricalArray
			exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)
		elif dataset_name in CONTINUOUS_DATASETS:
			structures = STRUCTURES
			exponential_family = NormalArray
			exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)

		for structure in structures:

			evaluation_message("Using the structure {}".format(structure))

			graph = None
			if structure == POON_DOMINGOS:
				structure_args = dict()
				structure_args[HEIGHT] = MNIST_HEIGHT
				structure_args[WIDTH] = MNIST_WIDTH
				structure_args[PD_NUM_PIECES] = PD_NUM_PIECES
				graph = SPN.load_structure(1, structure, dataset_name, structure_args)
			else:
				structure_args = dict()
				structure_args[NUM_VAR] = train_x.shape[1]
				structure_args[DEPTH] = DEFAULT_DEPTH
				structure_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
				graph = SPN.load_structure(1, structure, dataset_name, structure_args)

			for num_distributions in NUM_INPUT_DISTRIBUTIONS_LIST:

				dataset_distribution_results = dict()

				evaluation_message("Number of distributions {}".format(num_distributions))

				einet_args = dict()
				einet_args[NUM_VAR] = train_x.shape[1]
				einet_args[NUM_SUMS] = num_distributions
				einet_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
				einet_args[EXPONENTIAL_FAMILY] = exponential_family
				einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
				einet_args[ONLINE_EM_FREQUENCY] = DEFAULT_ONLINE_EM_FREQUENCY
				einet_args[ONLINE_EM_STEPSIZE] = DEFAULT_ONLINE_EM_STEPSIZE
				einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

				evaluation_message("Loading Einet")

				einet = SPN.load_einet(structure, dataset_name, einet_args)

				trained_einet = None
				if is_adv:
					evaluation_message("Training adversarial einet")
					trained_einet = SPN.train_einet(1, structure, dataset_name, einet, train_labels, train_x, valid_x,
													test_x, einet_args, RESTRICTED_LOCAL_SEARCH,
													batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=True)
				else:
					evaluation_message("Training clean einet")
					trained_einet = SPN.train_einet(1, structure, dataset_name, einet, train_labels, train_x, valid_x,
													test_x, einet_args, RESTRICTED_LOCAL_SEARCH,
													batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=False)

				mean_ll, std_ll = SPN.test_einet(dataset_name, trained_einet, test_x, None, RESTRICTED_LOCAL_SEARCH,
												 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=False)
				evaluation_message("Clean Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

				dataset_distribution_results['Clean Mean LL'] = mean_ll
				dataset_distribution_results['Clean Std LL'] = std_ll

				mean_ll, std_ll = SPN.test_einet(dataset_name, trained_einet, test_x, None, RESTRICTED_LOCAL_SEARCH,
												 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
				evaluation_message("Adv Test - Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

				dataset_distribution_results['Adv Mean LL'] = mean_ll
				dataset_distribution_results['Adv Std LL'] = std_ll

				if dataset_name == BINARY_MNIST:
					evaluation_message("Generating samples")
					SPN.generate_samples(trained_einet, dataset_name, einet_args)

					evaluation_message("Generating conditional samples")
					SPN.generate_conditional_samples(trained_einet, dataset_name, einet_args, test_x)

				for evidence_percentage in EVIDENCE_PERCENTAGES:
					dataset_distribution_evidence_results = dict()

					mean_ll, std_ll = SPN.test_conditional_einet(CLEAN, 0, dataset_name, trained_einet,
																 evidence_percentage, test_x,
																 batch_size=DEFAULT_EVAL_BATCH_SIZE)
					evaluation_message(
						"Clean Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																							mean_ll,
																							std_ll))
					dataset_distribution_evidence_results['Clean Mean LL'] = mean_ll
					dataset_distribution_evidence_results['Clean Std LL'] = std_ll

					mean_ll, std_ll = SPN.test_conditional_einet(CLEAN, 0, dataset_name, trained_einet,
																 evidence_percentage, test_x,
																 batch_size=DEFAULT_EVAL_BATCH_SIZE)
					evaluation_message(
						"Adv Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage, mean_ll,
																						  std_ll))
					dataset_distribution_evidence_results['Adv Mean LL'] = mean_ll
					dataset_distribution_evidence_results['Adv Std LL'] = std_ll

					dataset_distribution_results[evidence_percentage] = dataset_distribution_evidence_results
				dataset_results[num_distributions] = dataset_distribution_results

		results[dataset_name] = dataset_results
		dictionary_to_file(dataset_name, dataset_results, 1, is_adv=is_adv, is_einet=True)
		pretty_print_dictionary(dataset_results)
	pretty_print_dictionary(results)


if __name__ == '__main__':
	test_standard_spn_discrete(DEBD_DATASETS, is_adv=False)
