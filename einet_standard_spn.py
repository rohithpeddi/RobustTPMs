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
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def test_standard_spn_discrete(run_id, specific_datasets=None, is_adv=False, train_attack_type=None,
							   test_attack_type=None):
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
				graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)
			else:
				structure_args = dict()
				structure_args[NUM_VAR] = train_x.shape[1]
				structure_args[DEPTH] = DEFAULT_DEPTH
				structure_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
				graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)

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

				einet = SPN.load_einet(run_id, structure, dataset_name, einet_args)

				trained_einet = None
				if is_adv:
					evaluation_message("Training adversarial einet with attack type {}".format(train_attack_type))
					trained_einet = SPN.train_einet(run_id, structure, dataset_name, einet, train_x, valid_x, test_x,
													einet_args, train_attack_type,
													batch_size=DEFAULT_TRAIN_BATCH_SIZE, is_adv=True)

				else:
					evaluation_message("Training clean einet")
					trained_einet = SPN.train_einet(run_id, structure, dataset_name, einet, train_x, valid_x, test_x,
													einet_args, CLEAN, batch_size=DEFAULT_TRAIN_BATCH_SIZE,
													is_adv=False)

				# 1. Original Test Set
				mean_ll, std_ll, text_x = SPN.test_einet(dataset_name, trained_einet, test_x, test_labels, None,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=False)
				evaluation_message("Clean Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

				dataset_distribution_results['Original Mean LL'] = mean_ll
				dataset_distribution_results['Original Std LL'] = std_ll

				if is_adv:
					clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args)
					evaluation_message("Training clean einet")
					trained_clean_einet = SPN.train_einet(run_id, structure, dataset_name, clean_einet, train_x,
														  valid_x, test_x,
														  einet_args, CLEAN, batch_size=DEFAULT_TRAIN_BATCH_SIZE,
														  is_adv=False)
				else:
					trained_clean_einet = trained_einet

				# 2. Test set modified using LS on model M
				mean_ll, std_ll, adv_test_x_ls = SPN.test_einet(dataset_name, trained_clean_einet, test_x, test_labels,
																LOCAL_SEARCH, batch_size=1, is_adv=True)
				evaluation_message(
					"Attack type : {}, Adv Test - Mean LL : {}, Std LL : {}".format(LOCAL_SEARCH, mean_ll, std_ll))

				dataset_distribution_results["Local Search Mean LL"] = mean_ll
				dataset_distribution_results["Local Search Std LL"] = std_ll

				# 3. Test set modified with neural network
				mean_ll, std_ll, adv_test_x_nn = SPN.test_einet(dataset_name, trained_clean_einet, test_x, test_labels,
																NEURAL_NET, batch_size=1, is_adv=True)
				evaluation_message(
					"Attack type : {}, Adv Test - Mean LL : {}, Std LL : {}".format(NEURAL_NET, mean_ll, std_ll))

				dataset_distribution_results["Neural net Mean LL"] = mean_ll
				dataset_distribution_results["Neural net Std LL"] = std_ll

				for evidence_percentage in EVIDENCE_PERCENTAGES:
					dataset_distribution_evidence_results = dict()

					# 1. Original Test Set
					mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_einet, evidence_percentage,
																 test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
					evaluation_message(
						"Clean Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																							mean_ll,
																							std_ll))
					# 2. Test set modified using LS on model M
					dataset_distribution_evidence_results['Clean Mean LL'] = mean_ll
					dataset_distribution_evidence_results['Clean Std LL'] = std_ll

					mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_einet, evidence_percentage,
																 adv_test_x_ls, batch_size=DEFAULT_EVAL_BATCH_SIZE)
					evaluation_message(
						"Adv type:  {}, Evidence percentage : {}, Mean CLL : {}, Std CLL  : {}".format(LOCAL_SEARCH,
																									   evidence_percentage,
																									   mean_ll,
																									   std_ll))
					# 3. Test set modified with neural network
					dataset_distribution_evidence_results["Local Search Mean CLL"] = mean_ll
					dataset_distribution_evidence_results["Local Search Std CLL"] = std_ll

					mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_einet, evidence_percentage,
																 adv_test_x_nn, batch_size=DEFAULT_EVAL_BATCH_SIZE)
					evaluation_message(
						"Adv type:  {}, Evidence percentage : {}, Mean CLL : {}, Std CLL  : {}".format(NEURAL_NET,
																									   evidence_percentage,
																									   mean_ll,
																									   std_ll))
					dataset_distribution_evidence_results["Neural Net Mean CLL"] = mean_ll
					dataset_distribution_evidence_results["Neural Net Std CLL"] = std_ll

					dataset_distribution_results[evidence_percentage] = dataset_distribution_evidence_results
				dataset_results[num_distributions] = dataset_distribution_results

		results[dataset_name] = dataset_results
		dictionary_to_file(dataset_name, dataset_results, run_id, is_adv=is_adv, is_einet=True)
		pretty_print_dictionary(dataset_results)
	pretty_print_dictionary(results)


if __name__ == '__main__':
	for dataset_name in DEBD_DATASETS:
		test_standard_spn_discrete(run_id=1, specific_datasets=[dataset_name], is_adv=True,
								   train_attack_type=LOCAL_SEARCH)
# for dataset_name in DEBD_DATASETS:
# 	test_standard_spn_discrete(run_id=1, specific_datasets=[dataset_name], is_adv=False,
# 							   train_attack_type=CLEAN, test_attack_type=[CLEAN, LOCAL_SEARCH, NEURAL_NET])
# 	test_standard_spn_discrete(run_id=2, specific_datasets=[dataset_name], is_adv=False,
# 							   train_attack_type=LOCAL_SEARCH, test_attack_type=[CLEAN, LOCAL_SEARCH, NEURAL_NET])
# 	test_standard_spn_discrete(run_id=3, specific_datasets=[dataset_name], is_adv=False,
# 							   train_attack_type=NEURAL_NET, test_attack_type=[CLEAN, LOCAL_SEARCH, NEURAL_NET])
# 	test_standard_spn_discrete(run_id=4, specific_datasets=[dataset_name], is_adv=False,
# 							   train_attack_type=LOCAL_RESTRICTED_SEARCH, test_attack_type=[CLEAN, LOCAL_SEARCH, NEURAL_NET])
