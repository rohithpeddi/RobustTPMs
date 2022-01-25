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


def fetch_einet_args(dataset_name, num_var, exponential_family, exponential_family_args):
	einet_args = dict()
	einet_args[NUM_VAR] = num_var

	num_distributions, online_em_frequency, batch_size = None, DEFAULT_ONLINE_EM_FREQUENCY, DEFAULT_TRAIN_BATCH_SIZE
	if dataset_name in ['plants', 'accidents', 'tretail']:
		num_distributions = 20
		batch_size = 100
		online_em_frequency = 50
	elif dataset_name in ['nltcs', 'msnbc', 'kdd', 'pumsb_star', 'kosarek', 'msweb']:
		num_distributions = 10
		batch_size = 100
		online_em_frequency = 50
	elif dataset_name in ['baudio', 'jester', 'bnetflix', 'book']:
		num_distributions = 10
		batch_size = 50
		online_em_frequency = 5
	elif dataset_name in ['cwebkb', 'bbc', 'tmovie', 'cr52', 'dna']:
		num_distributions = 20
		batch_size = 50
		online_em_frequency = 1
	elif dataset_name in ['c20ng']:
		num_distributions = 20
		batch_size = 50
		online_em_frequency = 5
	elif dataset_name in ['ad']:
		num_distributions = 20
		batch_size = 50
		online_em_frequency = 5

	einet_args[NUM_SUMS] = num_distributions
	einet_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
	einet_args[EXPONENTIAL_FAMILY] = exponential_family
	einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
	einet_args[ONLINE_EM_FREQUENCY] = online_em_frequency
	einet_args[ONLINE_EM_STEPSIZE] = DEFAULT_ONLINE_EM_STEPSIZE
	einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
	einet_args[BATCH_SIZE] = batch_size

	return einet_args


def test_standard_spn_discrete(run_id, specific_datasets=None, is_adv=False, train_attack_type=None, perturbations=None,
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

			dataset_distribution_results = dict()

			einet_args = fetch_einet_args(dataset_name, train_x.shape[1], exponential_family, exponential_family_args)

			evaluation_message("Loading Einet")

			trained_clean_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args)

			if trained_clean_einet is None:
				clean_einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph)
				evaluation_message("Training clean einet")
				trained_clean_einet = SPN.train_einet(run_id, structure, dataset_name, clean_einet, train_x, valid_x,
													  test_x, einet_args, perturbations, CLEAN,
													  batch_size=einet_args[BATCH_SIZE],
													  is_adv=False)

			einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph)
			trained_adv_einet = trained_clean_einet
			if is_adv:
				trained_adv_einet = SPN.load_pretrained_einet(run_id, structure, dataset_name, einet_args, train_attack_type, perturbations)
				if trained_adv_einet is None:
					evaluation_message("Training adversarial einet with attack type {}".format(train_attack_type))
					trained_adv_einet = SPN.train_einet(run_id, structure, dataset_name, einet, train_x, valid_x, test_x,
														einet_args, perturbations, train_attack_type,
														batch_size=einet_args[BATCH_SIZE], is_adv=True)
				else:
					evaluation_message("Loaded pretrained einet for the configuration")

			# ------------------------------------------------------------------------------------------
			# ------  TESTING AREA ------

			# 1. Original Test Set
			mean_ll, std_ll, clean_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														   test_labels, perturbations=0, attack_type=CLEAN,
														   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=False)
			evaluation_message("Clean Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Original Mean LL'] = mean_ll
			dataset_distribution_results['Original Std LL'] = std_ll

			# ------  LOCAL SEARCH AREA ------

			# 2. Local Search - 1 Test Set
			mean_ll, std_ll, ls1_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=1, attack_type=LOCAL_SEARCH,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Local Search - 1 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Local Search - 1 Mean LL'] = mean_ll
			dataset_distribution_results['Local Search - 1 Std LL'] = std_ll

			# 3. Local Search - 3 Test Set
			mean_ll, std_ll, ls3_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=3, attack_type=LOCAL_SEARCH,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Local Search - 3 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Local Search - 3 Mean LL'] = mean_ll
			dataset_distribution_results['Local Search - 3 Std LL'] = std_ll

			# 4. Local Search - 5 Test Set
			mean_ll, std_ll, ls5_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=5, attack_type=LOCAL_SEARCH,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Local Search - 5 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Local Search - 5 Mean LL'] = mean_ll
			dataset_distribution_results['Local Search - 5 Std LL'] = std_ll

			# ------  RESTRICTED LOCAL SEARCH AREA ------

			# 5. Local Search - 1 Test Set
			mean_ll, std_ll, rls1_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														  test_labels, perturbations=1,
														  attack_type=RESTRICTED_LOCAL_SEARCH,
														  batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Restricted Local Search - 1 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Restricted Local Search - 1 Mean LL'] = mean_ll
			dataset_distribution_results['Restricted Local Search - 1 Std LL'] = std_ll

			# 6. Local Search - 3 Test Set
			mean_ll, std_ll, rls3_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														  test_labels, perturbations=3,
														  attack_type=RESTRICTED_LOCAL_SEARCH,
														  batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Restricted Local Search - 3 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Restricted Local Search- 3 Mean LL'] = mean_ll
			dataset_distribution_results['Restricted Local Search - 3 Std LL'] = std_ll

			# 7. Local Search - 5 Test Set
			mean_ll, std_ll, rls5_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														  test_labels, perturbations=5,
														  attack_type=RESTRICTED_LOCAL_SEARCH,
														  batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Restricted Local Search - 5 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Restricted Local Search - 5 Mean LL'] = mean_ll
			dataset_distribution_results['Restricted Local Search - 5 Std LL'] = std_ll

			# ------  NEURAL NETWORK AREA ------

			# 8. Neural Network - 1 Test Set
			mean_ll, std_ll, nn1_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=1,
														 attack_type=NEURAL_NET,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Neural Network - 1 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Neural Network - 1 Mean LL'] = mean_ll
			dataset_distribution_results['Neural Network - 1 Std LL'] = std_ll

			# 9. Neural Network - 3 Test Set
			mean_ll, std_ll, nn3_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=3,
														 attack_type=NEURAL_NET,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Neural Network - 3 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Neural Network- 3 Mean LL'] = mean_ll
			dataset_distribution_results['Neural Network - 3 Std LL'] = std_ll

			# 10. Neural Network - 5 Test Set
			mean_ll, std_ll, nn5_text_x = SPN.test_einet(dataset_name, trained_adv_einet, trained_clean_einet, test_x,
														 test_labels, perturbations=5,
														 attack_type=NEURAL_NET,
														 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
			evaluation_message("Neural Network - 5 Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Neural Network - 5 Mean LL'] = mean_ll
			dataset_distribution_results['Neural Network - 5 Std LL'] = std_ll

			for evidence_percentage in EVIDENCE_PERCENTAGES:
				dataset_distribution_evidence_results = dict()

				# 1. Original Test Set
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 clean_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Clean Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																						mean_ll, std_ll))
				dataset_distribution_evidence_results['Clean Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Clean Std LL'] = std_ll

				# ---------- LOCAL SEARCH AREA ------

				# 2. Local search - 1
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 ls1_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Local search - 1  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																									mean_ll, std_ll))
				dataset_distribution_evidence_results['Local search - 1 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Local search - 1 Std LL'] = std_ll

				# 3. Local search - 3
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 ls3_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Local search - 3  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																									mean_ll, std_ll))
				dataset_distribution_evidence_results['Local search - 3 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Local search - 3 Std LL'] = std_ll

				# 4. Local search - 5
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 ls5_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Local search - 5  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																									mean_ll, std_ll))
				dataset_distribution_evidence_results['Local search - 5 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Local search - 5 Std LL'] = std_ll

				# ---------- RESTRICTED LOCAL SEARCH AREA ------

				# 5. Restricted Local search - 1
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 rls1_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Restricted Local search - 1  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Restricted Local search - 1 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Restricted Local search - 1 Std LL'] = std_ll

				# 6. Restricted Local search - 3
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 rls3_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Restricted Local search - 3  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Restricted Local search - 3 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Restricted Local search - 3 Std LL'] = std_ll

				# 7. Restricted Local search - 5
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 rls5_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Restricted Local search - 5  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Restricted Local search - 5 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Restricted Local search - 5 Std LL'] = std_ll

				# ---------- NEURAL NETWORK AREA ------

				# 8. Neural Network - 1
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 nn1_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Neural Network - 1  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Neural Network - 1 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Neural Network - 1 Std LL'] = std_ll

				# 9. Neural Network - 3
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 nn3_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Neural Network - 3  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Neural Network - 3 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Neural Network - 3 Std LL'] = std_ll

				# 10. Restricted Local search - 5
				mean_ll, std_ll = SPN.test_conditional_einet(dataset_name, trained_adv_einet, evidence_percentage,
															 nn5_text_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
				evaluation_message(
					"Neural Network - 5  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(
						evidence_percentage,
						mean_ll, std_ll))
				dataset_distribution_evidence_results['Neural Network - 5 Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Neural Network - 5 Std LL'] = std_ll

				dataset_distribution_results[evidence_percentage] = dataset_distribution_evidence_results
				dataset_results[einet_args[NUM_INPUT_DISTRIBUTIONS]] = dataset_distribution_results

		results[dataset_name] = dataset_results
		dictionary_to_file(dataset_name, dataset_results, run_id, train_attack_type, perturbations, is_adv=is_adv,
						   is_einet=True)
		pretty_print_dictionary(dataset_results)
	pretty_print_dictionary(results)


if __name__ == '__main__':
	for perturbation in PERTURBATIONS:
		evaluation_message("Training for Perturbation : {}".format(perturbation))
		test_standard_spn_discrete(run_id=21, specific_datasets=DEBD_DATASETS, is_adv=True,
								   train_attack_type=RESTRICTED_LOCAL_SEARCH, perturbations=perturbation)
