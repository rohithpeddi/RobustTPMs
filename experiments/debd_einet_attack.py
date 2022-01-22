import torch
import numpy as np
import torch

import einet_base_spn as SPN
from EinsumNetwork import EinsumNetwork
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray
from attacks.evolutionary import attack as de_attack
from constants import *
from utils import pretty_print_dictionary

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def train_adv_debd(dataset_name, is_adv=False):
	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)
	dataset_results = dict()
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
			einet_args = dict()
			einet_args[NUM_VAR] = train_x.shape[1]
			einet_args[NUM_SUMS] = num_distributions
			einet_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
			einet_args[EXPONENTIAL_FAMILY] = exponential_family
			einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
			einet_args[ONLINE_EM_FREQUENCY] = 1
			einet_args[ONLINE_EM_STEPSIZE] = DEFAULT_ONLINE_EM_STEPSIZE
			einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS

			einet = SPN.load_einet(structure, dataset_name, einet_args)

			original_N = train_x.shape[0]
			for epoch in range(40):
				einet.train()
				idx_batches = torch.randperm(train_x.shape[1], device=device).split(TRAIN_BATCH_SIZE)
				for batch_count, idx in enumerate(idx_batches):
					batch_x = train_x[idx, :]
					outputs = einet.forward(batch_x)
					ll_sample = EinsumNetwork.log_likelihoods(outputs)
					log_likelihood = ll_sample.sum()
					objective = log_likelihood
					objective.backward()
					einet.em_process_batch()
				einet.em_update()
				train_x = train_x[0:original_N, :]
				einet.eval()
				train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x, batch_size=EVAL_BATCH_SIZE)
				valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=EVAL_BATCH_SIZE)
				test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=EVAL_BATCH_SIZE)
				print("[{}] train LL {} valid LL {} test LL {}".format(epoch, train_ll / train_x.shape[0],
																	   valid_ll / valid_x.shape[0],
																	   test_ll / test_x.shape[0]))
			if is_adv:
				train_x = de_attack.generate_adv_dataset(einet, dataset_name, train_x)
				einet = SPN.load_einet(structure, dataset_name, einet_args)

				for epoch in range(40):
					einet.train()
					idx_batches = torch.randperm(train_x.shape[1], device=device).split(TRAIN_BATCH_SIZE)
					for batch_count, idx in enumerate(idx_batches):
						batch_x = train_x[idx, :]
						outputs = einet.forward(batch_x)
						ll_sample = EinsumNetwork.log_likelihoods(outputs)
						log_likelihood = ll_sample.sum()
						objective = log_likelihood
						objective.backward()
						einet.em_process_batch()
					einet.em_update()
					train_x_copy = train_x[0:original_N, :]
					einet.eval()
					train_ll = EinsumNetwork.eval_loglikelihood_batched(einet, train_x_copy, batch_size=EVAL_BATCH_SIZE)
					valid_ll = EinsumNetwork.eval_loglikelihood_batched(einet, valid_x, batch_size=EVAL_BATCH_SIZE)
					test_ll = EinsumNetwork.eval_loglikelihood_batched(einet, test_x, batch_size=EVAL_BATCH_SIZE)
					print("[{}] train LL {} valid LL {} test LL {}".format(epoch, train_ll / train_x_copy.shape[0],
																		   valid_ll / valid_x.shape[0],
																		   test_ll / test_x.shape[0]))

			einet.eval()
			test_lls = EinsumNetwork.fetch_likelihoods_for_data(einet, test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
			mean_ll = (torch.mean(test_lls)).cpu().item()
			std_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
			evaluation_message("Clean Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Clean Mean LL'] = mean_ll
			dataset_distribution_results['Clean Std LL'] = std_ll

			test_x_copy = de_attack.generate_adv_dataset(einet, dataset_name, test_x, combine=False)
			test_lls = EinsumNetwork.fetch_likelihoods_for_data(einet, test_x_copy, batch_size=DEFAULT_EVAL_BATCH_SIZE)
			mean_ll = (torch.mean(test_lls)).cpu().item()
			std_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
			evaluation_message("Adv Test - Mean LL : {}, Std LL : {}".format(mean_ll, std_ll))

			dataset_distribution_results['Adv Mean LL'] = mean_ll
			dataset_distribution_results['Adv Std LL'] = std_ll

			for evidence_percentage in EVIDENCE_PERCENTAGES:
				test_x_copy = test_x.clone().detach()
				dataset_distribution_evidence_results = dict()
				num_dims = train_x.shape[1]
				marginalize_idx = list(np.arange(int(num_dims * evidence_percentage), num_dims))
				einet.eval()
				test_lls = EinsumNetwork.fetch_conditional_likelihoods_for_data(einet, test_x_copy,
																				marginalize_idx=marginalize_idx,
																				batch_size=DEFAULT_EVAL_BATCH_SIZE)
				mean_ll = (torch.mean(test_lls)).cpu().item()
				std_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
				evaluation_message(
					"Clean Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage,
																						mean_ll,
																						std_ll))
				dataset_distribution_evidence_results['Clean Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Clean Std LL'] = std_ll

				test_x_copy = de_attack.generate_adv_dataset(einet, dataset_name, test_x_copy)
				test_lls = EinsumNetwork.fetch_conditional_likelihoods_for_data(einet, test_x_copy,
																				marginalize_idx=marginalize_idx,
																				batch_size=DEFAULT_EVAL_BATCH_SIZE)
				mean_ll = (torch.mean(test_lls)).cpu().item()
				std_ll = (2.0 * torch.std(test_lls) / np.sqrt(len(test_lls))).cpu().item()
				evaluation_message(
					"Adv Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(evidence_percentage, mean_ll,
																					  std_ll))
				dataset_distribution_evidence_results['Adv Mean LL'] = mean_ll
				dataset_distribution_evidence_results['Adv Std LL'] = std_ll

				dataset_distribution_results[evidence_percentage] = dataset_distribution_evidence_results
			dataset_results[num_distributions] = dataset_distribution_results
	pretty_print_dictionary(dataset_results)


if __name__ == '__main__':
	train_adv_debd('plants', is_adv=False)
