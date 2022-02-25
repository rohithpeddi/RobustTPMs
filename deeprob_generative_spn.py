import torch
import wandb
import base_spn as SPN

from constants import *
from utils import pretty_print_dictionary, dictionary_to_file

############################################################################


# run1 = wandb.init(project="ROSPN-O", entity="utd-ml-pgm")
# run1 = wandb.init(project="ROSPN", entity="rohithpeddi")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

discrete_columns = ["attack_type", "perturbations", "standard_mean_ll", "standard_std_ll",
					"ls1_mean_ll", "ls1_std_ll", "ls3_mean_ll", "ls3_std_ll", "ls5_mean_ll", "ls5_std_ll",
					"rls1_mean_ll", "rls1_std_ll", "rls3_mean_ll", "rls3_std_ll", "rls5_mean_ll", "rls5_std_ll",
					"av1_mean_ll", "av1_std_ll", "av3_mean_ll", "av3_std_ll", "av5_mean_ll", "av5_std_ll",
					"w1_mean_ll", "w1_std_ll", "w3_mean_ll", "w3_std_ll", "w5_mean_ll", "w5_std_ll"]

wandb_tables = dict()


############################################################################

def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def fetch_wandb_table(dataset_name):
	if dataset_name not in wandb_tables:
		dataset_wandb_tables = dict()

		ll_table = wandb.Table(columns=discrete_columns)
		dataset_wandb_tables[LOGLIKELIHOOD_TABLE] = ll_table

		cll_wandb_tables = dict()
		for evidence_percentage in EVIDENCE_PERCENTAGES:
			cll_table = wandb.Table(columns=discrete_columns)
			cll_wandb_tables[evidence_percentage] = cll_table
		dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES] = cll_wandb_tables

		wandb_tables[dataset_name] = dataset_wandb_tables

	return wandb_tables[dataset_name]


def fetch_spn_args(dataset_name, num_var, model_type):
	spn_args = dict()
	spn_args[NUM_EPOCHS] = MAX_NUM_EPOCHS
	spn_args[OUT_CLASSES] = GENERATIVE_NUM_CLASSES
	num_distributions, batch_size = None, DEFAULT_TRAIN_BATCH_SIZE
	if dataset_name in DEBD_DATASETS:
		if dataset_name in ['plants', 'accidents', 'tretail']:
			num_distributions = 20
			batch_size = 100
		elif dataset_name in ['nltcs', 'msnbc', 'kdd', 'pumsb_star', 'kosarek', 'msweb']:
			num_distributions = 10
			batch_size = 100
		elif dataset_name in ['baudio', 'jester', 'bnetflix', 'book']:
			num_distributions = 10
			batch_size = 50
		elif dataset_name in ['tmovie', 'dna']:
			num_distributions = 20
			batch_size = 50
		elif dataset_name in ['cwebkb', 'bbc', 'cr52', 'c20ng', 'ad']:
			num_distributions = 10
			batch_size = 50
		spn_args[N_FEATURES] = num_var
		spn_args[BATCH_SIZE] = batch_size
		spn_args[NUM_SUMS] = num_distributions
		spn_args[NUM_INPUT_DISTRIBUTIONS] = num_distributions
		spn_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
		spn_args[MODEL_TYPE] = BERNOULLI_RATSPN
		spn_args[LEARNING_RATE] = 1e-2
		spn_args[NUM_EPOCHS] = 1000
		spn_args[DEPTH] = 3
	elif dataset_name == MNIST:
		spn_args[BATCH_SIZE] = 100
		spn_args[NUM_EPOCHS] = 65
		if model_type == DGCSPN:
			spn_args[N_FEATURES] = (MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH)
			spn_args[PATIENCE] = 2
			spn_args[BATCHED_LEAVES] = 30
			spn_args[SUM_CHANNELS] = 60
			spn_args[SUM_DROPOUT] = 0.2
			spn_args[IN_DROPOUT] = 0.2
			spn_args[NUM_POOLING] = 2
			spn_args[MODEL_TYPE] = DGCSPN
			spn_args[LEARNING_RATE] = 1e-2
			spn_args[NUM_EPOCHS] = 1000
		elif model_type == GAUSSIAN_RATSPN:
			spn_args[N_FEATURES] = MNIST_CHANNELS * MNIST_HEIGHT * MNIST_WIDTH
			spn_args[PATIENCE] = 2
			spn_args[DEPTH] = 3
			spn_args[NUM_SUMS] = DEFAULT_NUM_MNIST_SUMS
			spn_args[NUM_INPUT_DISTRIBUTIONS] = DEFAULT_NUM_MNIST_INPUT_DISTRIBUTIONS
			spn_args[NUM_REPETITIONS] = DEFAULT_NUM_MNIST_REPETITIONS
			spn_args[MODEL_TYPE] = GAUSSIAN_RATSPN
			spn_args[LEARNING_RATE] = 5e-3
	elif dataset_name == FASHION_MNIST:
		spn_args[BATCH_SIZE] = 100
		spn_args[NUM_EPOCHS] = 65
		if model_type == DGCSPN:
			spn_args[N_FEATURES] = (FASHION_MNIST_CHANNELS, FASHION_MNIST_HEIGHT, FASHION_MNIST_WIDTH)
			spn_args[PATIENCE] = 2
			spn_args[BATCHED_LEAVES] = 30
			spn_args[SUM_CHANNELS] = 60
			spn_args[SUM_DROPOUT] = 0.2
			spn_args[IN_DROPOUT] = 0.2
			spn_args[NUM_POOLING] = 2
			spn_args[MODEL_TYPE] = DGCSPN
			spn_args[LEARNING_RATE] = 1e-2
			spn_args[NUM_EPOCHS] = 1000
		elif model_type == GAUSSIAN_RATSPN:
			spn_args[N_FEATURES] = FASHION_MNIST_CHANNELS * FASHION_MNIST_HEIGHT * FASHION_MNIST_WIDTH
			spn_args[PATIENCE] = 2
			spn_args[DEPTH] = 3
			spn_args[NUM_SUMS] = DEFAULT_NUM_MNIST_SUMS
			spn_args[NUM_INPUT_DISTRIBUTIONS] = DEFAULT_NUM_MNIST_INPUT_DISTRIBUTIONS
			spn_args[NUM_REPETITIONS] = DEFAULT_NUM_MNIST_REPETITIONS
			spn_args[MODEL_TYPE] = GAUSSIAN_RATSPN
			spn_args[LEARNING_RATE] = 5e-3
	spn_args[TRAIN_SETTING] = GENERATIVE
	return spn_args


def test_standard_spn(run_id, specific_datasets=None, is_adv=False, train_attack_type=None, perturbations=None):
	if specific_datasets is None:
		specific_datasets = [MNIST, FASHION_MNIST]
	else:
		specific_datasets = [specific_datasets] if type(specific_datasets) is not list else specific_datasets

	results = dict()
	for dataset_name in specific_datasets:
		# dataset_wandb_tables = fetch_wandb_table(dataset_name)
		# ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
		# cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]

		evaluation_message("Dataset : {}".format(dataset_name))
		dataset_results = dict()

		if dataset_name in DEBD_DATASETS:
			model_types = [BERNOULLI_RATSPN]
		else:
			model_types = [DGCSPN, GAUSSIAN_RATSPN]

		for model_type in model_types:
			evaluation_message("Using the model type {}".format(model_type))

			train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name,
																								 model_type)

			dataset_distribution_results = dict()
			spn_args = fetch_spn_args(dataset_name, train_x.shape[1], model_type)

			evaluation_message("Loading SPN")
			trained_clean_spn = SPN.load_pretrained_spn(run_id, dataset_name, spn_args, train_attack_type=CLEAN,
														perturbations=0.0)

			if trained_clean_spn is None:
				clean_spn = SPN.load_spn(dataset_name, spn_args)
				evaluation_message("Training clean spn")
				trained_clean_spn = SPN.train_generative_spn(run_id, dataset_name, clean_spn, train_x, valid_x, test_x,
															 spn_args, perturbations=0.0, is_adv=False,
															 train_labels=None, valid_labels=None, test_labels=None,
															 train_attack_type=CLEAN)

			adv_spn = SPN.load_spn(dataset_name, spn_args)
			trained_adv_spn = trained_clean_spn
			if is_adv:
				trained_adv_spn = SPN.load_pretrained_spn(run_id, dataset_name, spn_args,
														  train_attack_type=train_attack_type,
														  perturbations=perturbations)
				if trained_adv_spn is None:
					evaluation_message("Training adversarial spn with attack type {}".format(train_attack_type))
					trained_adv_spn = SPN.train_generative_spn(run_id, dataset_name, adv_spn, train_x, valid_x, test_x,
															   spn_args, perturbations=perturbations, is_adv=True,
															   train_labels=None, valid_labels=None, test_labels=None,
															   train_attack_type=train_attack_type)
				else:
					evaluation_message("Loaded pretrained spn for the configuration")

			# -----------------------------------------------------------------------------------------
			# ------  QUALITATIVE EXAMPLE AREA ------

			# ------------------------------------------------------------------------------------------
			# ------  TESTING AREA ------

			# ------  AVERAGE ATTACK AREA ------

			# av_mean_ll_dict, av_std_ll_dict = SPN.fetch_average_likelihoods_for_data(dataset_name, trained_adv_spn,
			# 																		 test_x)

			def attack_test_einet(dataset_name, trained_adv_spn, trained_clean_spn, train_x, test_x, test_labels,
								  perturbations, attack_type, batch_size, is_adv):
				mean_ll, std_ll, attack_test_x = SPN.test_spn(dataset_name, trained_adv_spn, trained_clean_spn,
															  train_x, test_x, test_labels,
															  perturbations=perturbations, spn_args=spn_args,
															  attack_type=attack_type,
															  is_adv=is_adv)
				evaluation_message("{}-{} Mean LL : {}, Std LL : {}".format(attack_type, perturbations*255, mean_ll, std_ll))

				dataset_distribution_results["{} Mean LL".format(attack_type)] = mean_ll
				dataset_distribution_results["{} Std LL".format(attack_type)] = std_ll

				return mean_ll, std_ll, attack_test_x

			# 1. Original Test Set
			standard_mean_ll, standard_std_ll, standard_test_x = attack_test_einet(dataset_name, trained_adv_spn,
																				   trained_clean_spn, train_x, test_x,
																				   test_labels, perturbations=0,
																				   attack_type=CLEAN,
																				   batch_size=DEFAULT_EVAL_BATCH_SIZE,
																				   is_adv=False)


			# standard_mean_ll, standard_std_ll, standard_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 																	   trained_clean_spn, train_x, test_x,
			# 																	   test_labels, perturbations=0,
			# 																	   attack_type=SPARSEFOOL,
			# 																	   batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 																	   is_adv=True)

			# # ---------- FGSM AREA -----------
			#
			# f1_mean_ll, f1_std_ll, f1_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=1. / 255,
			# 													 attack_type=FGSM,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# f3_mean_ll, f3_std_ll, f3_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=3. / 255,
			# 													 attack_type=FGSM,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# f5_mean_ll, f5_std_ll, f5_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=5. / 255,
			# 													 attack_type=FGSM,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# f8_mean_ll, f8_std_ll, f8_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=8. / 255,
			# 													 attack_type=FGSM,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# # ---------- PGD AREA -----------
			#
			# p1_mean_ll, p1_std_ll, p1_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=1. / 255,
			# 													 attack_type=PGD,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# p3_mean_ll, p3_std_ll, p3_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=3. / 255,
			# 													 attack_type=PGD,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# p5_mean_ll, p5_std_ll, p5_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=5. / 255,
			# 													 attack_type=PGD,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)
			#
			# p8_mean_ll, p8_std_ll, p8_test_x = attack_test_einet(dataset_name, trained_adv_spn,
			# 													 trained_clean_spn, train_x, test_x,
			# 													 test_labels, perturbations=8. / 255,
			# 													 attack_type=PGD,
			# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE,
			# 													 is_adv=True)

	# # ------  LOCAL SEARCH AREA ------
	#
	# # 2. Local Search - 1 Test Set
	# ls1_mean_ll, ls1_std_ll, ls1_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														trained_clean_spn, train_x, test_x, test_labels,
	# 														perturbations=1, attack_type=LOCAL_SEARCH,
	# 														batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 3. Local Search - 3 Test Set
	# ls3_mean_ll, ls3_std_ll, ls3_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														trained_clean_spn, train_x, test_x, test_labels,
	# 														perturbations=3, attack_type=LOCAL_SEARCH,
	# 														batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 4. Local Search - 5 Test Set
	# ls5_mean_ll, ls5_std_ll, ls5_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														trained_clean_spn, train_x, test_x, test_labels,
	# 														perturbations=5, attack_type=LOCAL_SEARCH,
	# 														batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # ------  RESTRICTED LOCAL SEARCH AREA ------
	#
	# # 5. Restricted Local Search - 1 Test Set
	# rls1_mean_ll, rls1_std_ll, rls1_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														   trained_clean_spn, train_x, test_x,
	# 														   test_labels, perturbations=1,
	# 														   attack_type=RESTRICTED_LOCAL_SEARCH,
	# 														   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 6. Restricted Local Search - 3 Test Set
	# rls3_mean_ll, rls3_std_ll, rls3_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														   trained_clean_spn, train_x, test_x,
	# 														   test_labels, perturbations=3,
	# 														   attack_type=RESTRICTED_LOCAL_SEARCH,
	# 														   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 7. Restricted Local Search - 5 Test Set
	# rls5_mean_ll, rls5_std_ll, rls5_test_x = attack_test_einet(dataset_name, trained_adv_spn,
	# 														   trained_clean_spn, train_x, test_x,
	# 														   test_labels, perturbations=5,
	# 														   attack_type=RESTRICTED_LOCAL_SEARCH,
	# 														   batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # ------------- WEAKER MODEL ATTACK --------
	#
	# # 11. Weaker model attack - 1 Test Set
	# w1_mean_ll, w1_std_ll, w1_test_x = attack_test_einet(dataset_name, trained_adv_spn, trained_clean_spn,
	# 													 train_x, test_x, test_labels, perturbations=1,
	# 													 attack_type=WEAKER_MODEL,
	# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 12. Weaker model attack - 3 Test Set
	# w3_mean_ll, w3_std_ll, w3_test_x = attack_test_einet(dataset_name, trained_adv_spn, trained_clean_spn,
	# 													 train_x, test_x, test_labels, perturbations=3,
	# 													 attack_type=WEAKER_MODEL,
	# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)
	#
	# # 13. Weaker model attack - 5 Test Set
	# w5_mean_ll, w5_std_ll, w5_test_x = attack_test_einet(dataset_name, trained_adv_spn, trained_clean_spn,
	# 													 train_x, test_x, test_labels, perturbations=5,
	# 													 attack_type=WEAKER_MODEL,
	# 													 batch_size=DEFAULT_EVAL_BATCH_SIZE, is_adv=True)


# # -------------------------------- LOG LIKELIHOOD TO WANDB TABLES ------------------------------------
#
# ll_table.add_data(train_attack_type, perturbations, standard_mean_ll, standard_std_ll,
# 				  ls1_mean_ll, ls1_std_ll, ls3_mean_ll, ls3_std_ll, ls5_mean_ll, ls5_std_ll,
# 				  rls1_mean_ll, rls1_std_ll, rls3_mean_ll, rls3_std_ll, rls5_mean_ll, rls5_std_ll,
# 				  av_mean_ll_dict[1], av_std_ll_dict[1], av_mean_ll_dict[3], av_std_ll_dict[3],
# 				  av_mean_ll_dict[5], av_std_ll_dict[5],
# 				  w1_mean_ll, w1_std_ll, w3_mean_ll, w3_std_ll, w5_mean_ll, w5_std_ll)
#
# # ------------------------------------------------------------------------------------------------
# # ----------------------------- CONDITIONAL LIKELIHOOD AREA --------------------------------------
# # ------------------------------------------------------------------------------------------------
#
# def attack_test_conditional_einet(test_attack_type, perturbations, dataset_name, trained_adv_einet,
# 								  evidence_percentage, test_x, batch_size):
#
# 	mean_ll, std_ll = SPN.test_conditional_einet(test_attack_type, perturbations, dataset_name,
# 												 trained_adv_einet,
# 												 evidence_percentage, test_x, batch_size=batch_size)
# 	evaluation_message(
# 		"{}-{},  Evidence percentage : {}, Mean LL : {}, Std LL  : {}".format(test_attack_type,
# 																			  perturbations,
# 																			  evidence_percentage,
# 																			  mean_ll, std_ll))
# 	dataset_distribution_evidence_results["{}-{} Mean LL".format(test_attack_type, perturbations)] = mean_ll
# 	dataset_distribution_evidence_results["{}-{} Std LL".format(test_attack_type, perturbations)] = std_ll
#
# 	return mean_ll, std_ll
#
# # ---------- AVERAGE ATTACK AREA ------
#
# # 8. Average attack dictionary
# av_mean_cll_dict, av_std_cll_dict = SPN.fetch_average_conditional_likelihoods_for_data(dataset_name,
# 																					   trained_adv_einet,
# 																					   test_x)
#
# for evidence_percentage in EVIDENCE_PERCENTAGES:
# 	cll_table = cll_tables[evidence_percentage]
#
# 	dataset_distribution_evidence_results = dict()
#
# 	# 1. Original Test Set
# 	standard_mean_cll, standard_std_cll = attack_test_conditional_einet(CLEAN, 0, dataset_name,
# 																		trained_adv_einet,
# 																		evidence_percentage,
# 																		standard_test_x,
# 																		batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# ---------- LOCAL SEARCH AREA ------
#
# 	# 2. Local search - 1
# 	ls1_mean_cll, ls1_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 1, dataset_name,
# 															  trained_adv_einet,
# 															  evidence_percentage,
# 															  ls1_test_x,
# 															  batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 3. Local search - 3
# 	ls3_mean_cll, ls3_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 3, dataset_name,
# 															  trained_adv_einet,
# 															  evidence_percentage,
# 															  ls3_test_x,
# 															  batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 4. Local search - 5
# 	ls5_mean_cll, ls5_std_cll = attack_test_conditional_einet(LOCAL_SEARCH, 5, dataset_name,
# 															  trained_adv_einet,
# 															  evidence_percentage,
# 															  ls5_test_x,
# 															  batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# ---------- RESTRICTED LOCAL SEARCH AREA ------
#
# 	# 5. Restricted Local search - 1
# 	rls1_mean_cll, rls1_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 1, dataset_name,
# 																trained_adv_einet, evidence_percentage,
# 																rls1_test_x,
# 																batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 6. Restricted Local search - 3
# 	rls3_mean_cll, rls3_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 3, dataset_name,
# 																trained_adv_einet, evidence_percentage,
# 																rls3_test_x,
# 																batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 7. Restricted Local search - 5
# 	rls5_mean_cll, rls5_std_cll = attack_test_conditional_einet(RESTRICTED_LOCAL_SEARCH, 5, dataset_name,
# 																trained_adv_einet, evidence_percentage,
# 																rls5_test_x,
# 																batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# ---------- WEAKER MODEL AREA ------
#
# 	# 8. Weaker model - 1
# 	w1_mean_cll, w1_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 1, dataset_name,
# 															trained_adv_einet, evidence_percentage,
# 															w1_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 9. Weaker model - 3
# 	w3_mean_cll, w3_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 3, dataset_name,
# 															trained_adv_einet, evidence_percentage,
# 															w3_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# 10. Weaker model - 5
# 	w5_mean_cll, w5_std_cll = attack_test_conditional_einet(WEAKER_MODEL, 5, dataset_name,
# 															trained_adv_einet, evidence_percentage,
# 															w5_test_x, batch_size=DEFAULT_EVAL_BATCH_SIZE)
#
# 	# -------------------------------- LOG CONDITIONALS TO WANDB TABLES ------------------------------------
#
# 	cll_table.add_data(train_attack_type, perturbations, standard_mean_cll, standard_std_cll,
# 					   ls1_mean_cll, ls1_std_cll, ls3_mean_cll, ls3_std_cll, ls5_mean_cll, ls5_std_cll,
# 					   rls1_mean_cll, rls1_std_cll, rls3_mean_cll, rls3_std_cll, rls5_mean_cll,
# 					   rls5_std_cll,
# 					   av_mean_cll_dict[1][evidence_percentage], av_std_cll_dict[1][evidence_percentage],
# 					   av_mean_cll_dict[3][evidence_percentage], av_std_cll_dict[3][evidence_percentage],
# 					   av_mean_cll_dict[5][evidence_percentage], av_std_cll_dict[5][evidence_percentage],
# 					   w1_mean_cll, w1_std_cll, w3_mean_cll, w3_std_cll, w5_mean_cll, w5_std_cll)
#
# 	dataset_distribution_results[str(evidence_percentage)] = dataset_distribution_evidence_results
# 	dataset_results[str(einet_args[NUM_INPUT_DISTRIBUTIONS])] = dataset_distribution_results


# 	results[dataset_name] = dataset_results
# 	dictionary_to_file(dataset_name, dataset_results, run_id, train_attack_type, perturbations, is_adv=is_adv,
# 					   is_einet=True)
# 	pretty_print_dictionary(dataset_results)
# pretty_print_dictionary(results)


if __name__ == '__main__':
	for dataset_name in [MNIST, FASHION_MNIST]:
		for perturbation in CONTINUOUS_PERTURBATIONS:
			if perturbation == 0.:
				TRAIN_ATTACKS = [CLEAN]
			else:
				TRAIN_ATTACKS = [FGSM, PGD]

			for train_attack_type in TRAIN_ATTACKS:
				evaluation_message(
					"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
																						  train_attack_type))
				test_standard_spn(run_id=172, specific_datasets=dataset_name, is_adv=True,
								  train_attack_type=train_attack_type, perturbations=perturbation)

# for dataset_name in DEBD_DATASETS:
# 	for perturbation in PERTURBATIONS:
# 		if perturbation == 0:
# 			TRAIN_ATTACKS = [CLEAN]
# 		else:
# 			TRAIN_ATTACKS = [LOCAL_SEARCH, RESTRICTED_LOCAL_SEARCH]
#
# 		for train_attack_type in TRAIN_ATTACKS:
# 			evaluation_message(
# 				"Logging values for {}, perturbation {}, train attack type {}".format(dataset_name, perturbation,
# 																					  train_attack_type))
# 			is_adv = train_attack_type in [LOCAL_SEARCH, RESTRICTED_LOCAL_SEARCH]
#
# 			test_standard_spn(run_id=1921, specific_datasets=dataset_name, is_adv=True,
# 							  train_attack_type=train_attack_type, perturbations=perturbation)

# dataset_wandb_tables = fetch_wandb_table(dataset_name)
# ll_table = dataset_wandb_tables[LOGLIKELIHOOD_TABLE]
# cll_tables = dataset_wandb_tables[CONDITIONAL_LOGLIKELIHOOD_TABLES]
#
# run1.log({"{}-LL".format(dataset_name): ll_table})
# for evidence_percentage in EVIDENCE_PERCENTAGES:
# 	cll_ev_table = cll_tables[evidence_percentage]
# 	run1.log({"{}-CLL-{}".format(dataset_name, evidence_percentage): cll_ev_table})
