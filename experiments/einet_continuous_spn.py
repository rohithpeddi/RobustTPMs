import argparse
import torch

import einet_base_spn as SPN
from EinsumNetwork.ExponentialFamilyArray import NormalArray, CategoricalArray
from constants import *
from utils import pretty_print_dictionary, dictionary_to_file

############################################################################


device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def test_mnist_continuous(args):
	dataset_name = args.dataset_name
	run_id = args.run_id
	structure = POON_DOMINGOS

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

	exponential_family = NormalArray
	exponential_family_args = SPN.generate_exponential_family_args(exponential_family, dataset_name)

	structure_args = dict()
	structure_args[HEIGHT] = MNIST_HEIGHT
	structure_args[WIDTH] = MNIST_WIDTH
	structure_args[PD_NUM_PIECES] = POON_DOMINGOS_GRID
	graph = SPN.load_structure(run_id, structure, dataset_name, structure_args)

	einet_args = dict()
	einet_args[NUM_VAR] = train_x.shape[1]
	einet_args[USE_EM] = False
	einet_args[NUM_CLASSES] = MNIST_NUM_CLASSES
	einet_args[NUM_SUMS] = DEFAULT_NUM_SUMS
	einet_args[NUM_INPUT_DISTRIBUTIONS] = DEFAULT_NUM_INPUT_DISTRIBUTIONS
	einet_args[EXPONENTIAL_FAMILY] = exponential_family
	einet_args[EXPONENTIAL_FAMILY_ARGS] = exponential_family_args
	einet_args[ONLINE_EM_FREQUENCY] = DEFAULT_ONLINE_EM_FREQUENCY
	einet_args[ONLINE_EM_STEPSIZE] = DEFAULT_ONLINE_EM_STEPSIZE
	einet_args[NUM_REPETITIONS] = DEFAULT_NUM_REPETITIONS
	einet_args[BATCH_SIZE] = 100

	einet = SPN.load_einet(run_id, structure, dataset_name, einet_args, graph)

	trained_einet = SPN.train_discriminative_einet(run_id, structure, dataset_name, einet, train_x, train_labels,
												   valid_x, valid_labels, test_x, test_labels, einet_args, epsilon=0,
												   attack_type=CLEAN, batch_size=einet_args[BATCH_SIZE], is_adv=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--run_id', type=int, default=91, help="")
	parser.add_argument('--dataset_name', type=str, required=True, help="dataset name")
	ARGS = parser.parse_args()
	print(ARGS)

	test_mnist_continuous(ARGS)
