import os
import torch
import base_spn as SPN
from EinsumNetwork import EinsumNetwork
from models.Net import Net

############################################################################

MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"

CONTINUOUS_DATASETS = [MNIST]
DISCRETE_DATASETS = [BINARY_MNIST, FASHION_MNIST]

POON_DOMINGOS = "poon_domingos"
BINARY_TREES = "binary_trees"

STRUCTURES = [POON_DOMINGOS, BINARY_TREES]

STRUCTURE_DIRECTORY = "checkpoints/structure"
MODEL_DIRECTORY = "checkpoints/models"

SAMPLES_DIRECTORY = "samples"
CONDITIONAL_SAMPLES_DIRECTORY = "conditional_samples"
MNIST_MODEL_DIRECTORY = "checkpoints/mnist"

NUM_CLASSES = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28
MAX_NUM_EPOCHS = 10

TRAIN_BATCH_SIZE = 100
EVAL_BATCH_SIZE = 100
PD_NUM_PIECES = [4]

DEFAULT_DEPTH = 3
DEFAULT_NUM_REPETITIONS = 10
DEFAULT_ONLINE_EM_STEPSIZE = 0.05
DEFAULT_ONLINE_EM_FREQUENCY = 1

NUM_INPUT_DISTRIBUTIONS_LIST = [20, 30, 40, 50]
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def evaluation_message(message):
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


def load_neural_network(dataset_name):
	net = None
	if dataset_name == MNIST:
		net = Net().to(device)
		net.load_state_dict(torch.load(os.path.join(MNIST_MODEL_DIRECTORY, "mnist_cnn.pt")))
		net.eval()
	return net


def train_and_generate_samples(structure, dataset_name, net, einet_args, num_distributions, train_x, train_labels,
							   valid_x,
							   valid_labels, test_x, test_labels, print_message, epsilon):
	einet = SPN.load_einet(structure, dataset_name, einet_args)

	print_message += (" " + "training robust einet")
	evaluation_message(print_message)

	trained_einet = SPN.train_robust_spn(structure, dataset_name, net, einet, train_x, train_labels, valid_x,
										 valid_labels, test_x, test_labels, einet_args,
										 batch_size=TRAIN_BATCH_SIZE, epsilon=epsilon)

	print_message += (" " + "generating samples")
	evaluation_message(print_message)

	SPN.generate_adv_samples(trained_einet, structure, dataset_name, einet_args, epsilon)

	SPN.generate_conditional_adv_samples(einet, structure, dataset_name, einet_args, test_x, epsilon)


def test_robust_spn_continuous():
	for dataset_name in CONTINUOUS_DATASETS:

		print_message = dataset_name
		evaluation_message(print_message)

		# Load data as tensors
		train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

		if dataset_name == MNIST:

			exponential_family = EinsumNetwork.NormalArray
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
					graph = SPN.load_structure(structure, dataset_name, structure_args)
				else:
					structure_args = dict()
					structure_args['num_var'] = train_x.shape[1]
					structure_args['depth'] = DEFAULT_DEPTH
					structure_args['num_repetitions'] = DEFAULT_NUM_REPETITIONS
					graph = SPN.load_structure(structure, dataset_name, structure_args)

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

					net = load_neural_network(dataset_name)

					for epsilon in EPSILON_LIST:
						train_and_generate_samples(structure, dataset_name, net, einet_args, num_distributions, train_x,
												   train_labels, valid_x,
												   valid_labels, test_x, test_labels, print_message, epsilon)


if __name__ == '__main__':
	test_robust_spn_continuous()
