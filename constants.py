MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"
IMDB = 'imdb'
WINE = 'wine'
CIFAR_10 = "cifar_10"

N_FEATURES = 'n_features'
OUT_CLASSES = 'out_classes'  # The number of classes
DEPTH = 'depth'  # The region graph's depth
NUM_REPETITIONS = 'num_repetitions'  # The region graph's number of repetitions
NUM_INPUT_DISTRIBUTIONS = 'num_input_distributions'
NUM_SUMS = "num_sums"
GENERATIVE = 'generative'
DISCRIMINATIVE = 'discriminative'
CONDITIONAL = 'conditional'
BATCH_SIZE = 'batch_size'
NUM_CLASSES = 'num_classes'
USE_EM = "use_em"
NUM_EPOCHS = 'num_epochs'
LEARNING_RATE = 'learning_rate'
PATIENCE = 'patience'
BATCHED_LEAVES = 'batched_leaves'
SUM_CHANNELS = 'sum_channels'
NUM_POOLING = 'num_pooling'
IN_DROPOUT = 'in_dropout'
SUM_DROPOUT = 'sum_dropout'

LOGLIKELIHOOD_TABLE = "ll_table"
CONDITIONAL_LOGLIKELIHOOD_TABLES = "cll_tables"

HEIGHT = 'height'
WIDTH = 'width'
PD_NUM_PIECES = 'pd_num_pieces'
NUM_VAR = 'num_var'
POON_DOMINGOS = "poon_domingos"
BINARY_TREES = "binary_trees"

EXPONENTIAL_FAMILY = 'exponential_family'
EXPONENTIAL_FAMILY_ARGS = 'exponential_family_args'
ONLINE_EM_FREQUENCY = 'online_em_frequency'
ONLINE_EM_STEPSIZE = 'online_em_stepsize'

CLEAN_EINET_MODEL_DIRECTORY = "checkpoints/einet/M"
LS_EINET_MODEL_DIRECTORY = "checkpoints/einet/AMLS"
RLS_EINET_MODEL_DIRECTORY = "checkpoints/einet/AMRLS"
NN_EINET_MODEL_DIRECTORY = "checkpoints/einet/AMNN"
EINET_MODEL_DIRECTORY = "checkpoints/einet"
WEIGHTED_EINET_MODEL_DIRECTORY = "checkpoints/weighted_einet"

TRAIN_DATASET = "train_dataset"
TEST_DATASET = "test_dataset"
VALID_DATASET = "valid_dataset"

MODEL_DIRECTORY = "checkpoints/models"
MNIST_MODEL_DIRECTORY = "checkpoints/models/mnist"
CIFAR_10_MODEL_DIRECTORY = "checkpoints/models/cifar_10"
DEBD_MODEL_DIRECTORY = "checkpoints/models/DEBD"
BINARY_MNIST_MODEL_DIRECTORY = "checkpoints/models/binary_mnist"

SAMPLES_DIRECTORY = "samples"
CONDITIONAL_SAMPLES_DIRECTORY = "conditional_samples"

#########################################################################################################

GENERATIVE_NUM_CLASSES = 1

MNIST_NUM_CLASSES = 10
MNIST_CHANNELS = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28

FASHION_MNIST_NUM_CLASSES = 10
FASHION_MNIST_CHANNELS = 1
FASHION_MNIST_HEIGHT = 28
FASHION_MNIST_WIDTH = 28

CIFAR_10_NUM_CLASSES = 10
CIFAR_10_CHANNELS = 3
CIFAR_10_HEIGHT = 32
CIFAR_10_WIDTH = 32

#########################################################################################################


NUM_CLUSTERS = 10

DEFAULT_LEAF_DROPOUT = 0.2
DEFAULT_SUM_DROPOUT = 0.2

BINARY_MNIST_THRESHOLD = 0.7
BINARY_MNIST_HAMMING_THRESHOLD = 7

BINARY_DEBD_THRESHOLD = 0.7

DEFAULT_GENERATIVE_NUM_CLASSES = 1

AUGMENTED_DATA_WEIGHT_PARAMETER = 0.5

TRAIN_BATCH_SIZE = 50
EVAL_BATCH_SIZE = 50

CONTINUOUS_NUM_INPUT_DISTRIBUTIONS_LIST = [20, 30]
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3]
EVIDENCE_PERCENTAGES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CHECKPOINT_DIRECTORY = "../../checkpoints"
EINET_MAX_NUM_EPOCHS = 30

MAX_ITER = 50
LAMBDA = 3.
DEFAULT_PATIENCE = 30
MNIST_NET_DIRECTORY = "checkpoints/mnist"
DEFAULT_SPARSEFOOL_ATTACK_BATCH_SIZE = 1

DATA_MNIST_ADV_SPARSEFOOL = "data/MNIST/augmented/sparsefool"

DATA_DEBD_DIRECTORY = "data/DEBD/datasets"
CLUSTERED_DATA_DEBD_DIRECTORY = "data/DEBD/clustered"

EINET_DEBD_RESULTS_DIRECTORY = "results/einet/DEBD"
EINET_BINARY_MNIST_RESULTS_DIRECTORY = "results/einet/binary_mnist"
EINET_MNIST_RESULTS_DIRECTORY = "results/einet/mnist"

RATSPN_DEBD_RESULTS_DIRECTORY = "results/ratspn/DEBD"
RATSPN_BINARY_MNIST_RESULTS_DIRECTORY = "results/ratspn/binary_mnist"
RATSPN_MNIST_RESULTS_DIRECTORY = "results/ratspn/mnist"

# DEBD_DATASETS = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio',
# 				 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek']

DEBD_DATASETS = ['plants', 'baudio',
 				 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star', 'dna', 'kosarek', 'msweb', 'book',
				 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc']

SMALL_VARIABLE_DATASETS = ['nltcs', 'msnbc', 'kdd', 'plants', 'baudio',
						   'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star']

LARGE_VARIABLE_DATASETS = ['dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']

CONTINUOUS_DATASETS = [MNIST, FASHION_MNIST]
DISCRETE_DATASETS = ['plants', 'nltcs', 'msnbc', 'kdd', 'baudio',
					 'jester', 'bnetflix', 'accidents', 'tretail', 'pumsb_star',
					 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
					 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad', BINARY_MNIST]

#################################################################################################

TRAIN_NEURAL_NET_MAX_NUM_EPOCHS = 100
MANUAL_SEED = 999

DATA_DIRECTORY = "data/"
MNIST_NET_PATH = "checkpoints/neural/mnist/"
FASHION_MNIST_NET_PATH = "checkpoints/neural/fashion_mnist/"
EMNIST_NET_PATH = "checkpoints/neural/emnist/"
QMNIST_NET_PATH = "checkpoints/neural/qmnist/"
KMNIST_NET_PATH = "checkpoints/neural/kmnist/"

MNIST_NET_FILE = "mnist_cnn.pt"
FASHION_MNIST_NET_FILE = "fashion_mnist_cnn.pt"
EMNIST_NET_FILE = "emnist_cnn.pt"
QMNIST_NET_FILE = "qmnist_cnn.pt"
KMNIST_NET_FILE = "kmnist_cnn.pt"
BINARY_MNIST_NET_FILE = "binary_mnist_cnn.pt"

BINARY_MNIST_NET_PATH = "checkpoints/neural/binary_mnist"
DEBD_NET_PATH = "checkpoints/neural/DEBD"

##############################################################################################


EXPERIMENTS_DIRECTORY = "checkpoints/experiments"
STRUCTURE_DIRECTORY = "structure"

DEFAULT_DE_POPULATION_SIZE = 400
DEFAULT_DE_MAX_ITERATIONS = 75

##############################################################################################

# Attack Types

CLEAN = "clean"
LOCAL_SEARCH = "local_search"
RESTRICTED_LOCAL_SEARCH = "local_restricted_search"
NEURAL_NET = "neural_net"
EVOLUTIONARY = "evolutionary"
GRADIENT = "gradient"
AVERAGE = "average"
WEAKER_MODEL = "weaker_model"

##############################################################################################
BINARY_DEBD_HAMMING_THRESHOLD = 5

STRUCTURES = [POON_DOMINGOS, BINARY_TREES]

DEFAULT_PD_NUM_PIECES = [8]

DEFAULT_ONLINE_EM_STEPSIZE = 0.05
DEFAULT_ONLINE_EM_FREQUENCY = 1

MAX_NUM_EPOCHS = 400

EARLY_STOPPING_DELTA = 1e-3
DEFAULT_EINET_PATIENCE = 1
EARLY_STOPPING_FILE = 'checkpoint.pt'

DEFAULT_NUM_REPETITIONS = 50
DEFAULT_LEARNING_RATE = 5e-3
DEFAULT_TRAIN_BATCH_SIZE = 100
DEFAULT_EVAL_BATCH_SIZE = 100

DEFAULT_DEPTH = 3

# NUM_INPUT_DISTRIBUTIONS_LIST = [10, 20, 30, 40, 50]
NUM_INPUT_DISTRIBUTIONS_LIST = [10]
PERTURBATIONS = [0, 1, 3, 5]
DEFAULT_AVERAGE_REPEAT_SIZE = 100

NEURAL_NETWORK_ATTACK_MODEL_SUB_DIRECTORY = "AMNN"
LOCAL_SEARCH_ATTACK_MODEL_SUB_DIRECTORY = "AMLS"
LOCAL_RESTRICTED_SEARCH_ATTACK_MODEL_SUB_DIRECTORY = "AMRLS"
CLEAN_MODEL_SUB_DIRECTORY = "M"

##############################################################################################

POON_DOMINGOS_GRID = [7]

DEFAULT_NUM_SUMS = 10
DEFAULT_NUM_INPUT_DISTRIBUTIONS = 10

PGD = "pgd"
FGSM = "fgsm"

DGCSPN = "dgcspn"
NET = "nn"
CW = "cw"
PGDL2 = "pgdl2"
SQUARE = "square"
DEEPFOOL = "deepfool"
SPARSEFOOL = "sparsefool"
FAB = "fab"
ONE_PIXEL = "one_pixel"
