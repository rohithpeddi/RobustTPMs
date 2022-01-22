MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"

N_FEATURES = 'n_features'
OUT_CLASSES = 'out_classes'  # The number of classes
DEPTH = 'depth'  # The region graph's depth
NUM_REPETITIONS = 'num_repetitions'  # The region graph's number of repetitions
NUM_INPUT_DISTRIBUTIONS = 'num_input_distributions'
NUM_SUMS = "num_sums"
GENERATIVE = 'generative'
CONDITIONAL = 'conditional'

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

EINET_MODEL_DIRECTORY = "checkpoints/einet"
WEIGHTED_EINET_MODEL_DIRECTORY = "checkpoints/weighted_einet"

TRAIN_DATASET = "train_dataset"
TEST_DATASET = "test_dataset"
VALID_DATASET = "valid_dataset"

MODEL_DIRECTORY = "checkpoints/models"
MNIST_MODEL_DIRECTORY = "checkpoints/models/mnist"
DEBD_MODEL_DIRECTORY = "checkpoints/models/DEBD"
BINARY_MNIST_MODEL_DIRECTORY = "checkpoints/models/binary_mnist"

SAMPLES_DIRECTORY = "samples"
CONDITIONAL_SAMPLES_DIRECTORY = "conditional_samples"

GENERATIVE_NUM_CLASSES = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28

EINET_MAX_NUM_EPOCHS = 30
MAX_NUM_EPOCHS = 400
DEFAULT_EINET_MAX_TRAIN = 50

EARLY_STOPPING_DELTA = 1e-2
DEFAULT_PATIENCE = 30
DEFAULT_EINET_PATIENCE = 4
EARLY_STOPPING_FILE = 'checkpoint.pt'

DEFAULT_LEAF_DROPOUT = 0.2
DEFAULT_SUM_DROPOUT = 0.2
DEFAULT_NUM_REPETITIONS = 50
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_TRAIN_BATCH_SIZE = 100
DEFAULT_EVAL_BATCH_SIZE = 100

BINARY_MNIST_THRESHOLD = 0.7
BINARY_MNIST_HAMMING_THRESHOLD = 7

BINARY_DEBD_THRESHOLD = 0.7
BINARY_DEBD_HAMMING_THRESHOLD = 1

DEFAULT_GENERATIVE_NUM_CLASSES = 1

AUGMENTED_DATA_WEIGHT_PARAMETER = 0.5


TRAIN_BATCH_SIZE = 50
EVAL_BATCH_SIZE = 50

DEFAULT_DEPTH = 3
NUM_CLUSTERS = 10

# NUM_INPUT_DISTRIBUTIONS_LIST = [10, 20, 30, 40, 50]
NUM_INPUT_DISTRIBUTIONS_LIST = [10]
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3]
EVIDENCE_PERCENTAGES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

CHECKPOINT_DIRECTORY = "../../checkpoints"

MNIST_CHANNELS = 1
MAX_ITER = 50
LAMBDA = 3.

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

# DEBD_DATASETS = [ 'baudio', 'jester', 'bnetflix', 'tretail', 'dna', 'kosarek', 'msweb', 'book', 'tmovie', 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad', 'msnbc']

DEBD_DATASETS = ['plants', 'accidents', 'nltcs', 'msnbc', 'kdd', 'baudio',
				 'jester', 'bnetflix', 'tretail', 'pumsb_star',
				 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
				 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']

CONTINUOUS_DATASETS = [MNIST]
DISCRETE_DATASETS = ['plants', 'accidents', 'nltcs', 'msnbc', 'kdd', 'baudio',
					 'jester', 'bnetflix', 'tretail', 'pumsb_star',
					 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
					 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad', BINARY_MNIST]

#################################################################################################

TRAIN_NEURAL_NET_MAX_NUM_EPOCHS = 100

DATA_DIRECTORY = "data/"
MNIST_NET_PATH = "checkpoints/neural/mnist/"
MNIST_NET_FILE = "mnist_cnn.pt"
BINARY_MNIST_NET_FILE = "binary_mnist_cnn.pt"

BINARY_MNIST_NET_PATH = "checkpoints/neural/binary_mnist"
DEBD_NET_PATH = "checkpoints/neural/DEBD"

MANUAL_SEED = 999

##############################################################################################


STRUCTURES = [POON_DOMINGOS, BINARY_TREES]

STRUCTURE_DIRECTORY = "checkpoints/structure"

DEFAULT_ONLINE_EM_STEPSIZE = 0.05
DEFAULT_ONLINE_EM_FREQUENCY = 50

DEFAULT_PD_NUM_PIECES = [4]


EXPERIMENTS_DIRECTORY = "checkpoints/experiments"


DEFAULT_DE_POPULATION_SIZE = 400
DEFAULT_DE_MAX_ITERATIONS = 75

##############################################################################################

# Attack Types

CLEAN = "clean"
LOCAL_SEARCH = "local_search"
LOCAL_RESTRICTED_SEARCH = "local_restricted_search"
NEURAL_NET = "neural_net"
EVOLUTIONARY = "evolutionary"
GRADIENT = "gradient"
