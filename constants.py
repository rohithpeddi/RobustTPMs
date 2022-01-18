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

MODEL_DIRECTORY = "checkpoints/models"
MNIST_MODEL_DIRECTORY = "checkpoints/models/mnist"
DEBD_MODEL_DIRECTORY = "checkpoints/models/DEBD"
BINARY_MNIST_MODEL_DIRECTORY = "checkpoints/models/binary_mnist"

SAMPLES_DIRECTORY = "samples"
CONDITIONAL_SAMPLES_DIRECTORY = "conditional_samples"

GENERATIVE_NUM_CLASSES = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28

MAX_NUM_EPOCHS = 200

DEFAULT_PATIENCE = 30
DEFAULT_LEAF_DROPOUT = 0.2
DEFAULT_SUM_DROPOUT = 0.2
DEFAULT_NUM_REPETITIONS = 10
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_TRAIN_BATCH_SIZE = 100
DEFAULT_EVAL_BATCH_SIZE = 100

BINARY_MNIST_THRESHOLD = 0.7
BINARY_MNIST_HAMMING_THRESHOLD = 7

NUM_CLASSES = 1

TRAIN_BATCH_SIZE = 50
EVAL_BATCH_SIZE = 50

DEFAULT_DEPTH = 3
NUM_CLUSTERS = 10

# NUM_INPUT_DISTRIBUTIONS_LIST = [20, 30, 40, 50]
NUM_INPUT_DISTRIBUTIONS_LIST = [50]
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

DEBD_DATASETS = ['pumsb_star',
				 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
				 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']

CONTINUOUS_DATASETS = [MNIST]
DISCRETE_DATASETS = ['plants', 'accidents', 'nltcs', 'msnbc', 'kdd', 'baudio',
					 'jester', 'bnetflix', 'tretail', 'pumsb_star',
					 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
					 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad', BINARY_MNIST]

#################################################################################################

TRAIN_NEURAL_NET_MAX_NUM_EPOCHS = 25

DATA_DIR = "data/"
MNIST_NET_PATH = "checkpoints/neural/mnist/"
MNIST_NET_FILE = "mnist_cnn.pt"
BINARY_MNIST_NET_FILE = "binary_mnist_cnn.pt"

BINARY_MNIST_NET_PATH = "checkpoints/neural/binary_mnist"
DEBD_NET_PATH = "checkpoints/neural/DEBD"

MANUAL_SEED = 999
