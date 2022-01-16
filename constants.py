MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"

CONTINUOUS_DATASETS = [MNIST]
DISCRETE_DATASETS = [BINARY_MNIST, FASHION_MNIST]

N_FEATURES = 'n_features'
OUT_CLASSES = 'out_classes'  # The number of classes
DEPTH = 'depth'  # The region graph's depth
NUM_REPETITIONS = 'num_repetitions'  # The region graph's number of repetitions
NUM_INPUT_DISTRIBUTIONS = 'num_input_distributions'
NUM_SUMS = "num_sums"
GENERATIVE = 'generative'

MODEL_DIRECTORY = "checkpoints/models"
DEBD_MODEL_DIRECTORY = "checkpoints/models/DEBD"
SAMPLES_DIRECTORY = "samples"
CONDITIONAL_SAMPLES_DIRECTORY = "conditional_samples"

GENERATIVE_NUM_CLASSES = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28

MAX_NUM_EPOCHS = 20

DEFAULT_PATIENCE = 5
DEFAULT_LEAF_DROPOUT = 0.2
DEFAULT_SUM_DROPOUT = 0.2
DEFAULT_NUM_REPETITIONS = 10
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_TRAIN_BATCH_SIZE = 100
DEFAULT_EVAL_BATCH_SIZE = 100

NUM_CLASSES = 1
MNIST_HEIGHT = 28
MNIST_WIDTH = 28
MAX_NUM_EPOCHS = 10

TRAIN_BATCH_SIZE = 50
EVAL_BATCH_SIZE = 50

DEFAULT_DEPTH = 3
DEFAULT_NUM_REPETITIONS = 10

NUM_INPUT_DISTRIBUTIONS_LIST = [20, 30, 40, 50]
EPSILON_LIST = [0.05, 0.1, 0.2, 0.3]

MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
BINARY_MNIST = "binary_mnist"

DATA_MNIST_ADV_SPARSEFOOL = "data/MNIST/augmented/sparsefool"

DEBD_DATASETS = ['plants', 'accidents', 'nltcs', 'msnbc', 'kdd', 'baudio',
						 'jester', 'bnetflix', 'tretail', 'pumsb_star',
						 'dna', 'kosarek', 'msweb', 'book', 'tmovie',
						 'cwebkb', 'cr52', 'c20ng', 'bbc', 'ad']
