import os
import errno
import torch
import numpy as np
from PIL import Image
from neural_models.MNet import Net
from constants import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_torch_tensor(train_x, valid_x, test_x, train_labels, valid_labels, test_labels):
	train_x = torch.tensor(train_x, dtype=torch.float32, device=torch.device(device))
	valid_x = torch.tensor(valid_x, dtype=torch.float32, device=torch.device(device))
	test_x = torch.tensor(test_x, dtype=torch.float32, device=torch.device(device))

	train_labels = torch.tensor(train_labels, dtype=torch.int64, device=torch.device(device))
	valid_labels = torch.tensor(valid_labels, dtype=torch.int64, device=torch.device(device))
	test_labels = torch.tensor(test_labels, dtype=torch.int64, device=torch.device(device))

	return train_x, valid_x, test_x, train_labels, valid_labels, test_labels


def predict_labels_mnist(data):
	net = Net().to(device)
	net.load_state_dict(torch.load(os.path.join(MNIST_NET_DIRECTORY, "mnist_cnn.pt")))
	net.eval()

	return net(data)


def mkdir_p(path):
	"""Linux mkdir -p"""
	try:
		os.makedirs(path)
	except OSError as exc:  # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else:
			raise


def one_hot(x, K, dtype=torch.float):
	"""One hot encoding"""
	with torch.no_grad():
		ind = torch.zeros(x.shape + (K,), dtype=dtype, device=x.device)
		ind.scatter_(-1, x.unsqueeze(-1), 1)
		return ind


def save_image_stack(samples, num_rows, num_columns, filename, margin=5, margin_gray_val=1., frame=0,
					 frame_gray_val=0.0):
	"""Save image stack in a tiled image"""

	# for gray scale, convert to rgb
	if len(samples.shape) == 3:
		samples = np.stack((samples,) * 3, -1)

	height = samples.shape[1]
	width = samples.shape[2]

	samples -= samples.min()
	samples /= samples.max()

	img = margin_gray_val * np.ones(
		(height * num_rows + (num_rows - 1) * margin, width * num_columns + (num_columns - 1) * margin, 3))
	for h in range(num_rows):
		for w in range(num_columns):
			img[h * (height + margin):h * (height + margin) + height, w * (width + margin):w * (width + margin) + width,
			:] = samples[h * num_columns + w, :]

	framed_img = frame_gray_val * np.ones((img.shape[0] + 2 * frame, img.shape[1] + 2 * frame, 3))
	framed_img[frame:(frame + img.shape[0]), frame:(frame + img.shape[1]), :] = img

	img = Image.fromarray(np.round(framed_img * 255.).astype(np.uint8))

	img.save(filename)


def sample_matrix_categorical(p):
	"""Sample many categorical distributions represented as rows in a matrix"""
	with torch.no_grad():
		cp = torch.cumsum(p[:, 0:-1], -1)
		rand = torch.rand((cp.shape[0], 1), device=cp.device)
		rand_idx = torch.sum(rand > cp, -1).long()
		return rand_idx
