import argparse
import json
import numpy as np
import torch
import torchattacks
import os
import wandb

import base_spn as SPN
from constants import *
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from neural_models.MNet import MNet
from train_neural_models import test_neural

############################################################################


wandb.init(project="my-test-project", entity="rohithpeddi")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################################################

def evaluation_message(message):
	print("\n")
	print("-----------------------------------------------------------------------------")
	print("#" + message)
	print("-----------------------------------------------------------------------------")


# def generate_adv_samples_transfer(test_x, test_labels, attack):
# 	test_dataset = TensorDataset(test_x, test_labels)
# 	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=100)
# 	data_loader = tqdm(
# 		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
# 		desc='Test', unit='batch'
# 	)
#
# 	originals = []
# 	adversaries = []
# 	adv_test = []
# 	for inputs, targets in data_loader:
# 		inputs, targets = inputs.to(device), targets.to(device)
# 		inputs = inputs * 0.3081
# 		inputs = inputs + 0.1307
# 		adv_inputs = attack(inputs, targets)
# 		adv_inputs = adv_inputs - 0.1307
# 		adv_inputs = adv_inputs / 0.3081
# 		adv_test.append(adv_inputs)
#
# 		if len(adversaries) < 1:
# 			adversaries = (adv_inputs[0:5, :, :, :]).detach().clone()
# 			originals = (inputs[0:5, :, :, :]).detach().clone()
#
# 	count = 0
# 	plt.figure(figsize=(8, 10))
# 	for id in range(len(adversaries)):
# 		adv_ex = adversaries[id][0].cpu().numpy()
# 		ori_ex = originals[id][0].cpu().numpy()
# 		print(np.sum(np.abs(adv_ex-ori_ex)))
# 		count += 1
# 		plt.subplot(2, len(adversaries), count)
# 		plt.imshow(adv_ex, cmap="gray")
# 		count += 1
# 		plt.subplot(2, len(adversaries), count)
# 		plt.imshow(ori_ex, cmap="gray")
# 	plt.tight_layout()
# 	plt.show()
#
# 	return torch.cat(adv_test)

def generate_adv_samples_transfer(test_x, test_labels, attack, attack_type, attack_model):
	test_dataset = TensorDataset(test_x, test_labels)
	test_loader = DataLoader(test_dataset, shuffle=False, batch_size=50)
	data_loader = tqdm(
		test_loader, leave=False, bar_format='{l_bar}{bar:24}{r_bar}',
		desc='Test', unit='batch'
	)

	originals = []
	adversaries = []
	adv_test = []
	for inputs, targets in data_loader:
		inputs, targets = inputs.to(device), targets.to(device)
		# inputs = inputs * 0.3081
		# inputs = inputs + 0.1307
		adv_inputs = attack(inputs, targets)
		# adv_inputs = adv_inputs - 0.1307
		# adv_inputs = adv_inputs / 0.3081
		adv_test.append(adv_inputs)

		if len(adversaries) < 1:
			adversaries = (adv_inputs[0:5, :, :, :]).detach().clone()
			originals = (test_x[0:5, :, :, :]).detach().clone()

	count = 0
	plt.figure(figsize=(8, 10))
	for id in range(len(adversaries)):
		adv_ex = adversaries[id][0].cpu().numpy()
		ori_ex = originals[id][0].cpu().numpy()
		# print(np.sum(np.abs(adv_ex-ori_ex)))
		count += 1
		plt.subplot(2, len(adversaries), count)
		plt.title("Adv:{},M:{}".format(attack_type, attack_model))
		plt.imshow(adv_ex, cmap="gray")
		count += 1
		plt.subplot(2, len(adversaries), count)
		plt.title("Clean image")
		plt.imshow(ori_ex, cmap="gray")
	plt.tight_layout()
	plt.show()

	return torch.cat(adv_test)


def test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type):
	if attack_type == PGD:
		attack_net = torchattacks.PGD(trained_net, eps=16 / 255, alpha=8 / 255, steps=40, random_start=True)
	elif attack_type == FGSM:
		attack_net = torchattacks.FGSM(trained_net, eps=0.1)
	elif attack_type == CW:
		attack_net = torchattacks.CW(trained_net, c=2, kappa=0, steps=1000, lr=0.01)
	elif attack_type == PGDL2:
		attack_net = torchattacks.PGDL2(trained_net, eps=1.0, alpha=0.2, steps=40, random_start=True)
	elif attack_type == SQUARE:
		attack_net = torchattacks.Square(trained_net, norm='Linf', n_queries=5000, n_restarts=1, eps=16 / 255,
										 p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)
	elif attack_type == DEEPFOOL:
		attack_net = torchattacks.DeepFool(trained_net, steps=50, overshoot=0.02)
	elif attack_type == SPARSEFOOL:
		attack_net = torchattacks.SparseFool(trained_net, steps=20, lam=3, overshoot=0.02)
	elif attack_type == FAB:
		attack_net = torchattacks.FAB(trained_net, norm='Linf', steps=100, eps=None, n_restarts=1, alpha_max=0.1,
									  eta=1.05,
									  beta=0.9, verbose=False, seed=0, targeted=False, n_classes=10)
	elif attack_type == ONE_PIXEL:
		attack_net = torchattacks.OnePixel(trained_net, pixels=1, steps=75, popsize=400, inf_batch=128)

	# Generating adversarial samples using neural network
	adv_test_x_neural = generate_adv_samples_transfer(test_x, test_labels, attack_net, attack_type, attack_model=NET)

	nll, metrics = SPN.test_spn(trained_spn, adv_test_x_neural, spn_args, test_labels)
	evaluation_message(" Adv data statistics attack : {} ".format(attack_type))
	print('Test NLL: {:.4f}'.format(nll))
	print('Test Accuracy: {}'.format(metrics['accuracy']))

	test_loader = DataLoader(TensorDataset(adv_test_x_neural, test_labels))
	test_neural(trained_net, test_loader)


def test_mnist_continuous(args):
	dataset_name = args.dataset_name
	run_id = args.run_id

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

	spn_args = dict()
	spn_args[N_FEATURES] = (MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH)
	spn_args[OUT_CLASSES] = MNIST_NUM_CLASSES
	spn_args[BATCH_SIZE] = args.batch_size
	spn_args[NUM_EPOCHS] = 65
	spn_args[LEARNING_RATE] = 5e-3
	spn_args[PATIENCE] = 30
	spn_args[BATCHED_LEAVES] = 30
	spn_args[SUM_CHANNELS] = 60
	spn_args[SUM_DROPOUT] = 0.2
	spn_args[IN_DROPOUT] = 0.2
	spn_args[NUM_POOLING] = 2

	trained_spn = SPN.load_pretrained_ratspn(run_id, dataset_name, spn_args)
	if trained_spn is None:
		dgcspn = SPN.load_spn(dataset_name, spn_args)
		trained_spn = SPN.train_spn(run_id, dataset_name, dgcspn, train_x, valid_x, test_x, spn_args,
									train_labels=train_labels, valid_labels=valid_labels, test_labels=test_labels)

	evaluation_message("Loading pretrained neural net for adversarial example generation")
	trained_net = MNet().to(device)
	trained_net.load_state_dict(torch.load(os.path.join(MNIST_NET_PATH, MNIST_NET_FILE)))

	nll, metrics = SPN.test_spn(trained_spn, test_x, spn_args, test_labels)

	evaluation_message(" Clean data statistics ")
	print('Test NLL: {:.4f}'.format(nll))
	print('Test Accuracy: {}'.format(metrics['accuracy']))
	test_loader = DataLoader(TensorDataset(test_x, test_labels))
	test_neural(trained_net, test_loader)

	# # 1. FGSM
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=FGSM)
	#
	# # 2. PGD
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=PGD)
	#
	# # 3. CW
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=CW)
	#
	# # 4. RFGSM
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=PGDL2)
	#
	# # 5. SQUARE
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=SQUARE)

	# # 6. DEEPFOOL
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=DEEPFOOL)

	# 7. SPARSEFOOL
	test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=SPARSEFOOL)
	#
	# # 8. FAB
	# test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=FAB)

	# 9. ONE PIXEL
	test_attack(test_x, test_labels, spn_args, trained_spn, trained_net, attack_type=ONE_PIXEL)


def test_fashion_mnist_continuous(args):
	dataset_name = args.dataset_name
	run_id = args.run_id

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

	spn_args = dict()
	spn_args[N_FEATURES] = (FASHION_MNIST_CHANNELS, FASHION_MNIST_HEIGHT, FASHION_MNIST_WIDTH)
	spn_args[OUT_CLASSES] = MNIST_NUM_CLASSES
	spn_args[BATCH_SIZE] = args.batch_size
	spn_args[NUM_EPOCHS] = 100
	spn_args[LEARNING_RATE] = 5e-3
	spn_args[PATIENCE] = 30
	spn_args[BATCHED_LEAVES] = 20
	spn_args[SUM_CHANNELS] = 40
	spn_args[SUM_DROPOUT] = 0.2
	spn_args[IN_DROPOUT] = 0.2
	spn_args[NUM_POOLING] = 2

	dgcspn = SPN.load_spn(dataset_name, spn_args)

	trained_spn = SPN.train_spn(1, dataset_name, dgcspn, train_x, valid_x, test_x, spn_args, train_labels=train_labels,
								valid_labels=valid_labels, test_labels=test_labels)

	nll, metrics = SPN.test_spn(trained_spn, test_x, spn_args, test_labels)

	evaluation_message(" Clean data statistics ")

	print('Test NLL: {:.4f}'.format(nll))
	metrics = json.loads(json.dumps(metrics), parse_float=lambda x: round(float(x), 6))
	print('Test Metrics: {}'.format(json.dumps(metrics, indent=4)))


def test_cifar_10_continuous(args):
	dataset_name = args.dataset_name
	run_id = args.run_id

	train_x, valid_x, test_x, train_labels, valid_labels, test_labels = SPN.load_dataset(dataset_name)

	spn_args = dict()
	spn_args[N_FEATURES] = (CIFAR_10_CHANNELS, CIFAR_10_HEIGHT, CIFAR_10_WIDTH)
	spn_args[OUT_CLASSES] = CIFAR_10_NUM_CLASSES
	spn_args[BATCH_SIZE] = args.batch_size
	spn_args[NUM_EPOCHS] = 100
	spn_args[LEARNING_RATE] = 1e-3
	spn_args[PATIENCE] = 30
	spn_args[BATCHED_LEAVES] = 50
	spn_args[SUM_CHANNELS] = 100
	spn_args[SUM_DROPOUT] = 0.2
	spn_args[IN_DROPOUT] = 0.2
	spn_args[NUM_POOLING] = 2

	dgcspn = SPN.load_spn(dataset_name, spn_args)

	trained_spn = SPN.train_spn(run_id, dataset_name, dgcspn, train_x, valid_x, test_x, spn_args,
								train_labels=train_labels,
								valid_labels=valid_labels, test_labels=test_labels)

	nll, metrics = SPN.test_spn(trained_spn, test_x, spn_args, test_labels)

	evaluation_message(" Clean data statistics ")

	print('Test NLL: {:.4f}'.format(nll))
	metrics = json.loads(json.dumps(metrics), parse_float=lambda x: round(float(x), 6))
	print('Test Metrics: {}'.format(json.dumps(metrics, indent=4)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--run_id', type=int, default=27, help="")
	parser.add_argument('--batch_size', type=int, default=100, help="")
	parser.add_argument('--dataset_name', type=str, required=True, help="dataset name")
	ARGS = parser.parse_args()
	print(ARGS)

	if ARGS.dataset_name == MNIST:
		test_mnist_continuous(ARGS)
	elif ARGS.dataset_name == FASHION_MNIST:
		test_fashion_mnist_continuous(ARGS)
	elif ARGS.dataset_name == CIFAR_10:
		test_cifar_10_continuous(ARGS)
