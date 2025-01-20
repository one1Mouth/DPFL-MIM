

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms

from generate_server_testset import generate_server_testset
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)
dir_path = "MNIST/"


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, class_per_client=2, dir_alpha=0.1,
                     need_server_testset=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, dir_alpha):
        return

    # # FIX HTTP Error 403: Forbidden
    # from six.moves import urllib
    # opener = urllib.request.build_opener()
    # opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # urllib.request.install_opener(opener)

    # Get MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=dir_path + "rawdata", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes,
                                    niid, balance, partition, class_per_client=class_per_client, alpha=dir_alpha)
    train_data, test_data = split_data(X, y)

    if need_server_testset:
        generate_server_testset(test_data, test_path)

    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
              statistic, niid, balance, partition, dir_alpha, class_per_client)


if __name__ == "__main__":
    # niid = True if sys.argv[1] == "noniid" else False
    # balance = True if sys.argv[2] == "balance" else False
    # partition = sys.argv[3] if sys.argv[3] != "-" else None

    # Dir(0.1):  niid = True, balance = False, partition = 'dir', dir_alpha = 0.1
    # Pat(2):    niid = True, balance = True,  partition = 'pat', class_per_client = 2
    # IID:       niid = False,balance = True,  partition = 'pat',

    niid = True
    # niid = False
    balance = False
    # balance = True
    partition = 'dir'  # 狄利克雷 balance = False
    # partition = 'pat'  # 分片 balance = Ture
    need_server_testset = True
    num_clients = 500
    dir_alpha = 0.3  # Dir参数，如果是IID/Pat，这个参数无所谓
    class_per_client = 2  # Pat参数，如果是IID/Dir，这个参数无所谓

    generate_dataset(dir_path, num_clients, niid, balance, partition, class_per_client, dir_alpha, need_server_testset)
