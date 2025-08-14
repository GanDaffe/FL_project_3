import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy
import torch
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST, CIFAR100
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import SGD
from sklearn.metrics import f1_score, recall_score, precision_score

from support_function_and_class_for_FL import NTD_Loss

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist


def load_data(dataset: str):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)
    
    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
        testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)
    
    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = FashionMNIST(root='data', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='data', train=True, download=True, transform=transform)

    elif dataset == "cifar100":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])

        trainset = CIFAR100("data", train=True, download=True, transform=train_transform)
        testset = CIFAR100("data", train=False, download=True, transform=test_transform)
    return trainset, testset


def partition_data(trainset, num_clients: int, ratio: float, alpha: float, beta: float, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    classes = trainset.classes
    client_size = int(len(trainset)/num_clients)
    label_size = int(len(trainset)/len(classes))
    data = list(map(lambda x: (trainset[x][1], x), range(len(trainset))))
    data.sort()
    data = list(map(lambda x: data[x][1], range(len(data))))
    data = [data[i*label_size:(i+1)*label_size] for i in range(len(classes))]

    ids = [[] for _ in range(num_clients)]
    label_dist = []
    labels = list(range(len(classes)))

    for i in range(num_clients):
        concentration = torch.ones(len(labels))*alpha if i < num_clients*ratio else torch.ones(len(labels))*beta
        dist = Dirichlet(concentration).sample()
        for _ in range(client_size):
            label = random.choices(labels, dist)[0]
            id = random.choices(data[label])[0]
            ids[i].append(id)
            data[label].remove(id)

            if len(data[label]) == 0:
                dist = renormalize(dist, labels, label)
                labels.remove(label)

        counter = Counter(list(map(lambda x: trainset[x][1], ids[i])))
        label_dist.append({classes[i]: counter.get(i) for i in range(len(classes))})

    return ids, label_dist

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy

def train(net, trainloader, learning_rate: float, proximal_mu: float = None, epochs: int = 1, use_ntd_loss: bool = False, tau = None, beta = None):

    if use_ntd_loss:
        last_layer = list(net.modules())[-1]
        num_classes = last_layer.out_features
        criterion = NTD_Loss(num_classes=num_classes, tau=tau, beta=beta)
    else: 
        criterion = nn.CrossEntropyLoss()

    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train() 
    global_params = [p.detach().clone().to(DEVICE) for p in net.parameters()]
    
    running_loss, running_corrects = 0.0, 0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            
            loss = criterion(outputs, labels)
            if proximal_mu is not None:
                prox_term = 0.0
                for local_w, global_w in zip(net.parameters(), global_params):
                    prox_term += torch.square((local_w - global_w).norm(2))

                loss = loss + (proximal_mu / 2) * prox_term

            
            loss.backward()
            optimizer.step()
            
            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += (preds == labels).sum().item()
    
    dataset_size = len(trainloader.sampler)
    return {'loss': running_loss / dataset_size, 'accuracy': running_corrects / dataset_size}



def test(net, testloader):
    criterion = nn.CrossEntropyLoss()
    corrects, loss = 0, 0.0
    all_labels = []
    all_preds = []
    net.to(DEVICE)
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images).to(DEVICE)
            predicted = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item() * images.shape[0]
            corrects += torch.sum(predicted == labels).item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)
    f1 = f1_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    return loss, accuracy, f1, recall, precision


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)
   
def partition_data_sharding(trainset, num_clients: int, shards_per_client: int = 2):
    """
    Partition the dataset using sharding strategy. Each client gets a custom number of shards.
    """
    classes = trainset.classes
    num_classes = len(classes)
    data = [(trainset[i][1], i) for i in range(len(trainset))]
    data.sort(key=lambda x: x[0])  # Sort by label

    # Group indices by class
    class_data = [[] for _ in range(num_classes)]
    for label, idx in data:
        class_data[label].append(idx)

    # Flatten all data into shards
    num_shards = num_clients * shards_per_client
    shards = []
    shard_size = len(trainset) // num_shards
    all_indices = sum(class_data, [])

    for i in range(num_shards):
        shard = all_indices[i * shard_size: (i + 1) * shard_size]
        shards.append(shard)

    # Shuffle and assign shards to clients
    random.shuffle(shards)
    ids = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        for j in range(shards_per_client):
            ids[i].extend(shards[i * shards_per_client + j])

    # Calculate label distribution per client
    label_dist = []
    for i in range(num_clients):
        counter = Counter([trainset[idx][1] for idx in ids[i]])
        label_dist.append({classes[c]: counter.get(c, 0) for c in range(num_classes)})

    return ids, label_dist
