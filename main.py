import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
import numpy as np
import random
import flwr as fl
from flwr.common import ndarrays_to_parameters
import utils
from flwr.common import Context
from algo import FedAvg, FedProx, FedNTD, FedCLS, MOON, Scaffold
from model import ResNet50, CNN2, MLP, Moon_MLP
from ClientManager import ClientManager
from mutual_handler import DEFAULT_CONFIG
from return_metrics_handler import fit_handler

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

# Loading the data

algo = "fedprox" 
NUM_CLIENTS = 40
ratio = 0.2
alpha = 100
beta = 0.01
lr = 0.01
BATCH_SIZE = 10
NUM_ROUNDS = 200
current_parameters = ndarrays_to_parameters(utils.get_parameters(MLP()))
client_resources = {"num_cpus": 2, "num_gpus": 0.125} if DEVICE.type == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

trainset, testset = utils.load_data("cifar100")
ids, dist = utils.partition_data(trainset, num_clients=NUM_CLIENTS, ratio=ratio, alpha=alpha, beta=beta)

#ids, dist = utils.partition_data_sharding(trainset, num_clients=NUM_CLIENTS)

entropies = [utils.compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]
print(entropies)
trainloaders = []

testloaders = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

for i in range(NUM_CLIENTS):
    trainloaders.append(DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(ids[i])))


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, entropy):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.entropy = 1 - entropy
        self.client_control = None # Scaffold only 

    def get_parameters(self, config):
        return utils.get_parameters(self.net)

    def fit(self, parameters, config):
        config = {**DEFAULT_CONFIG, **config, **{"entropy": self.entropy}}
        utils.set_parameters(self.net, parameters)
        metrics = fit_handler(algo_name=algo, cid=self.cid, net=self.net, trainloader=self.trainloader, config=config, client_control=self.client_control)
        
        if algo == "scaffold":
            self.client_control = metrics["client_control"]
            
        metrics = {k: v for k, v in metrics.items() if v is not None}
        return utils.get_parameters(self.net), len(self.trainloader.sampler), metrics

    def evaluate(self, parameters, config):
        config = {**DEFAULT_CONFIG, **config}
        utils.set_parameters(self.net, parameters)
        loss, accuracy, f1, recall, precision = utils.test(self.net, self.valloader)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision, "id": self.cid}


def client_fn(context: Context) -> FlowerClient:
    cid = int(context.node_config["partition-id"])
    net = MLP().to(DEVICE)  
    trainloader = trainloaders[int(cid)]  
    valloader = testloaders
    entropy = entropies[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader, entropy).to_client()


# Training
client_manager = ClientManager()

fl.simulation.start_simulation(
    client_fn = client_fn,
    num_clients = NUM_CLIENTS,
    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy = FedProx(
        num_rounds=NUM_ROUNDS,
        net=MLP(), 
        testloader=testloaders,
        num_clients=NUM_CLIENTS,
        current_parameters=current_parameters, 
        learning_rate = lr,
        decay_rate=1,
        fraction_fit=0.2,
        fraction_evaluate=0.02,
        proximal_mu = 1
       # all_classes = len(dist[0])
        ),
    client_manager=client_manager,
    client_resources = client_resources
)
