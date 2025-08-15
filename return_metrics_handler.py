from mutual_handler import DEFAULT_METRICS
import os
import copy
import torch

from utils import train
from support_function_and_class_for_FL import train_moon

MOON_SAVE_DIR = '/moon_save_point/'

def fit_handler(algo_name, cid, config, net, trainloader):
    """
    Handler function to return the metrics based on the algorithm name.
    
    Args:
        algo_name (str): Name of the algorithm.
        config (dict): Configuration parameters for the algorithm.
        metrics (dict): Metrics to be returned.
    
    Returns:
        dict: A dictionary containing the algorithm name, configuration, and metrics.
    """
    if algo_name == "fedprox":
        res_metrics = train(net, trainloader, learning_rate=config["learning_rate"], epochs=config["epochs"], proximal_mu=config['proximal_mu'] * config["entropy"])
    elif algo_name == "fedavg":
        res_metrics = train(net, trainloader, learning_rate=config["learning_rate"], epochs=config["epochs"])
    elif algo_name == "fedntd":
        res_metrics = train(net, trainloader, learning_rate=config["learning_rate"], epochs=config["epochs"], use_ntd_loss=True, tau=config["tau"], beta=config["beta"])
    elif algo_name == "fedcls": 
        last_layer = list(net.modules())[-1]
        num_classes = last_layer.out_features
        num_classes_metrics = {'num_classes': num_classes} 
        res_metrics = train(net, trainloader, learning_rate=config["learning_rate"], epochs=config["epochs"])
        res_metrics.update(num_classes_metrics)
    elif algo_name == "moon": 
        save_dir = os.path.join(MOON_SAVE_DIR, f"client_{cid}")
        pre_round_net = copy.deepcopy(net)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else: 
            prev_net_path = os.path.join(save_dir, "prev_net.pt")
            if os.path.exists(prev_net_path):
                pre_round_net.load_state_dict(torch.load(prev_net_path))
        
        global_net = copy.deepcopy(net)
        
        _, loss, acc = train_moon(
            net,
            global_net,
            pre_round_net,
            trainloader,
            lr=config["learning_rate"],
            temperature=config["temperature"],
            device=config["device"],
            epochs=config["epochs"],
        )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            torch.save(net.state_dict(), os.path.join(save_dir, "prev_net.pt"))
            
        res_metrics = {
            "loss": loss,
            "accuracy": acc
        }
    return {**DEFAULT_METRICS, **res_metrics, **{"id": cid}}

