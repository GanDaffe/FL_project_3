import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_METRICS = {
    "loss": None,
    "accuracy": None,
    "f1": None,
    "recall": None,
    "precision": None,
    "id": None, 
    "num_classes": None, 
    "client_control": None
}


DEFAULT_CONFIG = {
    "learning_rate": 0.01,
    "proximal_mu": 0.1,
    "epochs": 1,
    "num_classes": 10,  
    "device": DEVICE, 
    "tau": 0.0, 
    "beta": 0.0,
    "alpha": 0.0, 
    "entropy": 0.0,
}

