from algo import *

class FedProx(FedAvg):
    def __init__(
        self,
        *args,
        proximal_mu: float = 0.1,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def __repr__(self) -> str:
        return "FedProx"


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate, "proximal_mu": self.proximal_mu}
        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs
