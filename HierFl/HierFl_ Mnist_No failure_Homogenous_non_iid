import flwr as fl
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import random
import numpy as np
import copy
from collections import OrderedDict
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import matplotlib.pyplot as plt
import csv
import time

########################################
# SEED
########################################
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
########################################
# Machine Learning Model (Net)
########################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Adjusted to match the output size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Output: (batch, 6, 12, 12)
        x = self.pool(torch.relu(self.conv2(x)))  # Output: (batch, 16, 4, 4)
        x = torch.flatten(x, 1)                  # Output: (batch, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

########################################
# EdgeDevice Class
########################################
class EdgeDevice:
    def __init__(self, device_id, trainloader, valloader):
        self.device_id = device_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = Net()  # Each device has its own local model

    def get_client(self):
        return FlowerClient(self.model, self.trainloader, self.valloader, self.device_id)

########################################
# EdgeServer Class
########################################
class EdgeServer:
    def __init__(self, server_id, devices: List[EdgeDevice]):
        self.server_id = server_id
        self.devices = devices
        self.model = Net()  # Each edge server has its own local model

    def aggregate(self):
        """Aggregate models from all connected devices."""
        total_samples = 0
        weighted_params = None

        for device in self.devices:
            client = device.get_client()
            parameters = client.get_parameters()
            num_samples = len(device.trainloader.dataset)

            if weighted_params is None:
                weighted_params = [num_samples * np.array(param) for param in parameters]
            else:
                for i, param in enumerate(parameters):
                    weighted_params[i] += num_samples * np.array(param)

            total_samples += num_samples

        # Average the parameters
        aggregated_params = [param / total_samples for param in weighted_params]
        return aggregated_params

########################################
# Parameter Utility Functions
########################################
def set_parameters(net: nn.Module, parameters: List[np.ndarray]):
    state_keys = list(net.state_dict().keys())
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in zip(state_keys, parameters)})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def test(net, testloader):
    net.eval()
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))
            outputs = net(inputs)
            total_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(testloader)
    return avg_loss, accuracy

########################################
# FlowerClient Class
########################################
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cid = cid

    def get_parameters(self):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
      self.set_parameters(parameters)
      self.model.train()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      criterion = nn.CrossEntropyLoss()

      # Default to 1 if 'k1' is not in config
      local_updates = config.get("k1", 1)
      for epoch in range(local_updates):  # Use k1 for local updates
          for inputs, labels in self.trainloader:
              inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))
              optimizer.zero_grad()
              outputs = self.model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

      return self.get_parameters(), len(self.trainloader.dataset), {}


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}

########################################
# Hierarchical Federated Learning with Flower
########################################
def HierFL(args, trainloaders, valloaders, testloader):
    edge_devices = [EdgeDevice(i, trainloaders[i], valloaders[i]) for i in range(args['NUM_DEVICES'])]
    num_edge_servers = args['NUM_EDGE_SERVERS']
    edge_servers = []
    devices_per_server = len(edge_devices) // num_edge_servers

    for i in range(num_edge_servers):
        start_idx = i * devices_per_server
        devices = edge_devices[start_idx:] if i == num_edge_servers - 1 else edge_devices[start_idx: start_idx + devices_per_server]
        edge_servers.append(EdgeServer(i, devices))

    global_model = Net()
    global_weights = get_parameters(global_model)

    metrics = {
        'rounds': [],
        'accuracy': [],
        'train_time': []
    }

    def client_fn(cid: str):
        device = edge_devices[int(cid)]
        return device.get_client()

    def evaluate_fn(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
        
    )

    def fit_config(server_round: int):
        return {"server_round": server_round, "k1": args['k1']}

    start_time = time.time()

    # Flower server configuration
    server_config = fl.server.ServerConfig(num_rounds=args['GLOBAL_ROUNDS'])

    # Start Flower simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args['NUM_DEVICES'],
        config=server_config,
        strategy=strategy,
    )

    end_time = time.time()
    total_time = end_time - start_time

    # Update metrics after simulation
    for round in range(1, args['GLOBAL_ROUNDS'] + 1):
        start_time = time.time()
        
        round_metrics = history.metrics_centralized['accuracy']
        accuracy = round_metrics[round - 1][1] if round - 1 < len(round_metrics) else None
        if accuracy is not None:
            metrics['rounds'].append(round)
            metrics['accuracy'].append(accuracy)
            metrics['train_time'].append(total_time / args['GLOBAL_ROUNDS'])
        end_time = time.time()
        round_time = end_time - start_time
        total_time += round_time
        metrics['train_time'].append(round_time)
        average_time = total_time / args['GLOBAL_ROUNDS']


    loss, accuracy = test(global_model, testloader)
    print(f"Final Model: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")
    
    with open('training_time_log.csv', 'a') as log_file:
        log_file.write(f'{round},{training_time}\n')
        
    save_metrics_to_csv(metrics, "mnist_results.csv")
    plot_metrics(metrics)
    
#Edge Aggregation every k2 rounds**

if round_number % args["k2"] == 0:
            print(f"🔹 Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
                aggregated_params = edge_server.aggregate()
                set_parameters(edge_server.model, aggregated_params)

# Global Aggregation every (k1 × k2) rounds**
if round_number % (args["k1"] * args["k2"]) == 0:
            print(f"🌍 Aggregating at GLOBAL SERVER (every {args['k1'] * args['k2']} rounds)")
            global_weights = get_parameters(global_model)
            set_parameters(global_model, global_weights)

print("✅ Federated Training Complete!")
########################################
# Utility Functions
########################################
def save_metrics_to_csv(metrics, filename="mnist_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Round', 'Accuracy', 'Training Time'])
        for i in range(len(metrics['rounds'])):
            writer.writerow([metrics['rounds'][i], metrics['accuracy'][i], metrics['train_time'][i]])
    print(f"Metrics saved to {filename}")

def plot_metrics(metrics):
    plt.figure()
    plt.plot(metrics['rounds'], metrics['accuracy'], marker='o', label="Accuracy")
    plt.xlabel("Global Rounds")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Global Rounds")
    plt.grid()
    plt.legend()
    plt.savefig("accuracy_vs_rounds.png")
    plt.show()

def main():
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    def load_datasets(num_clients: int, batch_size: int = 32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3 channels
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(SEED))
        split = []
        num_classes = 10
        samples_per_client = len(train_dataset) // num_clients

        # Extract labels from dataset
        labels = np.array(train_dataset.targets)

        for i in range(num_clients):
            class_indices = np.where((labels == (i % num_classes)) | (labels == ((i+1) % num_classes)))[0]
            selected_indices = np.random.choice(class_indices, samples_per_client, replace=False)
            split.append(torch.tensor(selected_indices))

        trainloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED)) for s in split]
        valloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=False) for s in split]
        testloader = DataLoader(test_dataset, batch_size=batch_size)
        return trainloaders, valloaders, testloader

    args = {
        'NUM_DEVICES': 20,
        'NUM_EDGE_SERVERS': 5,
        'GLOBAL_ROUNDS': 50,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'CLIENT_FRACTION': 0.1,
        'EVALUATE_FRACTION': 0.1,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()

