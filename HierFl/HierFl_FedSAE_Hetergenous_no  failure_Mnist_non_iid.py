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
import os
from google.colab import files
import os
from google.colab import drive
import pandas as pd
from math import ceil

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
#history of clients
########################################
client_history = {}

########################################
# CSV Logging - Ensure File is Created at Start
########################################
log_file_path = "client_task_log.csv"
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as log_file:
        log_file.write("Round,Client,lowerBound,UpperBound,ClientAffordableWorkload\n")



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
      return FlowerClient(self.model, self.trainloader, self.valloader, self.device_id, self.device_id) 


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
# Adjust L and H After Each Round
########################################
def adjust_task_assignment(clients, alpha, r1, r2):
    """
    Adjusts the task assignment based on each client's affordable workload and dynamic threshold.
    :param clients: List of client objects containing workload information.
    :param alpha: Smoothing factor for threshold updates.
    :param r1, r2: Adjustment factors for workload updates.
    """
    with open(log_file_path, "a") as log_file:
      for client in clients:
          # Initialize affordable workload ~Etk
          mu_k = np.random.uniform(50, 60)
          sigma_k = np.random.uniform(mu_k / 4, mu_k / 2)
          client.affordable_workload = np.random.normal(mu_k, sigma_k)
          
          # Initialize workload range [10, 20)
          if not hasattr(client, 'lower_bound'):
              client.lower_bound = 10
          if not hasattr(client, 'upper_bound'):
              client.upper_bound = 20
          
          # Initialize threshold θtk if not already initialized
          if not hasattr(client, 'threshold'):
              client.threshold = 0
          
          # Update threshold θtk
          client.threshold = alpha * client.threshold + (1 - alpha) * client.affordable_workload
          
          # Extract workload range
          L_tk, H_tk = client.lower_bound, client.upper_bound
          
          # Update workload based on the algorithm steps
          if client.affordable_workload > H_tk:
              if client.threshold <= L_tk:
                  client.lower_bound += r2
                  client.upper_bound += r2
              elif L_tk < client.threshold <= H_tk:
                  client.lower_bound += r1
                  client.upper_bound += r2
              else:
                  client.lower_bound += r1
                  client.upper_bound += r1
                  client.affordable_workload = H_tk
          elif L_tk < client.affordable_workload <= H_tk:
              if client.threshold >= L_tk:
                  client.lower_bound = min(client.lower_bound + r2, 0.5 * H_tk)
                  client.upper_bound = max(client.lower_bound + r2, 0.5 * H_tk)
              elif L_tk < client.threshold <= H_tk:
                  client.lower_bound = min(client.lower_bound + r1, 0.5 * H_tk)
                  client.upper_bound = max(client.lower_bound + r1, 0.5 * H_tk)
                  client.affordable_workload = L_tk
              else:
                  client.lower_bound = 0.5 * L_tk
                  client.upper_bound = 0.5 * H_tk
                  client.affordable_workload = 0
                  
          # Log client information
          log_file.write(f"{round},{client.cid},{client.lower_bound},{client.upper_bound},{client.affordable_workload}\n")
    
    return clients
########################################
# Trainig round adjustment
########################################

def compute_training_rounds(client_id, clients, base_k1):
    """
    Compute training rounds dynamically based on the client's affordable workload.
    :param client_id: The ID of the client.
    :param clients: List of client objects.
    :param base_k1: Base number of local training rounds.
    """
    client = next(c for c in clients if c.cid == client_id)
    return max(5, int(round(base_k1 * (client.affordable_workload / 60), 2)))


########################################
# FlowerClient Class
########################################
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, model, trainloader, testloader, valloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.cid = cid


    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_keys = list(self.model.state_dict().keys())
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in zip(state_keys, parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
      global client_history

      client_id = self.cid

      # ✅ Ensure client_id exists in client_history before using it
      if client_id not in client_history:
          print(f"⚠️ Warning: Client {client_id} not found in client_history. Initializing with default values.")
          client_history[client_id] = {
              "L": 10, "H": 20, "epochs": 10, "task": "Easy",
              "training_time": [], "accuracy": []
          }

      num_epochs = config.get("num_epochs", 1)  # Default to 1 if missing

      # ✅ Ensure num_epochs is an integer
      if num_epochs is None:
          print(f"⚠️ Warning: 'num_epochs' is None for client {client_id}. Defaulting to 10.")
          num_epochs = 10
      num_epochs = int(num_epochs)  # Force integer conversion

      self.set_parameters(parameters)
      self.model.train()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      criterion = nn.CrossEntropyLoss()

      print(f"Client {client_id}: Training for {num_epochs} epochs.")

      start_time = time.time()
      for epoch in range(num_epochs):
          for inputs, labels in self.trainloader:
              optimizer.zero_grad()
              outputs = self.model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

      end_time = time.time()
      training_time = end_time - start_time

      client_history[client_id]["training_time"].append(training_time)
      print(f"✅ Client {client_id}: Training completed in {training_time:.2f} sec.")

      return self.get_parameters(), len(self.trainloader.dataset), {"training_time": training_time}


########################################
# Hierarchical Federated Learning with Flower
#######################################
def HierFL(args, trainloaders, valloaders, testloader):
    global client_history
    

    # Initialize Clients with Workload Parameters
    clients = [
    FlowerClient(
        model=Net(),  # Assuming you are using Net() for the model
        trainloader=trainloaders[i], 
        testloader=testloader, 
        valloader=valloaders[i], 
        cid=i
    ) 
    for i in range(args['NUM_DEVICES'])
]

    clients = adjust_task_assignment(clients, alpha=args['alpha'], r1=args['r1'], r2=args['r2'])

    # Initialize Edge Devices
    edge_devices = [EdgeDevice(i, trainloaders[i], valloaders[i]) for i in range(args['NUM_DEVICES'])]
    num_edge_servers = args['NUM_EDGE_SERVERS']
    edge_servers = []
    devices_per_server = len(edge_devices) // num_edge_servers

    for i in range(num_edge_servers):
        start_idx = i * devices_per_server
        devices = edge_devices[start_idx:] if i == num_edge_servers - 1 else edge_devices[start_idx: start_idx + devices_per_server]
        edge_servers.append(EdgeServer(i, devices))

    # Global Model
    global_model = Net()
    global_weights = get_parameters(global_model)

    # Define Evaluation Function
    def evaluate_fn(server_round, parameters, config):
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    # Define Federated Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
    )

    # **Start Federated Learning Simulation**
    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(
            model=Net(),
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            testloader=testloader,
            cid=int(cid)
        ),
        num_clients=len(trainloaders),
        config=fl.server.ServerConfig(num_rounds=args['GLOBAL_ROUNDS']),
        strategy=strategy
    )

    # **Federated Learning Rounds**
    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):

        available_clients = list(range(args["NUM_DEVICES"]))  # All clients are available

        if len(available_clients) < 2:
            print(f"⚠️ Very few available clients, reducing selected clients.")
            selected_clients = available_clients  # Use whoever is available
        else:
            selected_clients = random.sample(available_clients, min(2, len(available_clients)))

        # **Update workloads before assigning tasks**
        clients = adjust_task_assignment(clients, alpha=args['alpha'], r1=args['r1'], r2=args['r2'])

        for client_id in selected_clients:
            # Compute Training Rounds Dynamically
            num_epochs = compute_training_rounds(client_id, clients, args['base_k1'])

            print(f"Client {client_id}: Assigned {num_epochs} epochs (Affordable Workload: {clients[client_id].affordable_workload:.2f})")

            client = FlowerClient(
                model=Net(),
                trainloader=trainloaders[client_id],
                testloader=testloader,
                valloader=valloaders[client_id],
                cid=client_id
            )

            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})

        # **Edge Aggregation every k2 rounds**
        if round_number % args["k2"] == 0:
            print(f" Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
                aggregated_params = edge_server.aggregate()
                set_parameters(edge_server.model, aggregated_params)

        # **Global Aggregation every (k1 × k2) rounds**
        if round_number % (args["k1"] * args["k2"]) == 0:
            print(f" Aggregating at GLOBAL SERVER (every {args['k1'] * args['k2']} rounds)")
            global_weights = get_parameters(global_model)
            set_parameters(global_model, global_weights)



########################################
# Main
#######################################

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
        'NUM_DEVICES': 10,
        'NUM_EDGE_SERVERS': 5,
        'GLOBAL_ROUNDS':2,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'CLIENT_FRACTION': 0.2,
        'EVALUATE_FRACTION': 0.2,
        'FAILURE_DURATION': 50,
        'alpha': 0.95,
        'r1': 3,
        'r2': 1,
        'base_k1': 60,
        'num_epochs': 10,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()



    
    
    
