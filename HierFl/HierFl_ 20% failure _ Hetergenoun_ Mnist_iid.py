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
import torch.nn.functional as F
import os
import math


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
# CSV file
########################################
# ✅ Ensure CSV file exists
log_file_path = "training_log.csv"
if not os.path.exists(log_file_path):
        with open(log_file_path, "w") as log_file:
            log_file.write("Round,Client ID,NumEpoch,AffordableWorkload,TrainingTime\n")

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

          if parameters is None or len(parameters) == 0:
              print(f"⚠️ Warning: No parameters received from Client {client.cid}. Skipping.")
              continue  # Skip clients with no parameters

          if weighted_params is None:
              weighted_params = [num_samples * np.array(param) for param in parameters]
          else:
              for i, param in enumerate(parameters):
                  weighted_params[i] += num_samples * np.array(param)

          total_samples += num_samples

      # ✅ Fix: Skip aggregation instead of raising an error
      if total_samples == 0 or weighted_params is None:
          print("⚠️ No valid parameters received! Skipping aggregation for this round.")
          return None  # Skip this round

      # Average the parameters
      aggregated_params = [param / total_samples for param in weighted_params]
      return aggregated_params



########################################
# Parameter Utility Functions
########################################
model = Net()

def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model: nn.Module, parameters: Optional[List[np.ndarray]]):
    if parameters is None:
        print("⚠ Warning: Received NoneType parameters. Skipping model update.")
        return  

    print(f"✅ Updating model with {len(parameters)} parameter tensors.")  # Debugging print
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
 
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
# Generate affordable workload
########################################
def client_affordable_workload():
    """
    Generates a client's affordable workload based on a normal distribution.
    Returns:
        float: The sampled affordable workload.
    """
    mu_k = np.random.uniform(50, 60)  # Mean workload
    sigma_k = np.random.uniform(mu_k / 4, mu_k / 2)  # Std deviation
    return max(0, np.random.normal(mu_k, sigma_k))  # Ensure non-negative workload


########################################
# FlowerClient Class
########################################
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, cid):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cid = cid

        # ✅ Use function to initialize workload
        self.affordable_workload = client_affordable_workload()

    def get_parameters(self):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
      self.set_parameters(parameters)
      self.model.train()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      criterion = nn.CrossEntropyLoss()

      num_epochs = config.get("k1", 1)
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

      return self.get_parameters(), len(self.trainloader.dataset), {"training_time": training_time}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}


########################################
# Compute training rounds dynamically
########################################
def compute_training_rounds(client):
    """
    Compute k1 directly from the client's affordable workload.
    No normalization or scaling is applied.
    
    :param client: The specific client for which k1 is being calculated.
    :return: k1 value for the specific client.
    """
    return math.floor(client.affordable_workload)  # ✅ Always round down

########################################
# Random Failure Simulation
########################################
def simulate_failures(args, unavailability_tracker, failure_log, round_number, training_time):
    """Simulates client failures and updates the failure log dynamically per round."""
     #Identify clients who are available (not currently failing)
    available_clients = [
        cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0
    ]

    num_failures = int(len(available_clients) * args['FAILURE_RATE'])
    num_failures = max(1, num_failures)  # Ensure at least one client fails

   

    # Ensure failure is only assigned to NEW available clients, not already failing ones
    failing_clients = [
        cid for cid in available_clients if cid not in [c[0] for c in failure_log]
    ]
    
    failing_clients = random.sample(failing_clients, min(num_failures, len(failing_clients)))

    # Assign failure durations and update log
    new_failures = []
    for client_id in failing_clients:
        failure_duration = random.randint(1, args['FAILURE_DURATION'])  # Assign failure duration
        recovery_time_remaining = failure_duration  # Initially set full failure time

        unavailability_tracker[client_id] = 1  # Mark as unavailable
        new_failures.append([client_id, failure_duration, recovery_time_remaining])

    # Append new failures to log
    failure_log.extend(new_failures)

    # ✅ **Update recovery time for all currently failing clients**
    for client in failure_log:
        client[2] -= training_time  # Reduce recovery time by training duration
        client[2] = max(0, client[2])  # Ensure it doesn’t go negative

    # ✅ **Recover clients whose recovery time reaches 0**
    recovered_clients = [c for c in failure_log if c[2] == 0]
    for client in recovered_clients:
        client_id = client[0]
        unavailability_tracker[client_id] = 0  # Mark client as available
        #failure_log.remove(client)  # Remove from failure log

    return failure_log

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

    # ✅ Track client availability (0 = available, >0 = failure duration remaining)
    unavailability_tracker = {cid: 0 for cid in range(args['NUM_DEVICES'])}
    failure_log = []  # Track failed clients and recovery countdowns

    # ✅ Compute k1 values ONCE before training starts
    k1_values = {
        cid: compute_training_rounds(edge_devices[cid].get_client())  
        for cid in range(args['NUM_DEVICES'])
    }

    print(f"✅ k1 values initialized: {k1_values}")

    # ✅ Initialize Clients Once
    def client_fn(cid: str):
        device = edge_devices[int(cid)]
        return device.get_client()

    # ✅ Define Evaluation Function
    def evaluate_fn(server_round, parameters, config):
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    # ✅ Define Federated Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=max(1, int(args['NUM_DEVICES'] * args['CLIENT_FRACTION'])),
        min_evaluate_clients=max(1, int(args['NUM_DEVICES'] * args['EVALUATE_FRACTION'])),
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
    )

    # ✅ Start Federated Learning Simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(
            model=Net(),
            trainloader=trainloaders[int(cid)],
            testloader=testloader,
            cid=int(cid)
        ),
        num_clients=len(trainloaders),
        config=fl.server.ServerConfig(num_rounds=args['GLOBAL_ROUNDS']),
        strategy=strategy
    )

    # ✅ Federated Learning Rounds
    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):

        start_time = time.time()
        training_time = 0.0

        # ✅ Simulate failures before selecting clients
        failure_log = simulate_failures(args, unavailability_tracker, failure_log, round_number, training_time)

        # ✅ Select only available clients for training
        available_clients = [cid for cid in range(args["NUM_DEVICES"]) if unavailability_tracker[cid] == 0]

        if not available_clients:
            print(f"⚠️ No available clients in round {round_number}. Skipping.")
            continue  

        selected_clients = random.sample(available_clients, max(1, int(len(available_clients) * args['CLIENT_FRACTION'])))

        for client_id in selected_clients:
            num_epochs = k1_values[client_id]  # ✅ k1 remains fixed per client
            client = edge_devices[client_id].get_client()

            print(f"Client {client_id}: Fixed {num_epochs} epochs (Affordable Workload: {client.affordable_workload:.2f})")

            # ✅ Pass fixed k1 to client training
            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})
            training_time = train_metrics["training_time"]

            # ✅ Log training details
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{round_number},{client_id},{num_epochs},{client.affordable_workload:.2f},{training_time:.2f}\n")
                log_file.flush()

        # ✅ Edge Aggregation
        if round_number % args["k2"] == 0:
            print(f"🔹 Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
                aggregated_params = edge_server.aggregate()
                set_parameters(edge_server.model, aggregated_params)

        # ✅ Global Aggregation
        if round_number % (args["k1"] * args["k2"]) == 0:
            print(f"🌍 Aggregating at GLOBAL SERVER (every {args['k1'] * args['k2']} rounds)")
            global_weights = get_parameters(global_model)
            set_parameters(global_model, global_weights)

    print("✅ Federated Training Complete!")

    loss, accuracy = test(global_model, testloader)
    print(f"Final Model: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")




########################################
# Main
########################################
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
        split = torch.split(indices, len(train_dataset) // num_clients)

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
        'CLIENT_FRACTION': 0.5,
        'EVALUATE_FRACTION': 0.5,
        'FAILURE_RATE': 0.2,
        'FAILURE_DURATION': 50,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()


