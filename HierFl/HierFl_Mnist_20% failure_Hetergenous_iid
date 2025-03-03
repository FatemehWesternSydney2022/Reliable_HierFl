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
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
# MODEL DEFINITION
########################################
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_client(self):
        """Creates a FlowerClient when needed"""
        return FlowerClient(Net(), self.trainloader, self.valloader, self.device_id, self.device)

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

      # ✅ Ensure aggregation only happens if there are valid samples
      if total_samples == 0:
          print("⚠️ Warning: No valid clients provided parameters. Skipping aggregation.")
          return None  

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
# FLOWER CLIENT
########################################
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader, cid, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.cid = cid
        self.device = device

    def get_parameters(self):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters) 

    def fit(self, parameters, config):
          self.set_parameters(parameters)
          self.model.train()
          optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
          criterion = nn.CrossEntropyLoss()

          # Use the dynamically assigned `k1`
          local_updates = config.get("k1", 60)  # Default to 60 if missing
          print(f"Client {self.cid} training for {local_updates} rounds")

          start_time = time.time()  # 🔹 Start timer
          for epoch in range(local_updates):  # Use dynamic `k1`
              for inputs, labels in self.trainloader:
                  inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))
                  optimizer.zero_grad()
                  outputs = self.model(inputs)
                  loss = criterion(outputs, labels)
                  loss.backward()
                  optimizer.step()

          end_time = time.time()  # 🔹 End timer
          training_time = end_time - start_time  # 🔹 Calculate training time

          return self.get_parameters(), len(self.trainloader), {"training_time": training_time}


    def evaluate(self, parameters, config):
          self.set_parameters(parameters)
          loss, accuracy = test(self.model, self.testloader)
          return float(loss), len(self.testloader.dataset), {"accuracy": accuracy}


########################################
# Random Failure Simulation
########################################
def simulate_failures(args, unavailability_tracker, failure_log, round_number, training_time, selected_clients, available_clients):
    """Simulates client failures and updates the failure log dynamically per round."""
     #Identify clients who are available (not currently failing)
    available_clients = [
        cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0 and cid in selected_clients
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
        failure_log.remove(client)  # Remove from failure log

    return failure_log


# ✅ Open file to store logs
log_file_path = "detailed_failure_log.csv"
with open(log_file_path, "w") as log_file:
    log_file.write("Round,ClientID,FailureDuration,RecoveryTimeRemaining,TrainingTime,AvailableClientsAfterFailure,CPU,RAM\n")

########################################
# HIERARCHICAL FEDERATED LEARNING (HierFL)
########################################
def HierFL(args, trainloaders, valloaders, testloader):
    # Define Edge Devices
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

    # Define Unavailability Tracker for Failures
    unavailability_tracker = {cid: 0 for cid in range(args["NUM_DEVICES"])}
    failure_log = []

    def evaluate_fn(server_round, parameters, config):
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    # Define Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
    )

    # Define Heterogeneous Clients
    client_profiles = {
        cid: {
            "CPU": random.uniform(1.5, 3.5),
            "RAM": random.randint(4, 16),
            
        }
        for cid in range(args["NUM_DEVICES"])
    }

    def compute_training_rounds(client_id, base_k1):
        """Compute training rounds dynamically based on client capabilities."""
        cpu_factor = client_profiles[client_id]["CPU"] / 3.5
        memory_factor = client_profiles[client_id]["RAM"] / 16
        performance_factor = (0.7 * cpu_factor) + (0.3 * memory_factor)
        return max(5, int(base_k1 * performance_factor))

    # Initialize Logging Files
    client_k1_log_path = "client_k1_log.csv"
    failure_log_path = "detailed_failure_log.csv"

    with open(client_k1_log_path, "w") as log_file:
        log_file.write("Round,Client ID,CPU (GHz),RAM (GB),Adjusted k1,Training Time\n")

    with open(failure_log_path, "w") as log_file:
        log_file.write("Round,ClientID,FailureDuration,RecoveryTimeRemaining,TrainingTime,AvailableClientsAfterFailure,CPU,RAM\n")

    # Training Rounds
    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):
        print(f"\n🔵 Round {round_number}")

         # 🔹 **Decrease failure time and recover clients**
        for client_id in list(unavailability_tracker.keys()):
            if unavailability_tracker[client_id] > 0:
                unavailability_tracker[client_id] -= 1  # Reduce remaining failure time
                if unavailability_tracker[client_id] == 0:
                    print(f"✅ Client {client_id} has recovered and is available again!")

        # Select available clients (excluding failed ones)
        available_clients = [cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0]
        if len(available_clients) < 2:
            print(f"⚠️ Round {round_number}: Not enough clients available. Skipping round.")
            continue

        selected_clients = random.sample(available_clients, min(2, len(available_clients)))
        print(f"🔹 Selected Clients: {selected_clients}")

        round_training_time = 0  # Track cumulative training time per round

        for client_id in selected_clients:
            client = FlowerClient(
                model=Net(),
                trainloader=trainloaders[client_id],
                testloader=testloader,
                cid=client_id,
                device=args['DEVICE']
            )

            # Compute Adjusted k1 Based on Hardware
            base_k1 = args["k1"]
            adjusted_k1 = compute_training_rounds(client_id, base_k1)
            print(f"Client {client_id}: Training for {adjusted_k1} local rounds.")

            _, _, train_metrics = client.fit(get_parameters(client.model), {"k1": adjusted_k1})
            client_training_time = train_metrics.get("training_time", 0)
            round_training_time += round(client_training_time,2)  # Accumulate training time

            # Log Adjusted k1 Per Client
            with open(client_k1_log_path, "a") as log_file:
                log_file.write(f"{round_number},{client_id},{client_profiles[client_id]['CPU']:.2f},"
                               f"{client_profiles[client_id]['RAM']},{adjusted_k1},{client_training_time:.2f}\n")

        

        #Simulate Failures After Training
        simulate_failures(args, unavailability_tracker, failure_log, round_number, client_training_time, available_clients, selected_clients)

        # Count available clients AFTER failures
        available_clients_after_failures = [cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0]
        print(f"✅ Round {round_number}: Available Clients AFTER failure = {len(available_clients_after_failures)}")

        # Log Failures in `detailed_failure_log.csv`
        if failure_log:
          with open(failure_log_path, "a") as log_file:
            recovered_clients = []
            for client in failure_log:
                if client[2] == 0:
                    log_file.write(f"{round_number},{client[0]},{client[1]},{client[2]:.2f},{client_training_time},"
                                   f"{len(available_clients_after_failures)},{client_profiles[client[0]]['CPU']},"
                                   f"{client_profiles[client[0]]['RAM']},{client_profiles[client[0]]}\n")
                    recovered_clients.append(client)  # Mark for removal after logging
                else:
                    log_file.write(f"{round_number},{client[0]},{client[1]},{client[2]:.2f},{client_training_time},"
                                   f"{len(available_clients_after_failures)},{client_profiles[client[0]]['CPU']},"
                                   f"{client_profiles[client[0]]['RAM']},{client_profiles[client[0]]}\n")

            # Remove fully recovered clients AFTER logging
            for client in recovered_clients:
                failure_log.remove(client)

    print(f"✅ Failure log saved as: {failure_log_path}")
    print(f"✅ Training rounds log saved as: {client_k1_log_path}")
    print(f"Total Training Time: {client_training_time} seconds")
    
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
# MAIN FUNCTION
########################################
def main():
    def load_datasets(num_clients: int, batch_size: int = 32):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(".", train=False, download=True, transform=transform)

        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(SEED))
        split = torch.split(indices, len(train_dataset) // num_clients)

        trainloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=True) for s in split]
        valloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=False) for s in split]
        testloader = DataLoader(test_dataset, batch_size=batch_size)
        return trainloaders, valloaders, testloader

    args = {
        'NUM_DEVICES': 20,
        'NUM_EDGE_SERVERS': 5,
        'GLOBAL_ROUNDS': 50,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'CLIENT_FRACTION': 0.2,
        'EVALUATE_FRACTION': 0.2,
        'FAILURE_RATE': 0.2,
        'FAILURE_DURATION': 50,
        'k1': 60,
        'k2': 1
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()


