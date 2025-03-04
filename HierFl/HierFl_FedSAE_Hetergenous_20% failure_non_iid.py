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
# Failure Tracking
########################################
failure_log = []
unavailability_tracker = {}

########################################
#history of clients
########################################
client_history = {}

########################################
# CSV Logging - Ensure File is Created at Start
########################################
log_file_path = "client_failure_task_log.csv"
with open(log_file_path, "w") as log_file:
    log_file.write("Round,Client ID,Before L,Before H,Before Task,Before Epochs,After L,After H,"
                   "After Task,After Epochs,Status,Failure Status,Failure Duration,Recovery Time Remaining\n")



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
        return FlowerClient(self.model, self.trainloader, self.valloader, self.device_id, cid)

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
# Random Failure Simulation
########################################


def simulate_failures(args, unavailability_tracker, failure_log, round_number, training_time, selected_clients, available_clients):
    """Ensures exactly `FAILURE_RATE` proportion of clients fail each round."""

    # ✅ Identify available clients who are NOT currently failing
    available_clients = [cid for cid in selected_clients if unavailability_tracker.get(cid, 0) == 0]

    if len(available_clients) == 0:
        print(f"⚠️ Round {round_number}: No clients available for failure.")
        return failure_log, []

    # ✅ Determine number of failures (use `ceil()` to ensure at least 20% fail)
    num_failures = min(len(available_clients), ceil(len(available_clients) * args['FAILURE_RATE']))

    # ✅ Select clients to fail (only from available, non-failing clients)
    failing_clients = random.sample(available_clients, num_failures)

    print(f"🔥 Round {round_number}: Failing {len(failing_clients)} clients: {failing_clients}")

    # ✅ Assign failure durations and mark clients as unavailable
    for client_id in failing_clients:
        failure_duration = random.uniform(1, args['FAILURE_DURATION'])  # Assign failure duration
        unavailability_tracker[client_id] = failure_duration  # Set client as unavailable
        failure_log.append([client_id, failure_duration, failure_duration])  # Log failure

        # ✅ Log failed clients in main log file
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{round_number},{client_id},-,-,-,-,-,-,-,-,"
                f"FAILED,{failure_duration:.2f},{failure_duration:.2f}\n"
            )

    # ✅ Update recovery times for all failing clients
    for client in failure_log:
        client[2] -= training_time  # Reduce recovery time
        client[2] = max(0, client[2])  # Ensure it doesn’t go negative

    # ✅ Recover clients whose recovery time reaches 0
    recovered_clients = [c for c in failure_log if c[2] == 0]
    for client in recovered_clients:
        client_id = client[0]
        unavailability_tracker[client_id] = 0  # ✅ Mark client as AVAILABLE
        failure_log.remove(client)  # ✅ Remove from failure log

        # ✅ Log recovered clients
        with open(log_file_path, "a") as log_file:
            log_file.write(
                f"{round_number},{client_id},-,-,-,-,-,-,-,-,"
                f"RECOVERED,{client[1]:.2f},0\n"
            )

    print(f"🔄 Round {round_number}: {len(recovered_clients)} clients recovered: {recovered_clients}")

    return failure_log, recovered_clients



########################################
# Adjust L and H After Each Round
########################################
def adjust_task_assignment(client_id, success):
    """Adjust L, H, and task difficulty based on client execution ability."""
    global client_history

    if client_id not in client_history:
        print(f"⚠️ Warning: Client {client_id} missing from history. Initializing now.")
        client_history[client_id] = {"L": 50, "H": 100, "task": "Easy", "epochs": 50}

    history = client_history[client_id]

    # ✅ If the client just recovered from failure, halve workload
    recovered_clients = [c[0] for c in failure_log if c[2] == 0]
    if client_id in recovered_clients:
        history["L"] = max(10, history["L"] // 2)  # Ensure L does not drop below 10
        history["H"] = max(20, history["H"] // 2)  # Ensure H does not drop below 20
        history["task"] = "Easy"
        history["epochs"] = history["L"]
        print(f"🛠 Client {client_id} recovered → workload halved: L={history['L']}, H={history['H']}")
        return history["task"], history["epochs"], history["L"], history["H"]

    # ✅ Extract values before modification
    before_L = history["L"]
    before_H = history["H"]
    before_task = history["task"]
    before_epochs = history["epochs"]

    if success:
        # ✅ If the client successfully completed H tasks, increase L and H
        history["L"] = before_L + 10
        history["H"] = before_H + 10
        history["task"] = "Difficult"
        history["epochs"] = history["L"]
    else:
        # ❌ If the client fails difficult tasks, halve both L and H
        history["L"] = max(10, before_L // 2)
        history["H"] = max(20, before_H // 2)
        history["task"] = "Easy"
        history["epochs"] = history["L"]

    return history["task"], history["epochs"], history["L"], history["H"]


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
        global client_history, unavailability_tracker

        client_id = self.cid
        num_epochs = config.get("num_epochs", 1)

        # ✅ Ensure client_id exists in unavailability_tracker before using it
        if client_id not in unavailability_tracker:
            unavailability_tracker[client_id] = 0  # Initialize if missing
            client_history[client_id] = {"L": 50, "H": 70, "epochs": 50, "task": "Easy",
                                      "training_time": [], "accuracy": []}

        if unavailability_tracker[client_id] == 1:
            print(f"❌ Client {client_id} is unavailable due to failure. Skipping training.")
            return self.get_parameters(), len(self.trainloader.dataset), {"training_time": 0}

        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        num_epochs = config["num_epochs"]
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
    global unavailability_tracker

    # Initialize Edge Devices
    edge_devices = [EdgeDevice(i, trainloaders[i], valloaders[i]) for i in range(args['NUM_DEVICES'])]
    num_edge_servers = args['NUM_EDGE_SERVERS']
    edge_servers = []
    devices_per_server = len(edge_devices) // num_edge_servers

    for i in range(num_edge_servers):
        start_idx = i * devices_per_server
        devices = edge_devices[start_idx:] if i == num_edge_servers - 1 else edge_devices[start_idx: start_idx + devices_per_server]
        edge_servers.append(EdgeServer(i, devices))

    
    
    # Initialize unavailability and client history
    unavailability_tracker = {cid: 0 for cid in range(args["NUM_DEVICES"])}
    client_history = {
        cid: {"L": 50, "H": 70, "epochs": 50, "task": "Easy", "training_time": [], "accuracy": []}
        for cid in range(args['NUM_DEVICES'])
    }

    # Global Model
    global_model = Net()
    global_weights = get_parameters(global_model)



    # Define Evaluation Function
    def evaluate_fn(server_round, parameters, config):
        set_parameters(global_model, parameters)
        loss, accuracy = test(global_model, testloader)
        return loss, {"accuracy": accuracy}

    # Define Heterogeneous Clients
    client_profiles = {
        cid: {
            "CPU": random.uniform(1.5, 3.5),  # CPU speed in GHz
            "RAM": random.randint(4, 16),  # RAM in GB
        }
        for cid in range(args["NUM_DEVICES"])
    }

    def compute_training_rounds(client_id, base_k1):
        """Compute training rounds dynamically based on client capabilities."""
        cpu_factor = client_profiles[client_id]["CPU"] / 3.5  # Normalize to max 3.5GHz
        memory_factor = client_profiles[client_id]["RAM"] / 16  # Normalize to max 16GB RAM

        # Weighted contribution: 70% CPU, 30% RAM
        performance_factor = (0.7 * cpu_factor) + (0.3 * memory_factor)

        adjusted_k1 = int(base_k1 * performance_factor)  # Scale training rounds
        return max(5, adjusted_k1)  # Ensure a minimum of 5 rounds


    # Define Federated Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args['CLIENT_FRACTION'],
        fraction_evaluate=args['EVALUATE_FRACTION'],
        min_fit_clients=2,
        min_evaluate_clients=2,
        initial_parameters=ndarrays_to_parameters(global_weights),
        evaluate_fn=evaluate_fn,
    )

    # **Start Federated Learning Simulation (Move Outside the Loop)**
    fl.simulation.start_simulation(
        client_fn=lambda cid: FlowerClient(
            model=Net(),
            trainloader=trainloaders[int(cid)],  
            valloader=valloaders[int(cid)],      
            testloader=testloader,
            cid = int(cid)
        ),
        num_clients=len(trainloaders),
        config=fl.server.ServerConfig(num_rounds=args['GLOBAL_ROUNDS']),
        strategy=strategy
    )

    # **Federated Learning Rounds**
    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):

        available_clients = [cid for cid, unavailable in unavailability_tracker.items() if unavailable == 0]

        if len(available_clients) < 2:
            print(f"⚠️ Very few available clients, reducing selected clients.")
            selected_clients = available_clients  # Use whoever is available
        else:
            selected_clients = random.sample(available_clients, min(2, len(available_clients)))

        for client_id in selected_clients:
            history = client_history[client_id]

            before_L = history["L"]
            before_H = history["H"]
            before_task = history["task"]
            before_epochs = history["epochs"]

            
            # Determine Training Rounds Dynamically
            num_epochs = compute_training_rounds(client_id, before_L) 
            success = True  # Assume success

            print(f"Client {client_id}: Assigned {num_epochs} epochs (CPU: {client_profiles[client_id]['CPU']} GHz, RAM: {client_profiles[client_id]['RAM']} GB)")

            round_training_time = 0  # Track cumulative training time per round

            client = FlowerClient(
                model=Net(),
                trainloader=trainloaders[client_id],
                testloader=testloader,
                valloader=valloaders[client_id],
                cid=client_id
            )

            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": before_L})
            client_training_time = train_metrics.get("training_time", 0)
            round_training_time += round(client_training_time, 2)  # Accumulate training time

            # **Failure Condition**
            success = False if train_metrics["training_time"] == 0 else (random.random() > 0.6)

            if success:
                after_task, after_epochs, after_L, after_H = adjust_task_assignment(client_id, success)
                status = "Trained"
            else:
                after_L, after_H, after_task, after_epochs = before_L, before_H, before_task, before_epochs
                status = "FAILED"

            # **Log Training Results**
            with open(log_file_path, "a") as log_file:
                log_file.write(f"{round_number},{client_id},{before_L},{before_H},{before_task},{before_epochs},"
                               f"{after_L},{after_H},{after_task},{after_epochs},{status}\n")

        # ✅ Simulate Failures After Training Round (Outside Loop)
        simulate_failures(args, unavailability_tracker, failure_log, round_number, round_training_time, selected_clients, available_clients)



        # **Edge Aggregation every k2 rounds**
        if round_number % args["k2"] == 0:
            print(f"🔹 Aggregating at EDGE SERVER (every {args['k2']} rounds)")
            for edge_server in edge_servers:
                aggregated_params = edge_server.aggregate()
                set_parameters(edge_server.model, aggregated_params)

        # **Global Aggregation every (k1 × k2) rounds**
        if round_number % (args["k1"] * args["k2"]) == 0:
            print(f"🌍 Aggregating at GLOBAL SERVER (every {args['k1'] * args['k2']} rounds)")
            global_weights = get_parameters(global_model)
            set_parameters(global_model, global_weights)

        # ✅ Display failures and recoveries at the end of each round
        display_failures_and_recoveries()




    # ✅ Print failed and recovered clients from main log file
        def display_failures_and_recoveries():
          df = pd.read_csv(log_file_path)
          failed_df = df[df["Failure Status"] == "FAILED"]
          recovered_df = df[df["Failure Status"] == "RECOVERED"]

          print("\n📋 Failed Clients:")
          print(failed_df[["Round", "Client ID", "Failure Duration"]])

          print("\n🔄 Recovered Clients:")
          print(recovered_df[["Round", "Client ID", "Recovery Time Remaining"]])

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
        'FAILURE_RATE': 0.2,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()
