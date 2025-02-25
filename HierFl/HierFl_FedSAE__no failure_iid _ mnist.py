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
csv_filename = "client_task_log.csv"
csv_path = os.path.join(os.getcwd(), csv_filename)  # Saves in the working directory

if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
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
# Adjust L and H After Each Round
########################################
def adjust_task_assignment(client_id, success):
    """Adjust L, H, and task difficulty based on client execution ability."""
    global client_history

    if client_id not in client_history:
        print(f"⚠️ Warning: Client {client_id} missing from history. Initializing now.")
        client_history[client_id] = {"L": 50, "H": 100, "task": "Easy", "epochs": 50}

    history = client_history[client_id]

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
      """Train the model for a given number of epochs."""
      global client_history 
      self.set_parameters(parameters)
      self.model.train()
      optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
      criterion = nn.CrossEntropyLoss()

      client_id = self.cid
      num_epochs = config["num_epochs"]

      print(f"Client {client_id}: Training for {num_epochs} epochs.")

      start_time = time.time()

      for epoch in range(num_epochs):
          for inputs, labels in self.trainloader:
              inputs, labels = inputs.to(torch.device('cpu')), labels.to(torch.device('cpu'))
              optimizer.zero_grad()
              outputs = self.model(inputs)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

      end_time = time.time()
      training_time = end_time - start_time

      # ✅ Ensure client history exists before updating training_time
      if client_id not in client_history:
          client_history[client_id] = {"training_time": []}
      elif "training_time" not in client_history[client_id]:
          client_history[client_id]["training_time"] = []

      client_history[client_id]["training_time"].append(training_time)

      print(f"Client {client_id}: Training completed in {training_time:.2f} sec.")

      return self.get_parameters(), len(self.trainloader.dataset), {"training_time": training_time}





########################################
# Hierarchical Federated Learning with Flower
#######################################
def HierFL(args, trainloaders, valloaders, testloader):

    # Initialize Edge Devices
    edge_devices = [EdgeDevice(i, trainloaders[i], valloaders[i]) for i in range(args['NUM_DEVICES'])]
    num_edge_servers = args['NUM_EDGE_SERVERS']
    edge_servers = []
    devices_per_server = len(edge_devices) // num_edge_servers

    for i in range(num_edge_servers):
        start_idx = i * devices_per_server
        devices = edge_devices[start_idx:] if i == num_edge_servers - 1 else edge_devices[start_idx: start_idx + devices_per_server]
        edge_servers.append(EdgeServer(i, devices))

    global client_history

    # ✅ Ensure all clients are initialized
    for client_id in range(args['NUM_DEVICES']):
        if client_id not in client_history:
            client_history[client_id] = {
                "L": 50, "H": 70, "epochs": 50, "task": "Easy",
                "training_time": [], "accuracy": []
            }

    # Global Model
    global_model = Net()
    global_weights = get_parameters(global_model)

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

    # Start Federated Training
    for round_number in range(1, args['GLOBAL_ROUNDS'] + 1):
        selected_clients = random.sample(range(args['NUM_DEVICES']), min(2, args['NUM_DEVICES']))
        print(f"\n🔵 Round {round_number}: Selected Clients -> {selected_clients}")

        for client_id in selected_clients:
            history = client_history[client_id]

            before_L = history["L"]
            before_H = history["H"]
            before_task = history["task"]
            before_epochs = history["epochs"]

            # **Decide Training Strategy**
            num_epochs = before_L  # Default to L
            success = True  # Assume success
            
            print(f"Client {client_id}: Assigned task {before_task} with {num_epochs} epochs.")

            # **Train Client**
            client = FlowerClient(
                model=Net(),
                trainloader=trainloaders[client_id],
                testloader=testloader,
                cid=client_id,
            )
            _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})

            training_time = train_metrics["training_time"]

            # ✅ Ensure `training_time` key exists before updating
            if "training_time" not in client_history[client_id]:
                client_history[client_id]["training_time"] = []
                
            client_history[client_id]["training_time"].append(training_time)

            # **Check if Client Can Attempt H**
            if before_task == "Easy":
                num_epochs = before_H
                print(f"Client {client_id}: Trying difficult task with {num_epochs} epochs.")

                _, _, train_metrics = client.fit(get_parameters(client.model), {"num_epochs": num_epochs})
                training_time = train_metrics["training_time"]

                # Simulated failure condition
                if random.random() < 0.6:
                    success = False

            # **Update L and H Based on Success**
            if success:
                history["L"] += 10
                history["H"] += 10
                history["task"] = "Difficult"
            else:
                history["L"] = max(10, before_L // 2)
                history["H"] = max(20, before_H // 2)
                history["task"] = "Easy"

            # Store updates in history
            history["epochs"] = history["L"]

            # Log to CSV
            with open('client_task_log.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                if os.stat("client_task_log.csv").st_size == 0:
                    writer.writerow(["Round", "Client ID", "Before Task", "Before Epochs", "Before L", "Before H",
                                     "After Task", "After Epochs", "After L", "After H"])
                writer.writerow([round_number, client_id, before_task, before_epochs, before_L, before_H,
                                 history["task"], history["epochs"], history["L"], history["H"]])

            print(f"Client {client_id}: Finished round with new L={history['L']}, H={history['H']}, Task={history['task']}")

    print(f"✅ Federated Training Complete! Results saved to client_task_log.csv.")



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
        split = torch.split(indices, len(train_dataset) // num_clients)

        trainloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED)) for s in split]
        valloaders = [DataLoader(torch.utils.data.Subset(train_dataset, s), batch_size=batch_size, shuffle=False) for s in split]
        testloader = DataLoader(test_dataset, batch_size=batch_size)
        return trainloaders, valloaders, testloader

    args = {
        'NUM_DEVICES': 5,
        'NUM_EDGE_SERVERS': 5,
        'GLOBAL_ROUNDS':3,
        'LEARNING_RATE': 0.001,
        'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'CLIENT_FRACTION': 0.2,
        'EVALUATE_FRACTION': 0.2,
        'k1': 60,  # Local updates parameter
        'k2': 1  # Edge-to-cloud aggregation frequency
    }

    trainloaders, valloaders, testloader = load_datasets(args['NUM_DEVICES'])
    HierFL(args, trainloaders, valloaders, testloader)

if __name__ == "__main__":
    main()
