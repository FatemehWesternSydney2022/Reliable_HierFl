import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the data from CSV files 
dataset1 = pd.read_csv("HierFl_FedSAE_Homogenous_Mnist_non_iid.csv")  # Dataset 1
dataset2 = pd.read_csv("HierFl_FedSAE_Homogenous_Mnist_20% failure_non_iid.csv")  # Dataset 2
dataset3 = pd.read_csv("HierFl_FedSAE_Homogenous_Mnist_40% failure_non_iid.csv")  # Dataset 3




# Extract the "Accuracy" column from each dataset
accuracy1 = dataset1["Accuracy"]
accuracy2 = dataset2["Accuracy"]
accuracy3 = dataset3["Accuracy"]



# Plotting all four datasets
plt.figure(figsize=(12, 6))
plt.plot(accuracy1, label="HierFl_FedSAE_no failure _ mnist_ Homogenous_non_iid", color='blue')
plt.plot(accuracy2, label="HierFl_FedSAE_20% failure _ mnist_ Homogenous_non_iid", color='orange')
plt.plot(accuracy3, label="HierFl_FedSAE_40% failure _ mnist_ Homogenous_non_iid", color='green')



# Adding titles and labels
plt.title("Homogenous HierFl_FedSAE_mnist , no failure , 20% failure, and 40% failure, non_iid")
plt.xlabel("Global Rounds")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

# Show the plot
plt.show()





