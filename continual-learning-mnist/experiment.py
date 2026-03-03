import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

from data import get_datasets, split_dataset, create_replay_buffer
from models import SimpleCNN, SmallCNN
from train import train, evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load datasets
train_dataset, test_dataset = get_datasets()

train_A, train_B = split_dataset(train_dataset)
test_A, test_B = split_dataset(test_dataset)

train_loader_A = DataLoader(train_A, batch_size=64, shuffle=True)
test_loader_A = DataLoader(test_A, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()


def run_experiment(model_class, buffer_size):
    model = model_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train Task A
    train(model, train_loader_A, optimizer, criterion, device)
    acc_A_after_A = evaluate(model, test_loader_A, device)

    # Replay buffer
    replay_buffer = create_replay_buffer(train_A, buffer_size)
    combined_dataset = ConcatDataset([train_B, replay_buffer])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # Train Task B
    train(model, combined_loader, optimizer, criterion, device)

    acc_A_after_B = evaluate(model, test_loader_A, device)

    forgetting = acc_A_after_A - acc_A_after_B
    bwt = acc_A_after_B - acc_A_after_A

    return acc_A_after_A, acc_A_after_B, forgetting, bwt

if __name__ == "__main__":

    print("Running from directory:", os.getcwd())

    buffer_sizes = [100, 500, 1000, 2000]

    os.makedirs("results", exist_ok=True)

    csv_path = "results/metrics.csv"

    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Buffer Size", "Acc_A_after_A", "Acc_A_after_B", "Forgetting", "BWT"])
        for buffer_size in buffer_sizes:

            print(f"\n===== Buffer Size: {buffer_size} =====")

            # Large model
            print("\n--- Large Model ---")
            large_results = run_experiment(SimpleCNN, buffer_size)

            writer.writerow([
                "LargeCNN",
                buffer_size,
                large_results[0],
                large_results[1],
                large_results[2],
                large_results[3]
            ])

            print("Large Model Forgetting:", large_results[2])

            # Small model
            print("\n--- Small Model ---")
            small_results = run_experiment(SmallCNN, buffer_size)

            writer.writerow([
                "SmallCNN",
                buffer_size,
                small_results[0],
                small_results[1],
                small_results[2]
            ])

            print("Small Model Forgetting:", small_results[2])

    print("\nResults saved to results/metrics.csv")

    # Generate plot
    df = pd.read_csv("results/metrics.csv")

    plt.figure()

    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        plt.plot(subset["Buffer Size"], subset["Forgetting"], marker='o', label=model)

    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Forgetting (%)")
    plt.title("Forgetting vs Buffer Size")
    plt.legend()

    plt.savefig("results/forgetting_vs_buffer.png")
    plt.close()

    print("Plot saved to results/forgetting_vs_buffer.png")


    # Generate BWT plot
    plt.figure()

    for model in df["Model"].unique():
        subset = df[df["Model"] == model]
        plt.plot(subset["Buffer Size"], subset["BWT"], marker='o', label=model)

    plt.xlabel("Replay Buffer Size")
    plt.ylabel("Backward Transfer (BWT)")
    plt.title("BWT vs Buffer Size")
    plt.legend()

    plt.savefig("results/bwt_vs_buffer.png")
    plt.close()

    print("Plot saved to results/bwt_vs_buffer.png")