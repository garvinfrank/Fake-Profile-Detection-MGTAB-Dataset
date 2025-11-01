import os
import csv
import torch
import argparse
from models.gcn import GCN
from models.gat import GAT
from models.sage import GraphSAGE
# from models.rgcn import RGCN
from models.gin import GIN
from utils.engine import train, test
import warnings

if __name__ == "__main__":
    print("-----------main.py started-----------")

    # Load data
    data = torch.load("mgtab_graph_split.pt")
    input_dim = data.x.shape[1]
    hidden_dim = 64
    output_dim = 2  # binary classification

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gcn', 'gat', 'sage', 'rgcn', 'gin'], required=True)
    parser.add_argument('--epochs', type=int, default=200)
    args = parser.parse_args()

    # Model selection
    if args.model == 'gcn':
        model = GCN(input_dim, hidden_dim, output_dim)
    elif args.model == 'gat':
        model = GAT(input_dim, hidden_dim, output_dim)
    elif args.model == 'sage':
        model = GraphSAGE(input_dim, hidden_dim, output_dim)
    # elif args.model == 'rgcn':
    #     num_relations = int(data.edge_type.max()) + 1
    #     model = RGCN(input_dim, hidden_dim, output_dim, num_relations)
    elif args.model == 'gin':
        model = GIN(input_dim, hidden_dim, output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Log files
    metrics_log = f"logs/{args.model}_metrics.csv"
    summary_log = "logs/all_model_results.csv"
    summary_exists = os.path.exists(summary_log)

    # Start logging metrics
    with open(metrics_log, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Test Accuracy"])
        for epoch in range(args.epochs):
            loss = train(model, data, optimizer)
            acc = test(model, data)
            writer.writerow([epoch, loss, acc])
            if epoch % 20 == 0 or epoch == args.epochs - 1:
                print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Test Acc: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), f"{args.model}_model.pt")
    print(f"{args.model.upper()} model saved.")

    # Save final results to master summary file
    with open(summary_log, "a", newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(["Model", "Epochs", "Final Loss", "Final Test Accuracy"])
        writer.writerow([args.model, args.epochs, f"{loss:.4f}", f"{acc:.4f}"])
    print(f"Summary updated in {summary_log}")
