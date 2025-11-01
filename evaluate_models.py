import csv

summary_file = "logs/all_model_results.csv"
best_model = None
best_accuracy = -1

with open(summary_file, "r") as f:
    reader = csv.DictReader(f)
    print("Model Performance Summary:\n")
    for row in reader:
        model = row["Model"]
        acc = float(row["Final Test Accuracy"])
        loss = float(row["Final Loss"])
        print(f"{model.upper():6} - Accuracy: {acc:.4f} | Loss: {loss:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            

print(f"\n✅ Best Model: {best_model.upper()} with Accuracy: {best_accuracy:.4f}")
