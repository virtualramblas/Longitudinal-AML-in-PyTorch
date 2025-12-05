import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from data_preprocessing import process_single_patient_data
from lstm_model import LSTMBinaryClassifier

def train_lstm_and_extract_importance(sequences, labels, patient, output_dir):
    # Ensure labels are of type int (if they are not already)
    labels = labels.astype(int)

    # Normalize data
    scaler = MinMaxScaler()
    sequences_scaled = scaler.fit_transform(sequences)

    # Reshape to (samples, timesteps, features)
    # For a single feature per timestep, input_size is 1
    sequences_scaled = sequences_scaled.reshape((sequences_scaled.shape[0], sequences_scaled.shape[1], 1))

    # Generate dummy labels matching samples (one per row/gene)
    labels = np.random.randint(0, 3, size=sequences.shape[0])

    # Convert to PyTorch tensors
    X = torch.tensor(sequences_scaled, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long) # Use Long for NLLLoss/CrossEntropyLoss

    # 70:15:15 split (train, validation, test)
    split_1 = int(0.7 * len(X))
    split_2 = int(0.85 * len(X))

    X_train, X_val, X_test = X[:split_1], X[split_1:split_2], X[split_2:]
    y_train, y_val, y_test = y[:split_1], y[split_1:split_2], y[split_2:]

    # Create TensorDataset
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # Create DataLoader
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate model, loss function, and optimizer
    input_size = X_train.shape[2]
    hidden_size = 64
    num_classes = len(np.unique(labels)) # Dynamically determine number of classes
    model = LSTMBinaryClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
    criterion = nn.NLLLoss() # Since LSTMBinaryClassifier uses log_softmax
    optimizer = optim.Adam(model.parameters())

    # Training loop
    epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\nStarting PyTorch training for patient: {patient}")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_accuracy = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculations during validation
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

    # Save training metrics
    history_df = pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_accuracy": train_accuracies,
        "val_accuracy": val_accuracies
    })
    history_df.to_csv(os.path.join(output_dir, f"{patient}_pytorch_training_metrics.csv"), index=False)

    # Evaluate on test data
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(targets.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_true_labels, all_predictions)
    report = classification_report(all_true_labels, all_predictions, target_names=[str(i) for i in range(num_classes)], zero_division=0)
    confusion = confusion_matrix(all_true_labels, all_predictions)

    # Save evaluation metrics
    with open(os.path.join(output_dir, f"{patient}_pytorch_evaluation_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{confusion}\n")

    return model, sequences

def extract_feature_importance(model, gene_names, patient):
    # Extract feature importance from LSTM weights
    # For PyTorch LSTM, input-to-hidden weights are typically in .weight_ih_l0
    # Shape: (4 * hidden_size, input_size)
    # The four sections correspond to input gate, forget gate, cell gate, output gate.
    # We are interested in the overall influence of each input feature.

    # Get input-to-hidden weights from the first LSTM layer
    lstm_input_weights = model.lstm.weight_ih_l0.data.cpu().numpy()

    mean_abs_lstm_weight = np.mean(np.abs(lstm_input_weights))
    feature_importance_per_gene = np.full(len(gene_names), mean_abs_lstm_weight)

    # Rank features by importance
    feature_ranking = pd.DataFrame({
        "Gene": gene_names,
        "Importance": feature_importance_per_gene
    }).sort_values(by="Importance", ascending=False)

    print(f"PyTorch training and evaluation for patient {patient} complete.")

    return feature_ranking

def train_model_and_rank_features(raw_data_dir, output_root_dir):
    # Organize files by patient
    processed_matrices_dir = raw_data_dir + "/processed_matrices_csv"
    output_dir = os.path.join(output_root_dir, 'lstm_bdm_analysisV2')
    os.makedirs(output_dir, exist_ok=True)
    file_groups = {}
    for file in os.listdir(processed_matrices_dir):
        if not file.endswith("_processed_matrix.csv"):
            continue
        parts = file.split("_")
        patient, state = parts[1], parts[2]
        file_groups.setdefault(patient, {})[state] = file

    # Run the analysis
    for patient, files in file_groups.items():
        sequences, labels, gene_names = process_single_patient_data(processed_matrices_dir, patient, files)
        model, sequences = train_lstm_and_extract_importance(sequences, labels, patient, 
                                                output_dir)
        
        feature_ranking = extract_feature_importance(model, gene_names, patient)
        
        # Save top 100 features
        top_100_features = feature_ranking.head(100)
        top_100_features.to_csv(os.path.join(output_dir, f"{patient}_top_100_features.csv"), index=False)

        print(f"Completed {patient}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a LSTM model for feature importance analysis.")
    parser.add_argument("--raw_data_dir", type=str, default=".", help="The root directory of the raw patient data.")
    parser.add_argument("--output_root_dir", type=str, default="./results", help="The root directory where to store training metrics and results.")

    args = parser.parse_args()

    train_model_and_rank_features(args.raw_data_dir, args.output_root_dir)