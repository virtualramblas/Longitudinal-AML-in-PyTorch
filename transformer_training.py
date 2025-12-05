import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import process_single_patient_data
from transformer_model import PyTorchTransformerModel

def train_transformer(sequences, labels, patient,
                        output_dir, device='cpu'):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Training PyTorch Transformer for patient {patient}...")

    # Normalize data
    scaler = MinMaxScaler()
    # Flatten sequences for scaler, then reshape back
    original_shape = sequences.shape
    sequences_reshaped_for_scaler = sequences.reshape(-1, sequences.shape[-1]) # (num_samples * timesteps, features)
    sequences_scaled_flattened = scaler.fit_transform(sequences_reshaped_for_scaler)
    sequences = sequences_scaled_flattened.reshape(original_shape)

    # Reshape to (samples, timesteps, features) - already in this shape if sequences is (num_genes, num_timepoints, 1)
    # The Keras model assumes `sequences.shape` is `(num_genes, num_timepoints)`. After reshaping it to `(num_genes, num_timepoints, 1)`
    # The provided `sequences` from `process_patient` would be `(num_genes, 3)`
    # so we need to reshape it to `(num_genes, 3, 1)`
    if len(sequences.shape) == 2:
        sequences = sequences.reshape((sequences.shape[0], sequences.shape[1], 1))

    # Generate dummy labels matching samples (one per row/gene) if actual labels are not provided or are problematic
    if labels is None or labels.shape[0] != sequences.shape[0]:
        labels = np.random.randint(0, 3, size=sequences.shape[0])  # Placeholder for actual labels for each gene

    X_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(labels, dtype=torch.long).to(device)

    # 70:15:15 split (train, validation, test)
    total_samples = X_tensor.shape[0]
    split_1 = int(0.7 * total_samples)
    split_2 = int(0.85 * total_samples)

    X_train, y_train = X_tensor[:split_1], y_tensor[:split_1]
    X_val, y_val = X_tensor[split_1:split_2], y_tensor[split_1:split_2]
    X_test, y_test = X_tensor[split_2:], y_tensor[split_2:]

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 16 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build PyTorch Transformer model
    embed_dim = X_train.shape[-1]  # Features dimension
    num_heads = 1
    num_classes = X_train.shape[-2] # DX, REL, REM
    epochs = 50

    model = PyTorchTransformerModel(embed_dim=embed_dim, num_heads=num_heads, num_classes=num_classes).to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()

        avg_train_loss = train_loss / total_train
        train_accuracy = correct_train / total_train

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)

                val_loss += loss_val.item() * batch_X_val.size(0)
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += batch_y_val.size(0)
                correct_val += (predicted_val == batch_y_val).sum().item()

        avg_val_loss = val_loss / total_val
        val_accuracy = correct_val / total_val

        history["loss"].append(avg_train_loss)
        history["accuracy"].append(train_accuracy)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Save training metrics
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, f"{patient}_pytorch_training_metrics.csv"), index=False)

    # Evaluate on test data
    model.eval() # Set model to evaluation mode
    y_pred_list = []
    y_test_list = []
    with torch.no_grad():
        for batch_X_test, batch_y_test in test_loader:
            outputs_test = model(batch_X_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            y_pred_list.extend(predicted_test.cpu().numpy())
            y_test_list.extend(batch_y_test.cpu().numpy())

    y_pred = np.array(y_pred_list)
    y_test_np = np.array(y_test_list)

    accuracy = accuracy_score(y_test_np, y_pred)
    report = classification_report(y_test_np, y_pred, target_names=["DX", "REL", "REM"], zero_division=0)
    confusion = confusion_matrix(y_test_np, y_pred)

    # Save evaluation metrics
    with open(os.path.join(output_dir, f"{patient}_pytorch_evaluation_metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{confusion}\n")

    print(f"PyTorch training and evaluation for patient {patient} completed. Metrics saved.")

    return model, sequences

def rank_features_by_importance(model, sequences, gene_names):
    model.eval()
    with torch.no_grad(): 
        attention_layer_for_inference = model.multi_head_attention
        outputs, attention_weights = attention_layer_for_inference(sequences, return_attention_weights=True)
        feature_importance_tensor = torch.mean(torch.abs(outputs), dim=(1, 2))
        feature_importance = feature_importance_tensor.cpu().numpy()

    # Rank features by importance
    feature_ranking = pd.DataFrame({
        "Gene": gene_names,
        "Importance": feature_importance
    }).sort_values(by="Importance", ascending=False)

    return feature_ranking

def train_model_and_rank_features(raw_data_dir, output_root_dir):
    device = "cpu"
    if torch.cuda.is_available():
        device = "gpu"
    else:
        if torch.mps.is_available():
            device = "mps"

    # Organize files by patient
    processed_matrices_dir = raw_data_dir + "/processed_matrices_csv"
    output_dir = os.path.join(output_root_dir, 'transformer_analysis')
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
        model, sequences = train_transformer(sequences, labels, patient, 
                                                output_dir, device)
        
        sequences_to_torch = torch.from_numpy(sequences)
        sequences_to_torch_fp32 = sequences_to_torch.to(dtype=torch.float32, device=device)
        feature_ranking = rank_features_by_importance(model, sequences_to_torch_fp32, gene_names)
        
        # Save top 100 features
        top_100_features = feature_ranking.head(100)
        top_100_features.to_csv(os.path.join(output_dir, f"{patient}_top_100_features.csv"), index=False)

        print(f"Completed {patient}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training a Transformer model for feature importance analysis.")
    parser.add_argument("--raw_data_dir", type=str, default=".", help="The root directory of the raw patient data.")
    parser.add_argument("--output_root_dir", type=str, default="./results", help="The root directory where to store training metrics and results.")

    args = parser.parse_args()

    train_model_and_rank_features(args.raw_data_dir, args.output_root_dir)