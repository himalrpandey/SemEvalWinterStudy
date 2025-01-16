import os
import json
import torch
from torch.optim import AdamW
from torch import nn

EARLY_STOPPING_PATIENCE = 5

def train_model_with_dynamics(
    model, train_loader, val_loader, device, epochs, patience=EARLY_STOPPING_PATIENCE, accumulation_steps=1):
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    best_loss = float('inf')
    no_improve_epochs = 0
    best_model = None

    # Training dynamics storage
    training_dynamics = {i: {"logits": [], "gold": None} for i in range(len(train_loader.dataset))}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)
            ids = batch["id"]

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Save logits and labels for training dynamics
            logits = outputs.logits.detach().cpu().numpy()
            for idx, logit, gold in zip(ids, logits, labels.cpu().numpy()):
                training_dynamics[idx]["logits"].append(logit.tolist())
                training_dynamics[idx]["gold"] = gold.tolist()

            # Accumulate gradients
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validate and early stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].unsqueeze(1).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            best_model = model.state_dict()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model)
                break

    return model, training_dynamics


def save_training_dynamics(output_dir, training_dynamics, filename="training_dynamics_epoch.jsonl"):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    with open(output_file, "w") as f:
        for instance_id, data in training_dynamics.items():
            entry = {
                "id": instance_id,  # Use the dataset ID for traceability
                "logits": [list(logit) for logit in data["logits"]],
                "gold": data["gold"],
            }
            f.write(json.dumps(entry) + "\n")
    print(f"Training dynamics saved to {output_file}")
