import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import os

EMOTIONS = ['anger', 'fear', 'joy', 'sadness', 'surprise']
#Threshold proportional to positive labeling in dataset
THRESHOLDS_ENG = {'joy': 0.24, 'anger': 0.12, 'sadness': 0.32, 'surprise': 0.30, 'fear': 0.58}
BATCH_SIZE = 32
EPOCHS = 20
NUM_WORKERS = 2  # Number of workers for DataLoader and gpus
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EARLY_STOPPING_PATIENCE = 20

torch.set_num_threads(NUM_WORKERS) # Set the number of threads for PyTorch DataLoader

#train_eng = pd.read_csv('public_data_dev/track_a/train/eng.csv')
dev_eng = pd.read_csv('public_data_dev/track_a/dev/eng.csv')

class EmotionDataset(Dataset):
    """
    This class is used to create a PyTorch dataset from the texts and labels.
    """
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    """
    This function returns the input_ids, attention_mask, and labels for the given index.
    """
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

"""
This function initializes the model and tokenizer. 
"""
def initialize_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-cased", num_labels=1
    )
    return tokenizer, model

"""
This function trains the model.
args:
model: The model to train
train_loader: DataLoader for training data
device: Device to use for training
epochs: Number of epochs to train
patience: Number of epochs to wait for improvement before early stopping
returns:
model: Trained model
"""
def train_model(model, train_loader, device, epochs, patience=EARLY_STOPPING_PATIENCE):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss() # Binary cross-entropy loss
    model.to(device)

    #training loop to calculate loss for early stopping
    best_loss = float('inf')
    no_improve_epochs = 0
    best_model = None

    #loop through epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        #loop through batches
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader) #average loss for epoch
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        #early stopping
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            no_improve_epochs = 0
            best_model = model.state_dict()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("Early stopped")
                model.load_state_dict(best_model)
                break

    return model

"""
This function gets predictions from the model.
args:
    data_loader: DataLoader for the data
    model: Model to get predictions from
    device: Device to use for predictions
returns:
    predictions: Predictions from the model
"""
def get_predictions(data_loader, model, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            predictions.extend(probs.flatten())

    return np.array(predictions)

"""
The main function to train the model and get predictions.
"""
def main():
    final_predictions = pd.DataFrame()
    final_predictions["id"] = dev_eng["id"]

    for emotion in EMOTIONS:
        print(f"\n{emotion}")

        easy_csv = os.path.join("cartography_output", f"{emotion}_easy-to-learn.csv")
        train_easy_df = pd.read_csv(easy_csv)
        print(f"{emotion} easy-to-learn size:", len(train_easy_df)) #debug
        
        train_texts, train_labels = train_easy_df["text"].tolist(), train_easy_df[emotion].values
        dev_texts = dev_eng["text"].tolist()

        tokenizer, model = initialize_model_and_tokenizer()

        train_dataset = EmotionDataset(train_texts, train_labels, tokenizer)
        dev_dataset = EmotionDataset(dev_texts, np.zeros(len(dev_texts)), tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = train_model(model, train_loader, DEVICE, EPOCHS)

        y_probs = get_predictions(dev_loader, model, DEVICE)

        threshold = THRESHOLDS_ENG[emotion]
        y_pred = (y_probs > threshold).astype(int)
        final_predictions[emotion] = y_pred

    # Save predictions the format for the comp
    output_csv_file = "pred_eng_a.csv"
    final_predictions.to_csv(output_csv_file, index=False)
    print(f"\nPredictions saved to {output_csv_file}")

if __name__ == "__main__":
    main()
