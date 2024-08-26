import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from convokit import Corpus, download
from utils import clean_utterance, split_utterance
import math


# Prepare dataset
def truncate_from_start(sequence, max_length):
    tokens = tokenizer.tokenize(sequence)
    if len(tokens) > max_length:
        tokens = tokens[-max_length:]  # Keep the last `max_length` tokens
    return tokenizer.convert_tokens_to_string(tokens)


class ConversationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        previous, current, label = self.data[idx]
        max_sequence_length = math.floor((self.max_length - 10)/2)
        previous_truncated = truncate_from_start(previous, max_sequence_length)
        current_truncated = truncate_from_start(current, max_sequence_length)
        encoding = self.tokenizer.encode_plus(
            f"[CLS] {previous_truncated} [SEP] {current_truncated} [SEP]",
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5

corpus = Corpus(filename=download("switchboard-processed-corpus"))

data = []
for conv in corpus.iter_conversations():
    utterances = list(conv.iter_utterances())
    for i, utt in enumerate(utterances):
        prev = ""
        if i > 0:
            prev = utterances[i-1].text
            prev = clean_utterance(prev)
        segments = split_utterance(utt.text)
        for j in range(len(segments)):
            cur = " ".join(segments[:j+1])
            data.append([prev, cur, j == len(segments) - 1])


print(f"The dataset has {len(data)} samples.")

# Split the data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Create datasets and dataloaders
train_dataset = ConversationDataset(train_data, tokenizer, MAX_LENGTH)
val_dataset = ConversationDataset(test_data, tokenizer, MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Setup optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_batches = len(train_dataloader)
print(f"The training dataset is divided into {num_batches} batches.")

for epoch in range(EPOCHS):
    model.train()
    for i, batch in enumerate(train_dataloader):
        print(f"\rCompleted batches: {i}/{num_batches}", end="", flush=True)

        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print()

    # Validation
    model.eval()
    val_preds = []
    val_true = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_true.extend(labels.cpu().numpy())

    val_accuracy = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_accuracy:.4f}")

    model.save_pretrained(f"./checkpoints/conversation_classifier_model_{epoch+1}")
    tokenizer.save_pretrained(f"./checkpoints/conversation_classifier_model_{epoch+1}")

# Final evaluation
print("\nClassification Report:")
print(classification_report(val_true, val_preds, target_names=['Not Finished', 'Finished']))