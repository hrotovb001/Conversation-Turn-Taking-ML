import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Constants
threshold = 0.25

# Load the saved model and tokenizer
model_path = './checkpoints/conversation_classifier_model_3'
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Set the device
device = torch.device('cpu')
model.to(device)


# Function to predict on a single sample
def speaker_done_probability(previous_utterance, current_utterance):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            f"[CLS] {previous_utterance} [SEP] {current_utterance} [SEP]",
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        probabilities = torch.softmax(outputs.logits, dim=1)

        return probabilities[:, 1]
