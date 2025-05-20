pip install pandas scikit-learn transformers torch tqdm openpyxl

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load data
df = pd.read_excel('Violation_Training_Data.xlsx')

train, val = train_test_split(df, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train['Combined Text Data'].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=512)
val_encodings = tokenizer(val['Combined Text Data'].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=512)

factor = 'Violation Factor'

train_input_ids, train_attention_masks, train_targets = train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train[factor].values)
val_input_ids, val_attention_masks, val_targets = val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val[factor].values)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
batch_size = 16
train_dataloader = DataLoader(TensorDataset(train_input_ids, train_attention_masks, train_targets), batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(TensorDataset(val_input_ids, val_attention_masks, val_targets), batch_size=batch_size)
num_epochs = 30
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
scaler = torch.cuda.amp.GradScaler()

# Training and Validation Loop
gradient_accumulation_steps = 2  
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    train_preds, train_labels = [], []

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        total_loss += loss.item() * gradient_accumulation_steps

        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(train_dataloader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_labels, train_preds, average='binary')

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}')

model.eval()
val_preds, val_labels = [], []
with torch.no_grad():
    for batch in tqdm(val_dataloader, desc='Validating'):
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids, attention_mask=attention_mask)  # Corrected line here
        preds = torch.argmax(outputs.logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.cpu().numpy())

val_acc = accuracy_score(val_labels, val_preds)
val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
print(f'Validation Accuracy: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

# Save the model and tokenizer
model_save_path = 'bert_model.pt'
tokenizer_save_path = 'bert_tokenizer'
torch.save(model.state_dict(), model_save_path)
tokenizer.save_pretrained(tokenizer_save_path)

print(f'Model saved to {model_save_path}')
print(f'Tokenizer saved to {tokenizer_save_path}')
