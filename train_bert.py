import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # Added for a nice progress bar
import os

# --- 1. SETTINGS ---
MODEL_NAME = 'bert-base-multilingual-cased'
SAVE_DIR = 'bert_emotion_model'
BATCH_SIZE = 16
EPOCHS = 4 
MAX_LEN = 64

# --- 2. DATA PREPARATION ---
def prepare_data():
    mapping_df = pd.read_csv('emotion_dataset.csv')
    emoji_df = pd.read_csv('emoji_dataset.csv')
    
    emotion_map = {'Depressed': 0, 'Sad': 1, 'Neutral': 2, 'Happy': 3, 'Excitement': 4}
    
    mapping_df['label'] = mapping_df['emotion'].str.strip().map(emotion_map)
    mapping_df = mapping_df.dropna(subset=['label'])
    
    texts = mapping_df['message'].astype(str).tolist()
    labels = mapping_df['label'].astype(int).tolist()
    
    emoji_cols = ['Depressed', 'Sad', 'Neutral', 'Happy', 'Excitement']
    for _, row in emoji_df.iterrows():
        texts.append(str(row['emoji']))
        labels.append(np.argmax(row[emoji_cols].values))
        
    return texts, labels

# --- 3. DATASET CLASS ---
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        # FIXED: Calling the tokenizer directly instead of .encode_plus
        encoding = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            return_token_type_ids=False, 
            padding='max_length',
            truncation=True, 
            return_attention_mask=True, 
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. TRAINING EXECUTION ---
def train_model():
    print("🚀 Starting BERT Fine-Tuning...")
    texts, labels = prepare_data()
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_data = EmotionDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Added tqdm progress bar for each epoch
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            loop.set_description(f"Epoch [{epoch + 1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())
        
        print(f"✅ Epoch {epoch + 1} Average Loss: {total_loss/len(train_loader):.4f}")

    # Save Model and Tokenizer together
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"\n🎉 BERT Model saved to folder: {SAVE_DIR}/")

if __name__ == "__main__":
    train_model()