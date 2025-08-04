import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import re
import time
import os

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA devices detected")

# Use GPU (might want to update CUDA and transformers library)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# answerdotai/ModernBERT-base
MODEL_NAME = os.path.join(os.path.dirname(__file__), "answerdotai")

'''
@misc{modernbert,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference}, 
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663}, 
}
'''

PATH_REPORTS = '../../../Data/NM-NLP/TCGA_Reports.csv'
PATH_CANCER_TYPES = '../../../Data/NM-NLP/tcga_patient_to_cancer_type.csv'

# Training Hyperparameters
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_STATE = 42

# Cleans texs via lowercase and normalize the whitespace
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()

    # Remove special chars
    text = re.sub(r'[^a-z0-9\s\.]', '', text)

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Processes Pathology_Reports.csv for training
def prep_data(reports_path, types_path):
    print("Loading and preparing data...")

    try:
        df_reports = pd.read_csv(reports_path)
        df_types = pd.read_csv(types_path)
        print(f"Loaded {len(df_reports)} reports and {len(df_types)} cancer type entries.")

    except FileNotFoundError as e:
        print(f"{e}")
        return None, None

    # Debug: Check the column names and sample data. Can probably comment out when done.
    print("Reports columns:", df_reports.columns.tolist())
    print("Cancer types columns:", df_types.columns.tolist())
    print("\nSample patient_filename from reports:")
    print(df_reports['patient_filename'].head())
    print("\nSample patient_id from cancer types:")
    print(df_types['patient_id'].head())

    # Split on period to get just the patient ID part
    # Example: 'TCGA-OR-A5J1.8866FD87-4F6F-4D7E-B99A-7DD427ED3BB3' -> 'TCGA-OR-A5J1' so it can sync up with patient dataset
    df_reports['patient_barcode'] = df_reports['patient_filename'].str.split('.').str[0]
    
    print("\nExtracted patient barcodes from reports:")
    print(df_reports['patient_barcode'].head())
    
    reports_patients = set(df_reports['patient_barcode'].unique())
    types_patients = set(df_types['patient_id'].unique())
    overlap = reports_patients.intersection(types_patients)
    
    print(f"\nPatients in reports: {len(reports_patients)}")
    print(f"Patients in cancer types: {len(types_patients)}")
    print(f"Overlapping patients: {len(overlap)}")

    # Merge the two dataframes
    df_merged = pd.merge(df_reports, df_types, left_on='patient_barcode', right_on='patient_id', how='inner')
    print(f"After merge: {len(df_merged)} samples")

    # Debug
    if len(df_merged) == 0:
        print("ERROR: Merge resulted in 0 samples!")
        return None, None

    df_merged['cleaned_text'] = df_merged['text'].apply(clean_text)

    # Drop rows with missing text or cancer types
    initial_count = len(df_merged)
    df_merged.dropna(subset=['cleaned_text', 'cancer_type'], inplace=True)
    df_merged = df_merged[df_merged['cleaned_text'] != '']
    print(f"After cleaning: {len(df_merged)} samples (removed {initial_count - len(df_merged)} empty/invalid entries)")
    
    # Check cancer type distribution
    print("\nCancer type distribution:")
    print(df_merged['cancer_type'].value_counts())
    
    label_encoder = LabelEncoder()
    df_merged['label'] = label_encoder.fit_transform(df_merged['cancer_type'])

    print(f"Data prepared. Found {len(df_merged)} samples and {len(label_encoder.classes_)} unique cancer types.")
    
    return df_merged, label_encoder

# Class for TCGA pathology report. Handles tokens and formatting of text for model input.
class TCGAPathologyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    # Debug: Just returns number of samples
    def __len__(self):
        return len(self.texts)

    # Debug: Returns a tokenized sample
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
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

# Does one epoch for training. Also returns the accuracy.
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

# Evaluates the performance of the model for val/test.
def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

# Generates a prediction. Main purpose is for final test.
def get_predictions(model, data_loader, device):
    model = model.eval()
    
    texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts_batch = d["text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            probs = torch.nn.functional.softmax(logits, dim=1)

            texts.extend(texts_batch)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(labels)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return texts, predictions, prediction_probs, real_values

# Runs everything...
def main():    
    # Load and prepare data
    df, label_encoder = prep_data(PATH_REPORTS, PATH_CANCER_TYPES)
    if df is None:
        return

    num_classes = len(label_encoder.classes_)

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label'])
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_STATE, stratify=df_test['label'])

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    # Initialize tokenizer
    print(f"Loading tokenizer from local path: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError as e:
        print(f"{e}")
        return

    # Create DataLoaders
    train_dataset = TCGAPathologyDataset(
        texts=df_train.cleaned_text.to_numpy(),
        labels=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = TCGAPathologyDataset(
        texts=df_val.cleaned_text.to_numpy(),
        labels=df_val.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    test_dataset = TCGAPathologyDataset(
        texts=df_test.cleaned_text.to_numpy(),
        labels=df_test.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # Initialize model
    print(f"Loading model from local path: {MODEL_NAME}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            output_attentions=False,
            output_hidden_states=False,
            local_files_only=True
        )
    except OSError as e:
        print(f"{e}")
        return
        
    model.to(device)

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Training loop
    best_accuracy = 0
    epoch_times = []
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        progress_percent = (epoch / EPOCHS) * 100
        
        print(f'Epoch {epoch + 1}/{EPOCHS} ({progress_percent:.1f}% complete)')
        print('-' * 50)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        epoch_duration = time.time() - epoch_start_time
        epoch_times.append(epoch_duration)
        
        # Calculate ETA
        if epoch > 0:
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = EPOCHS - (epoch + 1)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_minutes = eta_seconds / 60
            eta_hours = eta_minutes / 60
            
            if eta_hours >= 1:
                eta_str = f"{eta_hours:.1f}h"
            elif eta_minutes >= 1:
                eta_str = f"{eta_minutes:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"
        else:
            eta_str = "Calculating...!?"
        
        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f} (took {epoch_duration:.2f}s)')

        val_acc, val_loss = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val)
        )
        print(f'Val   loss {val_loss:.4f} accuracy {val_acc:.4f}')
        
        # Progress and ETA info
        completed_percent = ((epoch + 1) / EPOCHS) * 100
        print(f'Progress: {completed_percent:.1f}% complete | ETA: {eta_str}')
        
        if val_acc > best_accuracy:
            # Save the best model
            torch.save(model.state_dict(), 'model_finetune.bin')
            best_accuracy = val_acc
        
        print()
    
    print("\nTraining complete.")
    
    # Evaluation on Test Set
    print("\nEvaluating on the test set...")
    # Load the best model state for final evaluation
    model.load_state_dict(torch.load('model_finetune.bin'))
    
    _, y_pred, _, y_test = get_predictions(model, test_data_loader, device)
    
    # Decode labels for reporting
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    

    # Print stuff
    print(f'\nTest Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded):.4f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, target_names=label_encoder.classes_))
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(conf_matrix_df)

if __name__ == '__main__':
    main()