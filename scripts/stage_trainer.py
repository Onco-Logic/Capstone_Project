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
import argparse

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("No CUDA devices detected")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ModernBERT base model path
MODEL_NAME = os.path.join(os.path.dirname(__file__), "answerdotai")

# Data file paths
PATH_REPORTS = '../../../Data/NM-NLP/TCGA_Reports.csv'
PATH_T_STAGE = '../../../Data/NM-NLP/TCGA_T14_patients.csv'
PATH_N_STAGE = '../../../Data/NM-NLP/TCGA_N03_patients.csv'
PATH_M_STAGE = '../../../Data/NM-NLP/TCGA_M01_patients.csv'

# Training parameters
MAX_LEN = 512
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 2e-5
RANDOM_STATE = 42

def clean_text(text):
    # Basic text cleanup - lowercase and normalize whitespace
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Remove special characters but keep periods
    text = re.sub(r'[^a-z0-9\s\.]', '', text)
    # Replace multiple spaces/newlines with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prep_tnm_data(reports_path, stage_path, stage_column, stage_name):
    # Load and merge pathology reports with TNM staging data
    print(f"\nLoading and preparing {stage_name} staging data...")

    try:
        df_reports = pd.read_csv(reports_path)
        df_stage = pd.read_csv(stage_path)
        print(f"Loaded {len(df_reports)} reports and {len(df_stage)} {stage_name} staging entries.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

    # Debug: Check the column names and sample data
    print(f"Reports columns: {df_reports.columns.tolist()}")
    print(f"{stage_name} staging columns: {df_stage.columns.tolist()}")
    print(f"\nSample patient_filename from reports:")
    print(df_reports['patient_filename'].head())
    print(f"\nSample case_submitter_id from {stage_name} staging:")
    print(df_stage['case_submitter_id'].head())

    # Extract patient ID from reports filename
    # 'TCGA-BP-5195.25c0b433-5557-4165-922e-2c1eac9c26f0' -> 'TCGA-BP-5195'
    df_reports['patient_barcode'] = df_reports['patient_filename'].str.split('.').str[0]
    
    print(f"\nExtracted patient barcodes from reports:")
    print(df_reports['patient_barcode'].head())
    
    # Check overlap
    reports_patients = set(df_reports['patient_barcode'].unique())
    stage_patients = set(df_stage['case_submitter_id'].unique())
    overlap = reports_patients.intersection(stage_patients)
    
    print(f"\nPatients in reports: {len(reports_patients)}")
    print(f"Patients in {stage_name} staging: {len(stage_patients)}")
    print(f"Overlapping patients: {len(overlap)}")

    # Merge dataframes
    df_merged = pd.merge(
        df_reports, 
        df_stage, 
        left_on='patient_barcode', 
        right_on='case_submitter_id', 
        how='inner'
    )
    print(f"After merge: {len(df_merged)} samples")

    if len(df_merged) == 0:
        print("ERROR: Merge resulted in 0 samples!")
        return None, None

    # Clean text
    df_merged['cleaned_text'] = df_merged['text'].apply(clean_text)

    # Drop rows with missing text or staging info
    initial_count = len(df_merged)
    df_merged.dropna(subset=['cleaned_text', stage_column], inplace=True)
    df_merged = df_merged[df_merged['cleaned_text'] != '']
    print(f"After cleaning: {len(df_merged)} samples (removed {initial_count - len(df_merged)} empty/invalid entries)")
    
    # Check staging distribution
    print(f"\n{stage_name} staging distribution:")
    print(df_merged[stage_column].value_counts())
    
    # Encode labels
    label_encoder = LabelEncoder()
    df_merged['label'] = label_encoder.fit_transform(df_merged[stage_column])

    print(f"Data prepared. Found {len(df_merged)} samples and {len(label_encoder.classes_)} unique {stage_name} stages.")
    print(f"{stage_name} stages: {sorted(label_encoder.classes_)}")
    
    return df_merged, label_encoder

class TNMDataset(Dataset):
    # Dataset for TNM staging training
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

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    # Run one training epoch
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

def eval_model(model, data_loader, loss_fn, device, n_examples):
    # Evaluate model performance
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

def get_predictions(model, data_loader, device):
    # Generate predictions for final evaluation
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

def train_tnm_model(stage_type):
    # Train a single TNM staging model
    
    # Define paths and columns for each stage type
    stage_configs = {
        'T': {
            'path': PATH_T_STAGE,
            'column': 'ajcc_pathologic_t',
            'name': 'T-stage'
        },
        'N': {
            'path': PATH_N_STAGE,
            'column': 'ajcc_pathologic_n',
            'name': 'N-stage'
        },
        'M': {
            'path': PATH_M_STAGE,
            'column': 'ajcc_pathologic_m',
            'name': 'M-stage'
        }
    }
    
    if stage_type not in stage_configs:
        print(f"Error: Invalid stage type '{stage_type}'. Must be 'T', 'N', or 'M'.")
        return
    
    config = stage_configs[stage_type]
    
    print(f"\n{'='*60}")
    print(f"TRAINING {config['name'].upper()} MODEL")
    print(f"{'='*60}")
    
    # Load and prepare data
    df, label_encoder = prep_tnm_data(
        PATH_REPORTS, 
        config['path'], 
        config['column'], 
        config['name']
    )
    
    if df is None:
        print(f"Failed to prepare data for {config['name']} model.")
        return

    num_classes = len(label_encoder.classes_)
    print(f"\nTraining {config['name']} model with {num_classes} classes: {sorted(label_encoder.classes_)}")

    # Split data
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['label'])
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_STATE, stratify=df_test['label'])

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

    # Initialize tokenizer
    print(f"Loading tokenizer from local path: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    except OSError as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Create DataLoaders
    train_dataset = TNMDataset(
        texts=df_train.cleaned_text.to_numpy(),
        labels=df_train.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    val_dataset = TNMDataset(
        texts=df_val.cleaned_text.to_numpy(),
        labels=df_val.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

    test_dataset = TNMDataset(
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
        print(f"Error loading model: {e}")
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
        
        print(f'\nEpoch {epoch + 1}/{EPOCHS} ({progress_percent:.1f}% complete)')
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
            # Create model directory
            model_dir = f'models/{stage_type.lower()}_stage_model'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the best model using save_pretrained for better compatibility
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            best_accuracy = val_acc
            print(f'New best model saved to: {model_dir}')
    
    print(f"\n{config['name']} model training complete.")
    
    # Evaluation on Test Set
    print(f"\nEvaluating {config['name']} model on test set...")
    # The best model is already loaded, no need to reload...
    
    _, y_pred, _, y_test = get_predictions(model, test_data_loader, device)
    
    # Decode labels for reporting
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Print results
    print(f'\n{config["name"]} Test Accuracy: {accuracy_score(y_test_decoded, y_pred_decoded):.4f}')
    
    print(f"\n{config['name']} Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, target_names=label_encoder.classes_))
    
    print(f"\n{config['name']} Confusion Matrix:")
    conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(conf_matrix_df)
    
    # Save label encoder in the model directory
    model_dir = f'models/{stage_type.lower()}_stage_model'
    import pickle
    with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n{config['name']} model, tokenizer, and label encoder saved to: {model_dir}")
    print(f"Files saved:")
    print(f"  - {model_dir}/pytorch_model.bin")
    print(f"  - {model_dir}/config.json") 
    print(f"  - {model_dir}/tokenizer.json")
    print(f"  - {model_dir}/tokenizer_config.json")
    print(f"  - {model_dir}/label_encoder.pkl")

def main():
    # Main function to train TNM staging models
    parser = argparse.ArgumentParser(description='Train TNM staging models')
    parser.add_argument('--stage', type=str, choices=['T', 'N', 'M', 'all'], 
                       default='all', help='Which stage to train (T, N, M, or all)')
    
    args = parser.parse_args()
    
    if args.stage == 'all':
        # Train all three models
        for stage in ['T', 'N', 'M']:
            train_tnm_model(stage)
    else:
        # Train specific model
        train_tnm_model(args.stage)

if __name__ == '__main__':
    main()