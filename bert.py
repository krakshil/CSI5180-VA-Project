import os
import torch
import torch.nn as nn
from transformers import BertModel, BertForQuestionAnswering, AutoTokenizer
from transformers import get_scheduler
from dataset import SquadDataset

# Define save and load path
bert_dir = "model_weights/BERT/"

if not os.path.exists(bert_dir):
    os.makedirs(bert_dir)

save_path = bert_dir + "01-04-2023-10-57" # add date_time as name
load_model = False # change to True if weights saved locally and resuming training from between.

if load_model:
    load_path = bert_dir + "" # add date_time as name
    if not os.path.exists(load_path):
        load_model=False
        load_path = "bert-case-uncased"
else:
    load_path = "bert-base-uncased"

# Define hyperparameters
batch_size = 8
learning_rate = 2e-5
weight_decay = 0.05
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForQuestionAnswering.from_pretrained(load_path).to(device)

# Load dataset and dataloaders
squad_dataset = SquadDataset(tokenizer, max_length=384, batch_size=batch_size)
train_dataloader = squad_dataset.train_dataloader
eval_dataloader = squad_dataset.eval_dataloader

# Define loss function and optimizer
num_training_steps = num_epochs * len(train_dataloader)
num_warmup_steps = len(train_dataloader)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = get_scheduler(name="exponential", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    
    # Train
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        train_loss += loss.item() * input_ids.size(0)

        if batch_idx % 500 == 0:
            print(f"Train Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}", end="\r")
    
    # model.save_pretrained(save_path)
    model.save(save_path)

    # Evaluate
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs.loss

            eval_loss += loss.item() * input_ids.size(0)
            if (batch_idx) % 100 == 0:
                print(f"Eval Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(eval_dataloader)}, Loss: {loss.item():.4f}", end="\r")


    train_loss /= len(squad_dataset.train_dataset)
    eval_loss /= len(squad_dataset.eval_dataset)
    
    print(f"Train Loss: {train_loss:.8f}, Eval Loss: {eval_loss:.8f}")