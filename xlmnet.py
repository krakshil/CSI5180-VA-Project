import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering
from dataset import SquadDataset

# Initialize tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
model = XLMRobertaForQuestionAnswering.from_pretrained('xlm-roberta-large')

# Set hyperparameters
max_length = 512
batch_size = 4
num_epochs = 3
learning_rate = 5e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
squad_dataset = SquadDataset(tokenizer, max_length, batch_size)
train_dataloader = squad_dataset.train_dataloader
eval_dataloader = squad_dataset.eval_dataloader

# Move model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Train model
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 500 == 0:
            print(f'Train Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}')

    # Evaluate model
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            eval_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Eval Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(eval_dataloader)}, Loss: {loss.item()}')

    print(f'Train Epoch: {epoch+1}/{num_epochs}, Average Loss: {train_loss/len(train_dataloader)}, Eval Loss: {eval_loss/len(eval_dataloader)}')
