import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import SquadDataset
from models.bidaf import BiDAF

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
squad_dataset = SquadDataset(tokenizer=tokenizer, max_length=512, batch_size=8)

# Initialize BiDAF model
model = BiDAF()
model.to(device)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# Define loss function
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss

# Training loop
for epoch in range(5):
    model.train()
    train_loss = 0
    
    for batch in squad_dataset.train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        
        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)
        loss = loss_fn(start_logits, end_logits, start_positions, end_positions)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(squad_dataset.train_dataloader)
    print(f"Epoch {epoch+1}/{5}: Train Loss: {train_loss:.4f}")
    
    model.eval()
    eval_loss = 0
    
    with torch.no_grad():
        for batch in squad_dataset.eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)
            loss = loss_fn(start_logits, end_logits, start_positions, end_positions)

            eval_loss += loss.item()
        
        eval_loss /= len(squad_dataset.eval_dataloader)
        print(f"Epoch {epoch+1}/{5}: Eval Loss: {eval_loss:.4f}")
        
    scheduler.step()