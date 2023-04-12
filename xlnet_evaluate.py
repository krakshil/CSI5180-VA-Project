import torch
from torchmetrics import Accuracy, F1Score
from transformers import XLNetForQuestionAnsweringSimple, AutoTokenizer
from dataset import SquadDataset

# Define save and load path
xlnet_dir = "model_weights/XLNet/"

load_path = xlnet_dir + "06-04-2023-01-51" # add date_time as name

# Define hyperparameters
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased", padding_side="right")
model = XLNetForQuestionAnsweringSimple.from_pretrained(load_path).to(device)

# Load dataset and dataloaders
squad_dataset = SquadDataset(tokenizer, max_length=384, batch_size=batch_size, eval=True)
eval_dataloader = squad_dataset.eval_dataloader

# Define Metrics functions
acc_fn_1 = Accuracy("multiclass", num_classes=384, top_k=1).to(device)
f1_fn_1 = F1Score("multiclass", num_classes=384, top_k=1).to(device)

acc_fn_3 = Accuracy("multiclass", num_classes=384, top_k=3).to(device)
f1_fn_3 = F1Score("multiclass", num_classes=384, top_k=3).to(device)

acc_fn_5 = Accuracy("multiclass", num_classes=384, top_k=5).to(device)
f1_fn_5 = F1Score("multiclass", num_classes=384, top_k=5).to(device)

# Evaluate
model.eval()

eval_loss, top1_acc, top3_acc, top5_acc, top1_f1, top3_f1, top5_f1 = 0, 0, 0, 0, 0, 0, 0

with torch.no_grad():
    for batch_idx, batch in enumerate(eval_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        start_logits, end_logits, loss = outputs.start_logits, outputs.end_logits, outputs.loss
        
        start_acc_1, start_acc_3, start_acc_5 = acc_fn_1(start_logits, start_positions), acc_fn_3(start_logits, start_positions), acc_fn_5(start_logits, start_positions) 
        end_acc_1, end_acc_3, end_acc_5 = acc_fn_1(end_logits, end_positions), acc_fn_3(end_logits, end_positions), acc_fn_5(end_logits, end_positions)
        
        start_f1_1, start_f1_3, start_f1_5 = f1_fn_1(start_logits, start_positions), f1_fn_3(start_logits, start_positions), f1_fn_5(start_logits, start_positions) 
        end_f1_1, end_f1_3, end_f1_5 = f1_fn_1(end_logits, end_positions), f1_fn_3(end_logits, end_positions), f1_fn_5(end_logits, end_positions)
        
        eval_loss += loss.item() * input_ids.size(0)
        
        top1_acc += ((start_acc_1 * input_ids.size(0)) + (end_acc_1 * input_ids.size(0)))/2
        top3_acc += ((start_acc_3 * input_ids.size(0)) + (end_acc_3 * input_ids.size(0)))/2
        top5_acc += ((start_acc_5 * input_ids.size(0)) + (end_acc_5 * input_ids.size(0)))/2
        
        top1_f1 += ((start_f1_1 * input_ids.size(0)) + (end_f1_1 * input_ids.size(0)))/2
        top3_f1 += ((start_f1_3 * input_ids.size(0)) + (end_f1_3 * input_ids.size(0)))/2
        top5_f1 += ((start_f1_5 * input_ids.size(0)) + (end_f1_5 * input_ids.size(0)))/2
        
        if (batch_idx) % 100 == 0:
            print(f"Batch: {batch_idx}/{len(eval_dataloader)}, Loss: {loss.item():.4f}", end="\r")

eval_loss /= len(squad_dataset.eval_dataset)
top1_acc /= len(squad_dataset.eval_dataset)
top3_acc /= len(squad_dataset.eval_dataset)
top5_acc /= len(squad_dataset.eval_dataset)
top1_f1 /= len(squad_dataset.eval_dataset)
top3_f1 /= len(squad_dataset.eval_dataset)
top5_f1 /= len(squad_dataset.eval_dataset)

print(f"Eval Loss: {eval_loss:.8f}\nTop-1 Accuracy: {top1_acc:.8f}\nTop-3 Accuracy: {top3_acc:.8f}\nTop-5 Accuracy: {top5_acc:.8f}\nTop-1 F1-Score: {top1_f1:.8f}\nTop-3 F1-Score: {top3_f1:.8f}\nTop-5 F1-Score: {top5_f1:.8f}")