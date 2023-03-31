import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


class SquadDataset:
    def __init__(self, tokenizer, max_length, batch_size):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        
        self.train_dataset = load_dataset("squad")["train"]
        self.preprocessed_train_dataset = self.train_dataset.map(self.preprocess, batched=True, remove_columns=self.train_dataset.column_names).with_format("torch")
        
        self.eval_dataset = load_dataset("squad")["validation"]
        self.preprocessed_eval_dataset = self.eval_dataset.map(self.preprocess, batched=True, remove_columns=self.eval_dataset.column_names).with_format("torch")
    
        self.train_dataloader = DataLoader(self.preprocessed_train_dataset, batch_size=self.batch_size)
        self.eval_dataloader = DataLoader(self.preprocessed_eval_dataset, batch_size=self.batch_size)


    def preprocess(self, examples):
        
        questions = [q.strip() for q in examples["question"]]
        
        tokenized_examples = self.tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        offset_mapping = tokenized_examples.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        tokenized_examples["start_positions"] = start_positions
        tokenized_examples["end_positions"] = end_positions
        
        for k, v in tokenized_examples.items():
            if k == "attention_mask":
                tokenized_examples[k] = torch.FloatTensor(v)
            else:
                tokenized_examples[k] = torch.LongTensor(v)

        return tokenized_examples