import torch
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments
import data_loader
import dataset_classes

data_dir = "C:\\my_documents\\datasets"
model_dir = "C:\\my_documents\\models"

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base')

data = data_loader.read_data_from_csv(data_dir + '\\AMNLPFinal\\missing_last_sentence')

tokenized_data = []
for split in data:
    tokenized_data.append({'encoded_text': tokenizer(split['text'].tolist(), truncation=True, padding=True), 'targets': split['target'].tolist()})

dataset = []
for split in tokenized_data:
    dataset.append(dataset_classes.TextDataset(split['encoded_text'], split['targets']))

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir=model_dir + "\\roberta_missing_last_sentence",          # output directory
    num_train_epochs=25,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    gradient_accumulation_steps=8,
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_steps=250,
    save_total_limit=2,
    seed=42
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    # data_collator=data_collator,
    train_dataset=dataset[0],         # training dataset
    eval_dataset=dataset[1]             # evaluation dataset
)

trainer.train()
trainer.save_model(model_dir + "\\roberta_missing_last_sentence")
