import torch
import numpy as np
import pandas as pd
import argparse
from args_classes import TrainerArgs
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments, TrainerCallback
import data_loader
import dataset_classes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )

    parser.add_argument(
        "--data_dir",
        help="path to dataset directory",
        type=str,
        default="C:\\my_documents\\datasets\\AMNLPFinal\\miss_last_paragraph_3_paragraphs"
    )

    parser.add_argument(
        "--model_dir",
        help="path to model directory",
        type=str,
        default="C:\\my_documents\\models\\roberta_miss_last_paragraph_3_paragraphs"
    )

    parser.add_argument(
        "--model_name",
        help="name of the model to load from the web",
        type=str,
        default="roberta-base"
    )

    parser.add_argument(
        '--start_epoch',
        help='continue or start training from this epoch',
        type=int,
        default=0,
    )

    parser.add_argument(
        '--end_epoch',
        help='end training on this epoch',
        type=int,
        default=50,
    )

    args = parser.parse_args()
    if args.json_file:
        args = TrainerArgs(args.json_file)
    return args


def set_trainer(args):

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base')

    data = data_loader.read_data_from_csv(args.data_dir)
    # dev set is 1/10 the size of train set and test set
    splits_ratio = [1, 0.1, 1]

    tokenized_data = []
    for split, ratio in zip(data, splits_ratio):
        tokenized_data.append(
            {
                'encoded_text': tokenizer(split['text'].tolist()[:int(len(split) * ratio)], truncation=True, padding=True),
                'targets': split['target'].tolist()[:int(len(split) * ratio)]
            }
        )

    dataset = []
    for split in tokenized_data:
        dataset.append(dataset_classes.TextDataset(split['encoded_text'], split['targets']))

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    class StopEachEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            control.should_training_stop = True


    training_args = TrainingArguments(
        output_dir=args.model_dir,          # output directory
        num_train_epochs=50,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        gradient_accumulation_steps=32,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_steps=6,
        save_strategy="no",
        seed=42,
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        # data_collator=data_collator,
        train_dataset=dataset[0],         # training dataset
        eval_dataset=dataset[1],           # evaluation dataset
        callbacks=[StopEachEpochCallback()]
    )

    return trainer


def train_and_eval(trainer, args):
    start_epoch, end_epoch, model_dir = args.start_epoch, args.end_epoch, args.model_dir
    eval_loss = None
    for epoch in range(start_epoch, end_epoch):
        if epoch == 0:
            trainer.train()
        else:
            trainer.train(model_dir)

        new_eval_loss = trainer.evaluate()['eval_loss']
        if eval_loss is None or new_eval_loss < eval_loss:
            eval_loss = new_eval_loss
            trainer.save_model(model_dir + '\\best_model')
            trainer.save_state()
            print("saved epoch: " + str(epoch + 1))
        trainer.save_model(model_dir)
        trainer.save_state()
        print("epoch: " + str(epoch + 1))
        print("eval loss: " + str(new_eval_loss))


if __name__ == "__main__":
    args = parse_args()
    trainer = set_trainer(args)
    train_and_eval(trainer, args)
    # trainer.train()
