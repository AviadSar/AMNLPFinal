import torch
import numpy as np
import os
import sklearn
import pandas as pd
import argparse
from args_classes import TrainerArgs
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaForTokenClassification,\
    Trainer, TrainingArguments, TrainerCallback
from tokenizers import AddedToken
import data_loader
import dataset_classes
from datasets import load_metric
from logger import Logger

accuracy_metric = load_metric("accuracy")


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
        '--model_type',
        help='the type of model (head), e.g., sequence classification, token classification, etc.',
        type=str,
        default="classification",
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

    parser.add_argument(
        '--batch_size',
        help='number of samples in each batch of training/evaluating',
        type=int,
        default=4,
    )

    args = parser.parse_args()
    if args.json_file:
        args = TrainerArgs(args.json_file)
    return args


def get_model_from_string(args):
    if 'roberta' in args.model_name:
        if args.model_type == 'sequence_classification':
            return RobertaForSequenceClassification.from_pretrained(args.model_name)
        elif args.model_type == 'token_classification':
            return RobertaForTokenClassification.from_pretrained(args.model_name)
    raise Exception('no such model: name "{}", type "{}"'.format(args.model_name, args.model_type))


def encode_targets_for_token_classification(split, ratio, tokenizer):
    encoded_targets = tokenizer(split['target'].tolist()[:int(len(split) * ratio)], truncation=True, padding=True)['input_ids']
    for target in encoded_targets:
        for index, val in enumerate(target):
            # if the token is not one of the two special tokens <skip>, <no_skip>
            if val not in [len(tokenizer) - 2, len(tokenizer) - 1]:
                # huggingface's way of telling the trainer to ignore this token for loss calculations
                target[index] = -100
            elif val == len(tokenizer) - 2:
                target[index] = 0
            elif val == len(tokenizer) - 1:
                target[index] = 1
    return encoded_targets

def encode_targets(split, ratio, tokenizer, args):
    if args.model_type == 'sequence_classification':
        return split['target'].tolist()[:int(len(split) * ratio)]
    elif args.model_type == 'token_classification':
        return encode_targets_for_token_classification(split, ratio, tokenizer)


def compute_token_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions[labels != -100], references=labels[labels != -100])


def compute_sequence_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_metrics(args):
    if args.model_type == 'token_classification':
        return compute_token_accuracy
    elif args.model_type == 'sequence_classification':
        return compute_sequence_accuracy


def set_trainer(args):
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken('<skip>', lstrip=True), AddedToken('<no_skip>', lstrip=True)]})
    model = get_model_from_string(args)
    model.resize_token_embeddings(len(tokenizer))

    data = data_loader.read_data_from_csv(args.data_dir)
    # the ratio of train/dev/test sets where 1 is the full size of each the set
    splits_ratio = [0.01, 0.01, 1]

    tokenized_data = []
    for split, ratio in zip(data, splits_ratio):
        tokenized_data.append(
            {
                'encoded_text': tokenizer(split['text'].tolist()[:int(len(split) * ratio)], truncation=True, padding=True),
                'encoded_target': encode_targets(split, ratio, tokenizer, args)
            }
        )

    dataset = []
    for split in tokenized_data:
        dataset.append(dataset_classes.TextDataset(split['encoded_text'], split['encoded_target']))

    class StopEachEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            control.should_training_stop = True

    training_args = TrainingArguments(
        output_dir=args.model_dir,          # output directory
        num_train_epochs=args.end_epoch,              # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        gradient_accumulation_steps=128 // args.batch_size,
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_steps=10,
        save_strategy="no",
        seed=42,
    )


    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=dataset[0],         # training dataset
        eval_dataset=dataset[1],           # evaluation dataset
        callbacks=[StopEachEpochCallback()],
        compute_metrics=compute_metrics(args)
    )

    return trainer


def train_and_eval(trainer, args):
    start_epoch, end_epoch, model_dir = args.start_epoch, args.end_epoch, args.model_dir
    logger = Logger(args, start_epoch)
    best_eval_accuracy = logger.best_eval_accuracy
    for epoch in range(start_epoch, end_epoch):
        if epoch == 0:
            trainer.train()
        else:
            trainer.train(model_dir)

        eval = trainer.evaluate()
        eval_accuracy = eval['eval_accuracy']
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            trainer.save_model(model_dir + os.path.sep + 'best_model')
            trainer.save_state()
            print("saved epoch: " + str(epoch + 1))
        trainer.save_model(model_dir)
        trainer.save_state()
        logger.update(trainer.state.log_history[-2]['train_loss'], eval['eval_loss'], eval_accuracy)
        print("epoch: " + str(epoch + 1))
        print("eval loss: " + str(eval['eval_loss']))
        print("eval accuracy: " + str(eval_accuracy))


if __name__ == "__main__":
    args = parse_args()
    trainer = set_trainer(args)
    train_and_eval(trainer, args)
