import torch
import numpy as np
import os
import sklearn
import pandas as pd
import argparse
from args_classes import TrainerArgs
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, RobertaForTokenClassification,\
    Trainer, TrainingArguments, TrainerCallback, GPT2ForSequenceClassification, GPT2TokenizerFast,\
    BartForSequenceClassification, BartTokenizerFast
from tokenizers import AddedToken
import data_loader
import dataset_classes
from datasets import load_metric
from logger import Logger, log_from_log_history


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

    parser.add_argument(
        '--eval',
        help='whether or not to only evaluate',
        type=bool,
        default=False,
    )

    args = parser.parse_args()
    if args.json_file:
        args = TrainerArgs(args.json_file)
    return args


def get_model_and_tokenizer_from_args(args):
    model, tokenizer = None, None
    if 'roberta' in args.model_name:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
        if args.model_type == 'sequence_classification':
            model = RobertaForSequenceClassification.from_pretrained(args.model_name,
                                                                     hidden_dropout_prob=args.dropout,
                                                                     attention_probs_dropout_prob=args.dropout,
                                                                     num_labels=2)
        elif args.model_type == 'token_classification':
            tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken('<skip>', lstrip=True),
                                                                        AddedToken('<no_skip>', lstrip=True)]})
            model = RobertaForTokenClassification.from_pretrained(args.model_name,
                                                                  hidden_dropout_prob=args.dropout,
                                                                  attention_probs_dropout_prob=args.dropout,
                                                                  num_labels=2)
    elif 'gpt2' in args.model_name:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.model_name)
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        if args.model_type == 'sequence_classification':
            model = GPT2ForSequenceClassification.from_pretrained(args.model_name,
                                                                  resid_pdrop=args.dropout,
                                                                  embd_pdrop=args.dropout,
                                                                  attn_pdrop=args.dropout,
                                                                  pad_token_id=tokenizer.pad_token_id,
                                                                  num_labels=2)
    elif 'bart' in args.model_name:
        tokenizer = BartTokenizerFast.from_pretrained(args.model_name)
        if args.model_type == 'sequence_classification':
            model = BartForSequenceClassification.from_pretrained(args.model_name,
                                                                  attention_dropout=args.dropout,
                                                                  activation_dropout=args.dropout,
                                                                  num_labels=2)
    if args.eval:
        model = model.from_pretrained(args.model_dir)
    if model and tokenizer:
        return model, tokenizer
    raise Exception('no such model: name "{}", type "{}"'.format(args.model_name, args.model_type))


class eval_token_prediction(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, eval_pred, *args, **kwargs):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        eval_data = pd.read_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')
        eval_data[self.args.model_name + '_' + self.args.model_type] = pd.Series(predictions)
        eval_data.to_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')

        return accuracy_metric.compute(predictions=predictions[labels != -100], references=labels[labels != -100])


class eval_sequence_prediction(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, eval_pred, *args, **kwargs):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        eval_data = pd.read_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')
        eval_data[self.args.model_name + '_' + self.args.model_type] = pd.Series(predictions)
        eval_data.to_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')

        return accuracy_metric.compute(predictions=predictions, references=labels)


class eval_bart_sequence_prediction(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, eval_pred, *args, **kwargs):
        logits, labels = eval_pred
        predictions = np.argmax(logits[0], axis=-1)

        eval_data = pd.read_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')
        eval_data[self.args.model_name + '_' + self.args.model_type] = pd.Series(predictions)
        eval_data.to_csv(self.args.data_dir + os.path.sep + 'evaluated.tsv', sep='\t')

        return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_token_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions[labels != -100], references=labels[labels != -100])


def compute_sequence_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_bart_sequence_accuracy(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


def compute_metrics(args):
    if not args.eval:
        if 'bart' in args.model_name:
            return compute_bart_sequence_accuracy
        elif args.model_type == 'token_classification':
            return compute_token_accuracy
        elif args.model_type == 'sequence_classification':
            return compute_sequence_accuracy
    else:
        if 'bart' in args.model_name:
            return eval_bart_sequence_prediction(args)
        elif args.model_type == 'token_classification':
            return eval_token_prediction(args)
        elif args.model_type == 'sequence_classification':
            return eval_sequence_prediction(args)

def encode_targets_for_token_classification(batch_target, tokenizer):
    encoded_targets_list = tokenizer(batch_target, return_attention_mask=False, truncation=True, padding='max_length')['input_ids']
    encoded_targets = np.array(encoded_targets_list)
    encoded_targets[np.logical_and(encoded_targets != (len(tokenizer) - 2), encoded_targets != (len(tokenizer) - 1))] = -100
    encoded_targets[encoded_targets == (len(tokenizer) - 2)] = 0
    encoded_targets[encoded_targets == (len(tokenizer) - 1)] = 1

    return encoded_targets.tolist()


def encode_targets(batch_target, tokenizer, args):
    if args.model_type == 'sequence_classification':
        return batch_target
    elif args.model_type == 'token_classification':
        return encode_targets_for_token_classification(batch_target, tokenizer)


def load_and_tokenize_dataset(args, tokenizer):
    if args.eval:
        data = [pd.read_csv(args.data_dir + os.path.sep + 'dev.tsv', sep='\t')]
        splits_ratio = [1]
    else:
        data = data_loader.read_data_from_csv(args.data_dir)
        # the ratio of train/dev/test sets where 1 is the full size of each the set
        splits_ratio = args.data_split_ratio

    tokenized_data = []
    for split, ratio in zip(data, splits_ratio):
        if ratio == 0:
            continue
        text = split['text'].tolist()[:int(len(split) * ratio)]
        target = split['target'].tolist()[:int(len(split) * ratio)]

        tokenized_data.append(
            {
                'encoded_text': tokenizer(text, max_length=512, return_attention_mask=True,
                                          truncation=True, padding='max_length'),
                'encoded_target': encode_targets(target, tokenizer, args)
            }
        )

    dataset = []
    for split in tokenized_data:
        dataset.append(dataset_classes.TextDataset(split['encoded_text'], split['encoded_target']))

    return dataset


def set_trainer(args):
    model, tokenizer = get_model_and_tokenizer_from_args(args)
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_and_tokenize_dataset(args, tokenizer)
    if len(dataset) > 1:
        train = dataset[0]
        dev = dataset[1]
    else:
        train = None
        dev = dataset[0]

    class StopEachEpochCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, logs=None, **kwargs):
            control.should_training_stop = True

    class EvaluateAndSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, logs=None, **kwargs):
            if state.global_step % args.eval_steps == 0:
                control.should_evaluate = True
                control.should_save = True

    class LoggingCallback(TrainerCallback):
        def on_step_end(self, callback_args, state, control, logs=None, **kwargs):
            if state.global_step % args.logging_steps == 0:
                control.should_log = True

    training_args = TrainingArguments(
        output_dir=args.model_dir,
        max_steps=args.eval_steps * args.num_evals,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=128 // args.batch_size,
        eval_accumulation_steps=args.batch_size * 3,
        fp16=True,
        fp16_full_eval=True,
        fp16_backend='apex',
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy='no',
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_total_limit=1,
        seed=42,
        load_best_model_at_end=True,
        metric_for_best_model='eval_accuracy',
        greater_is_better=True
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=dev,
        callbacks=[EvaluateAndSaveCallback(), LoggingCallback()],
        compute_metrics=compute_metrics(args)
    )

    return trainer


def train_and_eval(trainer, args):
    """
    my training loop. currently not used due to switch to huggingface's loop. keeping it here for future reference
    """
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

    if args.eval:
        trainer.evaluate()
    else:
        try:
            trainer.train(resume_from_checkpoint=True)
        except ValueError as e:
            if 'No valid checkpoint' in e.args[0]:
                trainer.train()
            else:
                raise e

        trainer.save_model(args.model_dir)
        trainer.save_state()
        log_from_log_history(trainer.state.log_history, args.model_dir)
