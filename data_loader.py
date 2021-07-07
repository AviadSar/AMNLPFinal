import torch
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from datasets import load_dataset
import argparse
from args_classes import DataLoaderArgs

from manipulation_funcs import get_manipulation_func_from_args
from data_cleaning_funcs import get_data_cleaning_funcs_from_args


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_file",
        help="path to a json file to load arguments from",
        type=str,
        default=None
    )

    parser.add_argument(
        "--wiki_dir",
        help="path to original wikipedia dataset directory",
        type=str,
        default="C:\\my_documents\\datasets\\AMNLPFinal\\wikipedia"
    )

    parser.add_argument(
        "--filtered_data_dir",
        help="path to the newly created dataset directory",
        type=str,
        default="C:\\my_documents\\datasets\\AMNLPFinal\\wiki_gt_6_sentences_100k"
    )

    parser.add_argument(
        "--final_data_dir",
        help="path to the newly created dataset directory",
        type=str,
        default="C:\\my_documents\\datasets\\AMNLPFinal\\miss_last_sentence_5_sentences_"
    )

    parser.add_argument(
        '--n_data_samples',
        help='continue or start training from this epoch',
        type=int,
        default=30000,
    )

    parser.add_argument(
        '--manipulation_func',
        help='the function used to adjust the newly created dataset',
        type=str,
        default=None
    )

    parser.add_argument(
        '--clean_and_filter_funcs',
        help='the functions used to clean and filter the original wikipedia dataset',
        type=str,
        nargs='+',
        default=None
    )

    args = parser.parse_args()
    if args.json_file:
        args = DataLoaderArgs(args.json_file)
    return args


def write_data_as_csv(data, data_dir):
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        print("creating dataset directory " + data_dir)

    train, dev, test = data
    train.to_csv(data_dir + os.path.sep + 'train.tsv', sep='\t')
    dev.to_csv(data_dir + os.path.sep + 'dev.tsv', sep='\t')
    test.to_csv(data_dir + os.path.sep + 'test.tsv', sep='\t')


def read_data_from_csv(data_dir):
    train = pd.read_csv(data_dir + os.path.sep + 'train.tsv', sep='\t')
    dev = pd.read_csv(data_dir + os.path.sep + 'dev.tsv', sep='\t')
    test = pd.read_csv(data_dir + os.path.sep + 'test.tsv', sep='\t')

    return train, dev, test


def data_histograms(data):
    for split in data:

        max_len_lines = 0
        for index, row in split.iterrows():
            len_lines = len(row['text'].split('\n\n'))
            if len_lines > max_len_lines:
                max_len_lines = len_lines

        histogram = [0] * (max_len_lines + 1)
        for index, row in split.iterrows():
            len_lines = len(row['text'].split('\n\n'))
            histogram[len_lines] += 1

        print(max(histogram))
        print(histogram)

        plt.hist(np.array(histogram), bins=max_len_lines)
        plt.show()


def load_data(args):
    wiki = load_dataset('wikipedia', "20200501.en", cache_dir=args.wiki_dir)
    n_data_samples = args.n_data_samples

    selected_data_indices = np.random.choice(range(len(wiki['train'])), n_data_samples, replace=False)

    data = wiki['train'].select(selected_data_indices)

    train = pd.DataFrame({'title': data[:n_data_samples // 3]['title'], 'text': data[:n_data_samples // 3]['text']})
    dev = pd.DataFrame({'title': data[n_data_samples // 3:(n_data_samples // 3) * 2]['title'], 'text': data[n_data_samples // 3:(n_data_samples // 3) * 2]['text']})
    test = pd.DataFrame({'title': data[(n_data_samples // 3) * 2:]['title'], 'text': data[(n_data_samples // 3) * 2:]['text']})

    datasets = [train, dev, test]
    for func in args.clean_and_filter_funcs:
        for index, dataset in enumerate(datasets):
            datasets[index] = func(dataset)

    return datasets


def print_samples(split):
    samples_indices = np.random.randint(0, len(split['text']), 10)
    for i in samples_indices:
        print(i)
        print("target")
        print(split['target'][i])
        print("text")
        print(split['text'][i])
        print("original_text")
        print(split['original_text'][i])


def make_dataset(data, args):
    splits = []

    for split in data:
        texts_to_manipulate_indices = np.random.choice(range(len(split)), len(split) // 2, replace=False)
        texts_to_manipulate_bools = np.zeros(len(split))
        texts_to_manipulate_bools[texts_to_manipulate_indices] = 1

        split['original_text'] = split['text']
        split['target'] = split['text']
        split['should_manipulate'] = texts_to_manipulate_bools

        split = split.apply(args.manipulation_func, axis=1)
        print_samples(split)
        splits.append(split)

    write_data_as_csv(splits, args.final_data_dir)


if __name__ == '__main__':
    args = parse_args()
    args.manipulation_func = get_manipulation_func_from_args(args)
    args.clean_and_filter_funcs = get_data_cleaning_funcs_from_args(args)

    if args.clean_and_filter_funcs:
        np.random.seed(42)
        train, dev, test = load_data(args)
        write_data_as_csv((train, dev, test), args.filtered_data_dir)

    if args.manipulation_func:
        np.random.seed(42)
        train, dev, test = read_data_from_csv(args.filtered_data_dir)
        make_dataset((train, dev, test), args)
