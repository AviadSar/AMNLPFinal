import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datasets import load_dataset
import re

data_dir = "C:\\Users\\aavia\\OneDrive\\Documents\\datasets"


def clip_to_10k(dataset):
    if dataset.shape[0] < 10000:
        raise Exception('not enough samples')
    cleaned_dataset = dataset[:][:10000]
    return cleaned_dataset


def num_lines(text):
    return len(re.split(r'\.[^.]', text))


def num_paragraphs(text):
    return len(text.split('\n'))


def longer_then_3_sentences(dataset):
    cleaned_dataset = dataset[dataset['text'].map(num_lines) > 3]
    return cleaned_dataset


def longer_then_4_paragraphs(dataset):
    cleaned_dataset = dataset[dataset['text'].map(num_paragraphs) > 4]
    return cleaned_dataset


def get_longest_sequence(row):
    row = row.replace('\n\n', '\n')
    sequences = re.split(r'\n.*[^.]\n', row)
    return max(sequences, key=lambda sequence: len(sequence))


def longest_sequence(dataset):
    dataset['text'] = dataset['text'].apply(get_longest_sequence)
    return dataset


def omit_references_single_row(row):
    references_index = row.rfind('References')
    if references_index == -1:
        return row
    return row[:row.rfind('References')]


def omit_references(dataset):
    dataset['text'] = dataset['text'].apply(omit_references_single_row)
    return dataset


def load_data(n_data_samples, clean_and_filter_funcs):
    wiki = load_dataset('wikipedia', "20200501.en", cache_dir=data_dir)

    selected_data_indices = np.random.choice(range(len(wiki['train'])), n_data_samples, replace=False)

    data = wiki['train'].select(selected_data_indices)

    train = pd.DataFrame({'title': data[:n_data_samples // 3]['title'], 'text': data[:n_data_samples // 3]['text']})
    dev = pd.DataFrame({'title': data[n_data_samples // 3:(n_data_samples // 3) * 2]['title'], 'text': data[n_data_samples // 3:(n_data_samples // 3) * 2]['text']})
    test = pd.DataFrame({'title': data[(n_data_samples // 3) * 2:]['title'], 'text': data[(n_data_samples // 3) * 2:]['text']})

    datasets = [train, dev, test]
    for func in clean_and_filter_funcs:
        for index, dataset in enumerate(datasets):
            datasets[index] = func(dataset)

    return datasets


def write_data_as_csv(data, dir):
    train, dev, test = data
    train.to_csv(dir + '\\train.tsv', sep='\t')
    dev.to_csv(dir + '\\dev.tsv', sep='\t')
    test.to_csv(dir + '\\test.tsv', sep='\t')


def read_data_from_csv(dir):
    train = pd.read_csv(dir + '\\train.tsv', sep='\t')
    dev = pd.read_csv(dir + '\\dev.tsv', sep='\t')
    test = pd.read_csv(dir + '\\test.tsv', sep='\t')

    return train, dev, test


def miss_last_paragraph(series):
    text = series['original_text']
    paragraphs = text.split('\n')

    deleted_paragraph_index = -1
    while deleted_paragraph_index == -1:
        deleted_paragraph_index = np.random.randint(1, len(paragraphs) - 1, 1)[0]
        if not paragraphs[deleted_paragraph_index]:
            deleted_paragraph_index = -1

    if series['should_manipulate']:
        new_paragraphs = paragraphs[:deleted_paragraph_index] + paragraphs[deleted_paragraph_index + 1]
        series['text'] = '\n'.join(new_paragraphs)
        series['target'] = 1
    else:
        series['text'] = '\n'.join(paragraphs[:deleted_paragraph_index + 2])
        series['target'] = 0

    return series


def miss_random_paragraph(series):
    if series['should_manipulate']:
        text = series['original_text']
        paragraphs = text.split('\n')

        deleted_paragraph_index = -1
        while deleted_paragraph_index == -1:
            deleted_paragraph_index = np.random.randint(1, len(paragraphs) - 1, 1)[0]
            if not paragraphs[deleted_paragraph_index]:
                deleted_paragraph_index = -1
        new_paragraphs = paragraphs[:deleted_paragraph_index] + paragraphs[deleted_paragraph_index + 1:]
        series['text'] = '\n'.join(new_paragraphs)

        new_paragraphs[deleted_paragraph_index] = '<skip>' + new_paragraphs[deleted_paragraph_index]
        series['target'] = '\n'.join(new_paragraphs)

    return series


def miss_last_sentence(series):
    text = series['original_text']
    lines = text.split('.')

    deleted_sentence_index = -1
    while deleted_sentence_index == -1:
        deleted_sentence_index = np.random.randint(1, len(lines) - 2, 1)[0]
        if not lines[deleted_sentence_index]:
            deleted_sentence_index = -1

    if series['should_manipulate']:
        new_lines = lines[:deleted_sentence_index] + lines[deleted_sentence_index + 1]
        series['text'] = '.'.join(new_lines)
        series['target'] = 1
    else:
        series['text'] = '.'.join(lines[:deleted_sentence_index + 2])
        series['target'] = 0

    return series


def miss_random_sentence(series):
    if series['should_manipulate']:
        text = series['original_text']
        lines = text.split('.')

        deleted_sentence_index = -1
        while deleted_sentence_index == -1:
            deleted_sentence_index = np.random.randint(1, len(lines) - 2, 1)[0]
            if not lines[deleted_sentence_index]:
                deleted_sentence_index = -1
        new_lines = lines[:deleted_sentence_index] + lines[deleted_sentence_index + 1:]
        series['text'] = '.'.join(new_lines)

        new_lines[deleted_sentence_index] = '<skip>' + new_lines[deleted_sentence_index]
        series['target'] = '.'.join(new_lines)

    return series


def make_dataset(data, manipulation_func, dataset_dir):
    splits = []

    for split in data:
        texts_to_manipulate_indices = np.random.choice(range(len(split)), len(split) // 2, replace=False)
        texts_to_manipulate_bools = np.zeros(len(split))
        texts_to_manipulate_bools[texts_to_manipulate_indices] = 1

        split['original_text'] = split['text']
        split['target'] = split['text']
        split['should_manipulate'] = texts_to_manipulate_bools

        split = split.apply(manipulation_func, axis=1)
        splits.append(split)

    write_data_as_csv(splits, dataset_dir)


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


# first time loading base data
# np.random.seed(42)
# train, dev, test = load_data(60000, [longest_sequence, longer_then_3_sentences, clip_to_10k])
# write_data_as_csv((train, dev, test), data_dir + '\\AMNLPFinal\\wiki_gt_3_sentences')

# make a dataset of texts that are missing some random sentence. the target is the position of the missing sentence
np.random.seed(42)
train, dev, test = read_data_from_csv(data_dir + '\\AMNLPFinal\\wiki_gt_3_sentences')
make_dataset((train, dev, test), miss_random_sentence, data_dir + '\\AMNLPFinal\\missing_sentence')

# make a dataset of texts that are missing the sentence before last (or not missing). the target is 1 if the sentence is missing, 0 if not
np.random.seed(42)
train, dev, test = read_data_from_csv(data_dir + '\\AMNLPFinal\\wiki_gt_3_sentences')
make_dataset((train, dev, test), miss_last_sentence, data_dir + '\\AMNLPFinal\\missing_last_sentence')

# make a dataset of texts that are missing some random paragraph. the target is the position of the missing paragraph
np.random.seed(42)
train, dev, test = read_data_from_csv(data_dir + '\\AMNLPFinal\\wiki_gt_4_paragraphs')
make_dataset((train, dev, test), miss_random_paragraph, data_dir + '\\AMNLPFinal\\missing_paragraph')

# make a dataset of texts that are missing the paragraph before last (or not missing). the target is 1 if the paragraph is missing, 0 if not
np.random.seed(42)
train, dev, test = read_data_from_csv(data_dir + '\\AMNLPFinal\\wiki_gt_4_paragraphs')
make_dataset((train, dev, test), miss_last_paragraph, data_dir + '\\AMNLPFinal\\missing_last_paragraph')