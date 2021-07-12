import re
import nltk
from nltk import tokenize
nltk.download('punkt')


class clip(object):
    def __init__(self, clip_size):
        self.clip_size = clip_size

    def __call__(self, dataset, *args, **kwargs):
        if dataset.shape[0] < self.clip_size:
            raise Exception('not enough samples')
        cleaned_dataset = dataset[:][:self.clip_size]
        return cleaned_dataset


def num_lines(text):
    return len(tokenize.sent_tokenize(text))


def num_paragraphs(text):
    return len(text.split('\n'))


class longer_than_n_sentences(object):
    def __init__(self, sentences_limit):
        self.sentences_limit = sentences_limit

    def __call__(self, dataset, *args, **kwargs):
        cleaned_dataset = dataset[dataset['text'].map(num_lines) > self.sentences_limit]
        return cleaned_dataset


class longer_than_n_paragraphs(object):
    def __init__(self, paragraph_limit):
        self.paragraph_limit = paragraph_limit

    def __call__(self, dataset, *args, **kwargs):
        cleaned_dataset = dataset[dataset['text'].map(num_paragraphs) > self.paragraph_limit]
        return cleaned_dataset


class truncate_to_n_sentences_single_row(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, row, *args, **kwargs):
        lines = tokenize.sent_tokenize(row)
        return ' '.join(lines[:self.n])


class truncate_to_n_sentences(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, dataset, *args, **kwargs):
        dataset['text'] = dataset['text'].apply(truncate_to_n_sentences_single_row(self.n))
        dataset['len_lines'] = dataset['text'].apply(lambda text: len(tokenize.sent_tokenize(text)))
        dataset = dataset[dataset['len_lines'] >= self.n]
        return dataset


def longest_sequence_single_row(row):
    row = row.replace('\n\n', '\n')
    sequences = re.split(r'\n.*[^.]\n', row)
    return max(sequences, key=lambda sequence: len(sequence))


def longest_sequence(dataset):
    dataset['text'] = dataset['text'].apply(longest_sequence_single_row)
    return dataset


def omit_references_single_row(row):
    references_index = row.rfind('References')
    if references_index == -1:
        return row
    return row[:row.rfind('References')]


def omit_references(dataset):
    dataset['text'] = dataset['text'].apply(omit_references_single_row)
    return dataset


def get_data_cleaning_func_from_string(string, args):
    if string == 'clip':
        clip_size = args.n_train_samples + (2 * args.n_test_samples)
        return clip(clip_size)
    elif 'longer_than' in string and 'sentences' in string:
        n = int(string[24:])
        return longer_than_n_sentences(n)
    elif 'longer_than' in string and 'paragraphs' in string:
        n = int(string[25:])
        return longer_than_n_paragraphs(n)
    elif 'truncate_to_n_sentences' in string:
        n = int(string[24:])
        return truncate_to_n_sentences(n)
    elif string == 'longest_sequence':
        return longest_sequence
    elif string == 'omit_references':
        return omit_references
    raise ValueError('no such clean and filter function: ' + string)


def get_data_cleaning_funcs_from_args(args):
    if args.clean_and_filter_funcs is None:
        return None
    data_cleaning_funcs = []
    for data_cleaning_funcs_string in args.clean_and_filter_funcs:
        data_cleaning_func = get_data_cleaning_func_from_string(data_cleaning_funcs_string, args)
        if data_cleaning_func is None:
            return None
        else:
            data_cleaning_funcs.append(data_cleaning_func)
    return data_cleaning_funcs
