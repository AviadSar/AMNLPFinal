import re


class clip_to(object):
    def __init__(self, clip_size):
        self.clip_size = clip_size

    def __call__(self, dataset, *args, **kwargs):
        if dataset.shape[0] < self.clip_size:
            raise Exception('not enough samples')
        cleaned_dataset = dataset[:][:self.clip_size]
        return cleaned_dataset


def num_lines(text):
    return len(re.split(r'\.[^.]', text))


def num_paragraphs(text):
    return len(text.split('\n'))


class longer_then_n_sentences(object):
    def __init__(self, sentences_limit):
        self.sentences_limit = sentences_limit

    def __call__(self, dataset, *args, **kwargs):
        cleaned_dataset = dataset[dataset['text'].map(num_lines) > self.sentences_limit]
        return cleaned_dataset


class longer_then_n_paragraphs(object):
    def __init__(self, paragraph_limit):
        self.paragraph_limit = paragraph_limit

    def __call__(self, dataset, *args, **kwargs):
        cleaned_dataset = dataset[dataset['text'].map(num_paragraphs) > self.paragraph_limit]
        return cleaned_dataset


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


def get_data_cleaning_func_from_string(string):
    if "clip_to" in string:
        clip_size = int(string[8:])
        return clip_to(clip_size)
    elif "longer_then" in string and "sentences" in string:
        sentences_limit = int(string[24:])
        return longer_then_n_sentences(sentences_limit)
    elif "longer_then" in string and "paragraphs" in string:
        paragraphs_limit = int(string[25:])
        return longer_then_n_paragraphs(paragraphs_limit)
    elif string == 'longest_sequence':
        return longest_sequence
    elif string == 'omit_references':
        return omit_references
    return None


def get_data_cleaning_funcs_from_string_list(string_list):
    data_cleaning_funcs = []
    for data_cleaning_funcs_string in string_list:
        data_cleaning_func = get_data_cleaning_func_from_string(data_cleaning_funcs_string)
        if data_cleaning_func is None:
            return None
        else:
            data_cleaning_funcs.append(data_cleaning_func)
    return data_cleaning_funcs
