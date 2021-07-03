import re


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


def longer_then_10_sentences(dataset):
    cleaned_dataset = dataset[dataset['text'].map(num_lines) > 10]
    return cleaned_dataset


def longer_then_4_paragraphs(dataset):
    cleaned_dataset = dataset[dataset['text'].map(num_paragraphs) > 4]
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
    if string == 'clip_to_10k':
        return clip_to_10k
    elif string == 'num_lines':
        return num_lines
    elif string == 'num_paragraphs':
        return num_paragraphs
    elif string == 'longer_then_3_sentences':
        return longer_then_3_sentences
    elif string == 'longer_then_10_sentences':
        return longer_then_10_sentences
    elif string == 'longer_then_4_paragraphs':
        return longer_then_4_paragraphs
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
            data_cleaning_funcs.append(get_data_cleaning_func_from_string(data_cleaning_funcs_string))
    return data_cleaning_funcs
