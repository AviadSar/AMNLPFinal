import numpy as np


class remove_second_last_paragraph_out_of_n(object):
    def __init__(self, num_paragraphs):
        self.num_paragraphs = num_paragraphs

    def __call__(self, series, *args, **kwargs):
        num_paragraphs = self.num_paragraphs
        text = series['original_text']
        paragraphs = text.split('\n')

        sequence_start_pos = np.random.randint(0, len(paragraphs) - num_paragraphs - 1, 1)[0]
        if series['should_manipulate']:
            paragraphs = paragraphs[sequence_start_pos: sequence_start_pos + num_paragraphs + 1]
        else:
            paragraphs = paragraphs[sequence_start_pos: sequence_start_pos + num_paragraphs]

        if series['should_manipulate']:
            new_paragraphs = paragraphs[:-2] + [paragraphs[-1]]
            series['text'] = '\n'.join(new_paragraphs)
            series['target'] = 1
        else:
            series['text'] = '\n'.join(paragraphs)
            series['target'] = 0

        return series


def remove_second_last_paragraph(series):
    text = series['original_text']
    paragraphs = text.split('\n')

    deleted_paragraph_index = -1
    while deleted_paragraph_index == -1:
        deleted_paragraph_index = np.random.randint(1, len(paragraphs) - 1, 1)[0]
        if not paragraphs[deleted_paragraph_index]:
            deleted_paragraph_index = -1

    if series['should_manipulate']:
        new_paragraphs = paragraphs[:deleted_paragraph_index] + [paragraphs[deleted_paragraph_index + 1]]
        series['text'] = '\n'.join(new_paragraphs)
        series['target'] = 1
    else:
        series['text'] = '\n'.join(paragraphs[:deleted_paragraph_index + 2])
        series['target'] = 0

    return series


def remove_random_paragraph(series):
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


class remove_second_last_sentence_out_of_n(object):
    def __init__(self, num_sentences):
        self.num_sentences = num_sentences

    def __call__(self, series, *args, **kwargs):
        num_sentences = self.num_sentences
        text = series['original_text']
        lines = text.split('.')
        if lines[-1]:
            lines.append('')

        sequence_start_pos = np.random.randint(0, len(lines) - num_sentences - 1, 1)[0]
        if series['should_manipulate']:
            lines = lines[sequence_start_pos: sequence_start_pos + num_sentences + 1]
        else:
            lines = lines[sequence_start_pos: sequence_start_pos + num_sentences]

        if series['should_manipulate']:
            new_lines = lines[:-2] + [lines[-1], '']
            series['text'] = '.'.join(new_lines)
            series['target'] = 1
        else:
            series['text'] = '.'.join(lines + [''])
            series['target'] = 0

        return series


def remove_second_last_sentence(series):
    text = series['original_text']
    lines = text.split('.')

    deleted_sentence_index = -1
    while deleted_sentence_index == -1:
        deleted_sentence_index = np.random.randint(1, len(lines) - 2, 1)[0]
        if not lines[deleted_sentence_index]:
            deleted_sentence_index = -1

    if series['should_manipulate']:
        new_lines = lines[:deleted_sentence_index] + [lines[deleted_sentence_index + 1], '']
        series['text'] = '.'.join(new_lines)
        series['target'] = 1
    else:
        series['text'] = '.'.join(lines[:deleted_sentence_index + 2] + [''])
        series['target'] = 0

    return series


def remove_random_sentence(series):
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


def get_manipulation_func_from_string(string):
    if 'remove_second_last_paragraph_out_of' in string:
        return remove_second_last_paragraph_out_of_n(int(string[38:]))
    elif string == 'remove_second_last_paragraph':
        return remove_second_last_paragraph
    elif string == 'remove_random_paragraph':
        return remove_random_paragraph
    elif 'remove_second_last_sentence_out_of' in string:
        return remove_second_last_sentence_out_of_n(int(string[37:]))
    elif string == 'remove_second_last_sentence':
        return remove_second_last_sentence
    elif string == 'remove_random_sentence':
        return remove_random_sentence
    return None
