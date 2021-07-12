import numpy as np
import re
import nltk
from nltk import tokenize
nltk.download('punkt')


def get_n_sentences(text, n):
    lines = tokenize.sent_tokenize(text)

    if n is None:
        return lines
    else:
        sequence_start_pos = np.random.randint(0, len(lines) - (n - 1), 1)[0]
        return lines[sequence_start_pos: sequence_start_pos + n]


def get_n_paragraphs(text, n):
    paragraphs = text.split('\n')
    sequence_start_pos = np.random.randint(0, len(paragraphs) - (n - 1), 1)[0]
    return paragraphs[sequence_start_pos: sequence_start_pos + n]


class remove_second_last_paragraph_out_of_n(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, series, *args, **kwargs):
        n = self.n
        paragraphs = get_n_paragraphs(series['original_text'], n)

        if series['should_manipulate']:
            series['text'] = '\n'.join(paragraphs[:-2] + [paragraphs[-1]])
            series['target'] = 1
        else:
            paragraphs = paragraphs[: -1]
            series['text'] = '\n'.join(paragraphs)
            series['target'] = 0

        return series


class remove_second_last_paragraph_out_of_n_text_target(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, series, *args, **kwargs):
        n = self.n
        paragraphs = get_n_paragraphs(series['original_text'], n)

        if series['should_manipulate']:
            series['text'] = '\n'.join(paragraphs[:-2] + [' <mask>' + paragraphs[-1]])
            series['target'] = '\n'.join(paragraphs[:-2] + [' <skip>' + paragraphs[-1]])
        else:
            paragraphs = paragraphs[: -1]
            series['text'] = '\n'.join(paragraphs[:-1] + [' <mask>' + paragraphs[-1]])
            series['target'] = '\n'.join(paragraphs[:-1] + [' <no_skip>' + paragraphs[-1]])

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
        series['text'] = '\n'.join(paragraphs[:deleted_paragraph_index] + [paragraphs[deleted_paragraph_index + 1]])
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
    def __init__(self, n):
        self.n = n

    def __call__(self, series, *args, **kwargs):
        n = self.n
        lines = get_n_sentences(series['original_text'], n)

        if series['should_manipulate']:
            series['text'] = ' '.join(lines[:-2] + [lines[-1]])
            series['target'] = 1
        else:
            lines = lines[: -1]
            series['text'] = ' '.join(lines)
            series['target'] = 0

        return series


class remove_second_last_sentence_out_of_n_text_target(object):
    def __init__(self, n):
        self.n = n

    def __call__(self, series, *args, **kwargs):
        n = self.n
        lines = get_n_sentences(series['original_text'], n)

        if series['should_manipulate']:
            series['text'] = ' '.join(lines[:-2] + ['<mask> ' + lines[-1]])
            series['target'] = ' '.join(lines[:-2] + ['<skip> ' + lines[-1]])
        else:
            lines = lines[: -1]
            series['text'] = ' '.join(lines[:-1] + ['<mask> ' + lines[-1]])
            series['target'] = ' '.join(lines[:-1] + ['<no_skip> ' + lines[-1]])

        return series


class remove_middle_m_sentences_out_of_n(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, series, *args, **kwargs):
        m = self.m
        n = self.n
        lines = get_n_sentences(series['original_text'], n)

        removed_sequence_start_pos = (n - m) // 2
        removed_sequence_end_pos = n - ((n - m) // 2)  # this index is already after the removed sequence

        if series['should_manipulate']:
            series['text'] = ' '.join(lines[: removed_sequence_start_pos] + lines[removed_sequence_end_pos:])
            series['target'] = 1
        else:
            lines = lines[: -1]
            series['text'] = ' '.join(lines)
            series['target'] = 0

        return series


def remove_all_middle_sentences_text_target(series):
    lines = get_n_sentences(series['original_text'], None)

    if series['should_manipulate']:
        lines[-3] = '<mask> ' + lines[-3]
        series['text'] = ' '.join(lines[: 3] + lines[-3:])
        lines[-3] = '<skip> ' + lines[-3][7:]
        series['target'] = ' '.join(lines[: 3] + lines[-3:])
    else:
        lines = lines[: 6]
        lines[-3] = '<mask> ' + lines[-3]
        series['text'] = ' '.join(lines[:3] + lines[-3:])
        lines[-3] = '<no_skip> ' + lines[-3][7:]
        series['target'] = ' '.join(lines[:3] + lines[-3:])

    return series


class remove_middle_m_sentences_out_of_n_text_target(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, series, *args, **kwargs):
        m = self.m
        n = self.n
        lines = get_n_sentences(series['original_text'], n)

        removed_sequence_start_pos = (n - m) // 2
        removed_sequence_end_pos = n - ((n - m) // 2)  # this index is already after the removed sequence

        if series['should_manipulate']:
            lines[removed_sequence_end_pos] = '<mask> ' + lines[removed_sequence_end_pos]
            series['text'] = ' '.join(lines[: removed_sequence_start_pos] + lines[removed_sequence_end_pos:])
            lines[removed_sequence_end_pos] = '<skip> ' + lines[removed_sequence_end_pos][7:]
            series['target'] = ' '.join(lines[: removed_sequence_start_pos] + lines[removed_sequence_end_pos:])
        else:
            lines = lines[: -self.m]
            lines[removed_sequence_start_pos] = '<mask> ' + lines[removed_sequence_start_pos]
            series['text'] = ' '.join(lines[:removed_sequence_start_pos] + lines[removed_sequence_start_pos:])
            lines[removed_sequence_start_pos] = '<no_skip> ' + lines[removed_sequence_start_pos][7:]
            series['target'] = ' '.join(lines[:removed_sequence_start_pos] + lines[removed_sequence_start_pos:])

        return series


class remove_middle_m_sentences_out_of_n_text_target_with_clue(object):
    def __init__(self, m, n):
        self.m = m
        self.n = n

    def __call__(self, series, *args, **kwargs):
        m = self.m
        n = self.n
        lines = get_n_sentences(series['original_text'], n)

        removed_sequence_start_pos = (n - m) // 2
        removed_sequence_end_pos = n - ((n - m) // 2)  # this index is already after the removed sequence

        if series['should_manipulate']:
            lines[removed_sequence_end_pos] = 'skip <mask> ' + lines[removed_sequence_end_pos]
            series['text'] = ' '.join(lines[: removed_sequence_start_pos] + lines[removed_sequence_end_pos:])
            lines[removed_sequence_end_pos] = 'skip <skip> ' + lines[removed_sequence_end_pos][12:]
            series['target'] = ' '.join(lines[: removed_sequence_start_pos] + lines[removed_sequence_end_pos:])
        else:
            lines = lines[: -self.m]
            lines[removed_sequence_start_pos] = 'continue <mask> ' + lines[removed_sequence_start_pos]
            series['text'] = ' '.join(lines[:removed_sequence_start_pos] + lines[removed_sequence_start_pos:])
            lines[removed_sequence_start_pos] = 'continue <no_skip> ' + lines[removed_sequence_start_pos][16:]
            series['target'] = ' '.join(lines[:removed_sequence_start_pos] + lines[removed_sequence_start_pos:])

        return series


def remove_second_last_sentence(series):
    lines = get_n_sentences(series['original_text'], None)

    deleted_sentence_index = -1
    while deleted_sentence_index == -1:
        deleted_sentence_index = np.random.randint(1, len(lines) - 2, 1)[0]
        if not lines[deleted_sentence_index]:
            deleted_sentence_index = -1

    if series['should_manipulate']:
        new_lines = lines[:deleted_sentence_index] + [lines[deleted_sentence_index + 1]]
        series['text'] = ' '.join(new_lines)
        series['target'] = 1
    else:
        series['text'] = ' '.join(lines[:deleted_sentence_index + 2])
        series['target'] = 0

    return series


def remove_random_sentence(series):
    if series['should_manipulate']:
        lines = get_n_sentences(series['original_text'], None)

        deleted_sentence_index = -1
        while deleted_sentence_index == -1:
            deleted_sentence_index = np.random.randint(1, len(lines) - 2, 1)[0]
            if not lines[deleted_sentence_index]:
                deleted_sentence_index = -1

        new_lines = lines[:deleted_sentence_index] + lines[deleted_sentence_index + 1:]
        series['text'] = ' '.join(new_lines)

        new_lines[deleted_sentence_index] = '<skip>' + new_lines[deleted_sentence_index]
        series['target'] = ' '.join(new_lines)

    return series


def get_manipulation_func_from_args(args):
    func_name = args.manipulation_func
    func_args = args.manipulation_func_args

    if func_name == 'remove_second_last_paragraph_out_of_n_text_target':
        return remove_second_last_paragraph_out_of_n_text_target(func_args[0])
    elif func_name == 'remove_second_last_paragraph_out_of_n':
        return remove_second_last_paragraph_out_of_n(func_args[0])
    elif func_name == 'remove_second_last_paragraph':
        return remove_second_last_paragraph
    elif func_name == 'remove_random_paragraph':
        return remove_random_paragraph
    elif func_name == 'remove_all_middle_sentences_text_target':
        return remove_all_middle_sentences_text_target
    elif func_name == 'remove_middle_m_sentences_out_of_n_text_target':
        return remove_middle_m_sentences_out_of_n_text_target(func_args[0], func_args[1])
    elif func_name == 'remove_middle_m_sentences_out_of_n_text_target_with_clue':
        return remove_middle_m_sentences_out_of_n_text_target_with_clue(func_args[0], func_args[1])
    elif func_name == 'remove_middle_m_sentences_out_of_n':
        return remove_middle_m_sentences_out_of_n(func_args[0], func_args[1])
    elif func_name == 'remove_second_last_sentence_out_of_n_text_target':
        return remove_second_last_sentence_out_of_n_text_target(func_args[0])
    elif func_name == 'remove_second_last_sentence_out_of_n':
        return remove_second_last_sentence_out_of_n(func_args[0])
    elif func_name == 'remove_second_last_sentence':
        return remove_second_last_sentence
    elif func_name == 'remove_random_sentence':
        return remove_random_sentence
    return None
