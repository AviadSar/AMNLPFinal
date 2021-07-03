import numpy as np


def miss_last_paragraph_3_paragraphs(series):
    n_paragraphs = 3
    text = series['original_text']
    paragraphs = text.split('\n')

    sequence_start_pos = np.random.randint(0, len(paragraphs) - n_paragraphs - 1, 1)[0]
    if series['should_manipulate']:
        paragraphs = paragraphs[sequence_start_pos: sequence_start_pos + n_paragraphs + 1]
    else:
        paragraphs = paragraphs[sequence_start_pos: sequence_start_pos + n_paragraphs]

    if series['should_manipulate']:
        new_paragraphs = paragraphs[:-2] + [paragraphs[-1]]
        series['text'] = '\n'.join(new_paragraphs)
        series['target'] = 1
    else:
        series['text'] = '\n'.join(paragraphs)
        series['target'] = 0

    return series


def miss_last_paragraph(series):
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


def miss_last_sentence_5_sentences(series):
    n_sentences = 5
    text = series['original_text']
    lines = text.split('.')

    sequence_start_pos = np.random.randint(0, len(lines) - n_sentences - 1, 1)[0]
    if series['should_manipulate']:
        lines = lines[sequence_start_pos: sequence_start_pos + n_sentences + 1]
    else:
        lines = lines[sequence_start_pos: sequence_start_pos + n_sentences]

    if series['should_manipulate']:
        new_lines = lines[:-2] + [lines[-1], '']
        series['text'] = '.'.join(new_lines)
        series['target'] = 1
    else:
        series['text'] = '.'.join(lines + [''])
        series['target'] = 0

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
        new_lines = lines[:deleted_sentence_index] + [lines[deleted_sentence_index + 1], '']
        series['text'] = '.'.join(new_lines)
        series['target'] = 1
    else:
        series['text'] = '.'.join(lines[:deleted_sentence_index + 2] + [''])
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


def get_manipulation_func_from_string(string):
    if string == 'miss_last_paragraph_3_paragraphs':
        return miss_last_paragraph_3_paragraphs
    elif string == 'miss_last_paragraph':
        return miss_last_paragraph
    elif string == 'miss_random_paragraph':
        return miss_random_paragraph
    elif string == 'miss_last_sentence_5_sentences':
        return miss_last_sentence_5_sentences
    elif string == 'miss_last_sentence':
        return miss_last_sentence
    elif string == 'miss_last_paragraph':
        return miss_random_sentence
    elif string == 'miss_last_paragraph':
        return miss_random_sentence
    return None
