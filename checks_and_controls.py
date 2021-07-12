import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizerFast
from tokenizers import AddedToken
from data_loader import read_data_from_csv
from trainer import parse_args


def data_tokens_histograms(data):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [AddedToken('<skip>', lstrip=True), AddedToken('<no_skip>', lstrip=True)]})

    splits_ratio = [1, 1, 0]
    for split, ratio in zip(data, splits_ratio):
        histogram = {}
        text = split['text'].tolist()[:int(len(split['text']) * ratio)]
        n_samples = len(text)
        if n_samples == 0:
            continue

        batch_size = 10000
        batch_idx = 0
        while batch_idx * batch_size < n_samples:
            batch_text = text[batch_idx * batch_size: min((batch_idx + 1) * batch_size, n_samples)]

            encoded_texts = tokenizer(batch_text, return_attention_mask=False, truncation=False, padding=False)['input_ids']
            for encoded_text in encoded_texts:
                text_length = len(encoded_text)
                if text_length in histogram:
                    histogram[text_length] += 1
                else:
                    histogram[text_length] = 1
            print('batch ' + str(batch_idx) + ' done.')
            batch_idx += 1

        greater_than_512 = 0
        max_len_lines = max(histogram.keys()) + 1
        list_histogram = [0] * (max_len_lines + 1)
        for key, value in histogram.items():
            list_histogram[key] = value
            if key > 512:
                greater_than_512 += value

        print('maximal text length (in tokens): ' + str(max(histogram)))
        print('number of entries greater than 512: ' + str(greater_than_512))
        print('histogram values: ' + str(histogram))

        plt.hist(np.array(list_histogram), bins=max_len_lines)
        plt.show()


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


if __name__ == '__main__':
    args = parse_args()
    data = read_data_from_csv("C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\text_target")
    data_tokens_histograms(data)