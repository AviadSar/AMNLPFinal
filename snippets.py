import data_loader
import numpy as np
import pandas as pd
import json
import csv
import os


def create_human_eval():
    data = data_loader.read_data_from_csv("C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\classification_target")
    dev = data[1]
    np.random.seed(40)
    chosen_indices = np.random.choice(range(len(dev)), 100, replace=False)
    chosen_samples = dev.iloc[chosen_indices]
    chosen_samples.to_csv("C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\classification_target_human_eval.tsv", sep='\t')


class agreement(object):
    def __init__(self, columns):
        self.columns = columns

    def __call__(self, row, *args, **kwargs):
        for index in range(len(self.columns)):
            if index == 0:
                continue
            if row[self.columns[index]] != row[self.columns[index - 1]]:
                return False
        return True


def machine_agreement(row):
    if row['roberta-base_sequence_classification'] == row['gpt2_sequence_classification']\
            == row['facebook/bart-base_sequence_classification']:
        return 1
    else:
        return 0


def all_agreement(row):
    if row['roberta-base_sequence_classification'] == row['gpt2_sequence_classification']\
            == row['facebook/bart-base_sequence_classification'] == row['me']:
        return 1
    else:
        return 0


def check_evals():
    # eval = pd.read_csv('C:\\Users\\aavia\\OneDrive\\Documents\\study\\tau\\advanced_nlp\\project\\human_evaluation\\text_target_human_eval.tsv', sep='\t')
    eval = pd.read_csv('C:\\my_documents\\AMNLPFinal\\datasets\\missing_middle_5_sentences_out_of_11\\classification_target\\evaluated.tsv', sep='\t')

    eval['should_not_manipulate'] = eval['should_manipulate'].apply(lambda x: 1 - x)

    # eval['machine_agreement'] = eval.apply(agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
    #                                              'facebook/bart-base_sequence_classification')), axis=1)
    # eval['all_agreement'] = eval.apply(agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
    #                                              'facebook/bart-base_sequence_classification', 'me')), axis=1)
    eval['machine_agreement_wrong'] = eval.apply(
        agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
                   'facebook/bart-base_sequence_classification', 'should_not_manipulate')), axis=1)
    # eval['all_agreement_wrong'] = eval.apply(
    #     agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
    #                'facebook/bart-base_sequence_classification', 'me', 'should_not_manipulate')), axis=1)
    # eval['all_agreement_right'] = eval.apply(
    #     agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
    #                'facebook/bart-base_sequence_classification', 'me', 'should_manipulate')), axis=1)
    eval['roberta_wrong'] = eval.apply(
        agreement(('roberta-base_sequence_classification', 'should_not_manipulate')), axis=1)
    eval['gpt2_wrong'] = eval.apply(
        agreement(('gpt2_sequence_classification', 'should_not_manipulate')), axis=1)
    eval['bart_wrong'] = eval.apply(
        agreement(('facebook/bart-base_sequence_classification', 'should_not_manipulate')), axis=1)
    # eval['me_wrong'] = eval.apply(
    #     agreement(('me', 'should_not_manipulate')), axis=1)

    # eval['roberta_bart_wrong'] = eval.apply(
    #     agreement(('roberta-base_sequence_classification', 'facebook/bart-base_sequence_classification',
    #                'should_not_manipulate')), axis=1)
    # eval['roberta_gpt2_wrong'] = eval.apply(
    #     agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification',
    #                'should_not_manipulate')), axis=1)
    # eval['bart_gpt2_wrong'] = eval.apply(
    #     agreement(('facebook/bart-base_sequence_classification', 'gpt2_sequence_classification',
    #                'should_not_manipulate')), axis=1)

    print(len(eval[eval['roberta_wrong'] == 1]))
    print(len(eval[(eval['roberta_wrong'] == 1) & (eval['roberta-base_sequence_classification'] == 1)]))
    print(len(eval[eval['gpt2_wrong'] == 1]))
    print(len(eval[(eval['gpt2_wrong'] == 1) & (eval['gpt2_sequence_classification'] == 1)]))
    print(len(eval[eval['bart_wrong'] == 1]))
    print(len(eval[(eval['bart_wrong'] == 1) & (eval['facebook/bart-base_sequence_classification'] == 1)]))

    print(len(eval[eval['machine_agreement_wrong'] == 1]))
    print(len(eval[(eval['machine_agreement_wrong'] == 1) & (eval['facebook/bart-base_sequence_classification'] == 0)]))


    # eval['roberta_me_agreement'] = eval.apply(agreement(('roberta-base_sequence_classification', 'me')), axis=1)
    # eval['gpt2_me_agreement'] = eval.apply(agreement(('gpt2_sequence_classification', 'me')), axis=1)
    # eval['bart_me_agreement'] = eval.apply(agreement(('facebook/bart-base_sequence_classification', 'me')), axis=1)
    #
    # eval['roberta_gpt2_agreement'] = eval.apply(agreement(('roberta-base_sequence_classification', 'gpt2_sequence_classification')), axis=1)
    # eval['roberta_bart_agreement'] = eval.apply(agreement(('roberta-base_sequence_classification', 'facebook/bart-base_sequence_classification')), axis=1)
    # eval['gpt2_bart_agreement'] = eval.apply(agreement(('gpt2_sequence_classification', 'facebook/bart-base_sequence_classification')), axis=1)
    #
    # print(len(eval[eval['machine_agreement'] == 1]))
    # print(len(eval[eval['all_agreement'] == 1]))
    # print(len(eval[eval['machine_agreement_wrong'] == 1]))
    # print(len(eval[eval['all_agreement_wrong'] == 1]))
    # print(len(eval[eval['all_agreement_right'] == 1]))
    #
    # print(len(eval[eval['roberta_me_agreement'] == 1]))
    # print(len(eval[eval['gpt2_me_agreement'] == 1]))
    # print(len(eval[eval['bart_me_agreement'] == 1]))
    #
    # print(len(eval[eval['roberta_gpt2_agreement'] == 1]))
    # print(len(eval[eval['roberta_bart_agreement'] == 1]))
    # print(len(eval[eval['gpt2_bart_agreement'] == 1]))


    # print('ALL WRONG')
    # for row in eval[eval['all_agreement_wrong'] == 1].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')
    # print('ALL RIGHT')
    # for row in eval[eval['all_agreement_right'] == 1][:5].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')

    # print(len(eval[eval['roberta-base_sequence_classification'] == 1]))
    # print(len(eval[eval['gpt2_sequence_classification'] == 1]))
    # print(len(eval[eval['facebook/bart-base_sequence_classification'] == 1]))
    # print(len(eval[eval['me'] == 1]))

    # print('MACHINE WRONG')
    # print(len(eval[eval['machine_agreement_wrong'] == 1]))
    # for row in eval[eval['machine_agreement_wrong'] == 1].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')
    # print('ROBERTA WRONG')
    # print(len(eval[eval['roberta_wrong'] == 1]))
    # for row in eval[eval['roberta_wrong'] == 1].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')
    # print('GPT2 WRONG')
    # print(len(eval[eval['gpt2_wrong'] == 1]))
    # for row in eval[eval['gpt2_wrong'] == 1].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')
    # print('BART WRONG')
    # print(len(eval[eval['bart_wrong'] == 1]))
    # for row in eval[eval['bart_wrong'] == 1].iterrows():
    #     print(row[1]['text'])
    #     print(str(row[1]['should_manipulate']) + '\n')

    # eval.to_csv('C:\\Users\\aavia\\OneDrive\\Documents\\study\\tau\\advanced_nlp\\project\\human_evaluation\\text_target_human_eval.tsv', sep='\t')


def get_data_from_json_log(json_log, dirpath):
    best_accuracy_index = np.argmax(json_log['eval_accuracy'])
    best_eval_loss_index = np.argmax(json_log['eval_loss'])
    best_train_loss_index = np.argmax(json_log['train_loss'])

    return {
        'experiment_name': json_log['experiment_name'],
        'directory': dirpath,
        'best_accuracy': json_log['eval_accuracy'][best_accuracy_index],
        'best_accuracy_loss': json_log['eval_loss'][best_accuracy_index],
        'best_accuracy_index': int(best_accuracy_index),
        'best_eval_loss': json_log['eval_loss'][best_eval_loss_index],
        'best_eval_loss_accuracy': json_log['eval_accuracy'][best_eval_loss_index],
        'best_eval_loss_index': int(best_eval_loss_index),
        'best_train_loss': json_log['train_loss'][best_train_loss_index],
        'best_train_loss_index': int(best_train_loss_index)
    }


def assemble_train_data():
    data_dicts = []
    for dirpath, dirnames, filenames in os.walk('C:\\my_documents\\AMNLPFinal\\models'):
        for filename in [f for f in filenames if f.endswith('log.json')]:
            with open(os.path.join(dirpath, filename), 'r') as log_file:
                json_log = json.load(log_file)
                data_dict = get_data_from_json_log(json_log, dirpath)

                # data_dicts[json_log['experiment_name']] = data_dict
                data_dicts.append(data_dict)

    filename = 'assembled_train_data\\data'
    with open(filename + '.tsv', 'w') as tsv_file:
        fieldnames = ['experiment_name', 'directory', 'best_accuracy', 'best_accuracy_loss', 'best_accuracy_index',
                      'best_eval_loss', 'best_eval_loss_accuracy', 'best_eval_loss_index', 'best_train_loss',
                      'best_train_loss_index']
        tsv_writer = csv.DictWriter(tsv_file, fieldnames=fieldnames, dialect='excel-tab')

        tsv_writer.writeheader()
        tsv_writer.writerows(data_dicts)

    with open(filename + '.json', 'w') as json_file:
        json.dump(data_dicts, json_file, indent=4)

assemble_train_data()
# check_evals()
