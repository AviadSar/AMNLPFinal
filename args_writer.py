import os
import json

operating_systems = ['windows', 'linux']
batch_sizes = [4, 16]

sizes = ['10k', '100k', '1m']
data_split_ratios = [[0.01, 1, 0], [0.1, 1, 0], [1, 1, 0]]
num_evals = [50, 100, 100]

task_names = ['missing_middle_5_sentences_out_of_11']
model_types = ['token_classification']

model_names = ['roberta-base']
targets = ['text_target']

dropouts = [0.1, 0.11, 0.125, 0.15]

for os_idx, operating_system in enumerate(operating_systems):
    for model_name in model_names:
        for task_idx, task_name in enumerate(task_names):
            for target in targets:
                for size_idx, size in enumerate(sizes):
                    for dropout in dropouts:
                        if operating_system == 'windows':
                            args_file_dir = operating_system + '_args\\trainers\\' + model_name + '\\' + task_name + '\\' + target + '\\' + size + '\\' + str(dropout)[0] + str(dropout)[2:]
                            args_file = '\\trainer_args.json'
                            data_dir = 'C:\\my_documents\\AMNLPFinal\\datasets\\' + task_name + '\\' + target
                            model_dir = 'C:\\my_documents\\AMNLPFinal\\models\\' + model_name + '\\' + task_name + '\\' + target + '\\' + size + '\\' + str(dropout)[0] + str(dropout)[2:]
                        elif operating_system == 'linux':
                            args_file_dir = operating_system + r'_args/trainers/' + model_name + '/' + task_name + '/' + target + '/' + size + '/' + str(dropout)[0] + str(dropout)[2:]
                            args_file = '/trainer_args.json'
                            data_dir = '/home/aviad/Documents/AMNLPFinal/datasets/' + task_name + '/' + target
                            model_dir = '/home/aviad/Documents/AMNLPFinal/models/' + model_name + '/' + task_name + '/' + target + '/' + size + '/' + str(dropout)[0] + str(dropout)[2:]
                        else:
                            raise ValueError('No such operating system: ' + operating_system)

                        data_dict = {
                                      "data_dir": data_dir,
                                      "model_dir": model_dir,
                                      "model_name": model_name,
                                      "model_type": model_types[task_idx],
                                      "batch_size": batch_sizes[os_idx],
                                      "data_split_ratio": data_split_ratios[size_idx],
                                      "logging_steps": 10,
                                      "eval_steps": 78,
                                      "num_evals": num_evals[size_idx],
                                      "dropout": dropout
                                    }

                        if not os.path.isdir(args_file_dir):
                            os.makedirs(args_file_dir)
                            print("creating dataset directory " + args_file_dir)
                        with open(args_file_dir + args_file, 'w') as json_file:
                            json.dump(data_dict, json_file, indent=4)