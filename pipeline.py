import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_loader_args",
        help="path to a json file to load data loader arguments from",
        type=str,
        default=None
    )

    parser.add_argument(
        "--trainer_args",
        help="path to a json file to load trainer arguments from",
        type=str,
        default=None
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # if args.data_loader_args:
    #     os.system("python3 data_loader.py --json_file " + args.data_loader_args)
    # if args.trainer_args:
    #     os.system("python3 trainer.py --json_file " + args.trainer_args)
    os.system(
        "python3 trainer.py --json_file " + "linux_args/trainers/facebook/bart-base/missing_middle_5_sentences_out_of_11/classification_target/10k/01/trainer_args.json")
    os.system(
        "python3 trainer.py --json_file " + "linux_args/trainers/facebook/bart-base/missing_middle_5_sentences_out_of_11/classification_target/1m/01/trainer_args.json")
    os.system(
        "python3 trainer.py --json_file " + "linux_args/trainers/facebook/bart-base/missing_middle_5_sentences_out_of_11/classification_target/100k/01/trainer_args.json")
    os.system(
        "python3 trainer.py --json_file " + "linux_args/trainers/roberta-base/missing_middle_5_sentences_out_of_11/classification_target/100k/01/trainer_args.json")
    os.system(
        "python3 trainer.py --json_file " + "linux_args/trainers/gpt2/missing_middle_5_sentences_out_of_11/classification_target/100k/01/trainer_args.json")
