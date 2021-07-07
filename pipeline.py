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
    #     os.system("data_loader.py --json_file " + args.data_loader_args)
    # if args.trainer_args:
    #     os.system("python3 trainer.py --json_file " + args.trainer_args)

    os.system("python3 data_loader.py --json_file " + "linux_args/data_loader_missing_middle_5_sentences_out_of_11_text_target_100k.json")
    os.system("python3 trainer.py --json_file " + "linux_args/trainer_roberta_missing_middle_5_sentences_out_of_11_text_target_100k.json")
