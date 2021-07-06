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
    # args = parse_args()
    #
    # if args.data_loader_args:
    #     os.system("data_loader.py --json_file " + args.data_loader_args)
    # if args.trainer_args:
    #     os.system("trainer.py --json_file " + args.trainer_args)

    os.system("trainer.py --json_file " + "args/trainer_roberta_missing_second_last_paragraph_out_of_3_text_target_10k.json")
    os.system("trainer.py --json_file " + "args/trainer_roberta_missing_second_last_sentence_out_of_5_text_target_10k.json")