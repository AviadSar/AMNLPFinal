import json


class DataLoaderArgs(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.n_data_samples = json_data["n_data_samples"]
            self.n_train_samples = json_data["n_train_samples"]
            self.n_test_samples = json_data["n_test_samples"]
            self.wiki_dir = json_data["wiki_dir"]
            self.filtered_data_dir = json_data["filtered_data_dir"]
            self.final_data_dir = json_data["final_data_dir"]
            self.manipulation_func = json_data["manipulation_func"]
            self.manipulation_func_args = json_data["manipulation_func_args"]
            self.clean_and_filter_funcs = json_data["clean_and_filter_funcs"]


class TrainerArgs(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.data_dir = json_data["data_dir"]
            self.model_dir = json_data["model_dir"]
            self.model_name = json_data["model_name"]
            self.model_type = json_data["model_type"]
            self.batch_size = json_data["batch_size"]
            self.data_split_ratio = json_data["data_split_ratio"]
            self.logging_steps = json_data["logging_steps"]
            self.eval_steps = json_data["eval_steps"]
            self.num_evals = json_data["num_evals"]
            self.dropout = json_data["dropout"]
