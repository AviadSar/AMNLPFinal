import json


class DataLoaderArgs(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.n_data_samples = json_data["n_data_samples"]
            self.wiki_dir = json_data["wiki_dir"]
            self.filtered_data_dir = json_data["filtered_data_dir"]
            self.final_data_dir = json_data["final_data_dir"]
            self.manipulation_func = json_data["manipulation_func"]
            self.clean_and_filter_funcs = json_data["clean_and_filter_funcs"]


class TrainerArgs(object):
    def __init__(self, json_file):
        with open(json_file, 'r') as json_file:
            json_data = json.load(json_file)
            self.n_data_samples = json_data["n_data_samples"]
            self.wiki_dir = json_data["wiki_dir"]
            self.filtered_data_dir = json_data["filtered_data_dir"]
            self.final_data_dir = json_data["final_data_dir"]
            self.manipulation_func = json_data["manipulation_func"]
            self.clean_and_filter_funcs = json_data["clean_and_filter_funcs"]
