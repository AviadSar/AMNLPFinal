import os
import json
from matplotlib import pyplot as plt


class Logger(object):
    def __init__(self, args, start_epoch):
        self.logs_dir = args.model_dir + os.path.sep + 'logs'
        if not os.path.isdir(self.logs_dir):
            os.makedirs(self.logs_dir)
            print("creating dataset directory " + self.logs_dir)

        self.experiment_name = os.path.basename(os.path.normpath(args.model_dir))
        self.train_loss = []
        self.eval_loss = []
        self.eval_accuracy = []
        self.best_eval_accuracy = 0
        self.best_model_epoch = 0

        if start_epoch != 0:
            self.init_from_previous_log()

    def init_from_previous_log(self):
        with open(self.logs_dir + os.path.sep + 'log.json', 'r') as json_file:
            json_data = json.load(json_file)
            self.train_loss = json_data["train_loss"]
            self.eval_loss = json_data["eval_loss"]
            self.eval_accuracy = json_data["eval_accuracy"]
            self.best_model_epoch = json_data["best_model_epoch"]

    def write_json(self):
        data_dict = {'train_loss': self.train_loss, 'eval_loss': self.eval_loss, 'eval_accuracy': self.eval_accuracy,
                     'best_model_epoch': self.best_model_epoch}

        with open(self.logs_dir + os.path.sep + 'log.json', 'w') as json_file:
            json.dump(data_dict, json_file)

    def draw_train_graphs(self):
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='train')
        plt.plot(range(1, len(self.eval_loss) + 1), self.eval_loss, label='evaluation')
        plt.xticks(range(1, len(self.train_loss) + 1))
        plt.legend(loc='upper right')
        plt.title('Train and evaluation loss over training\n' + self.experiment_name)
        plt.savefig(self.logs_dir + os.path.sep + 'loss.jpg')
        plt.figure()

        plt.plot(range(1, len(self.eval_accuracy) + 1), self.eval_accuracy, label='evaluation accuracy')
        plt.xticks(range(1, len(self.eval_accuracy) + 1))
        plt.legend(loc='upper right')
        plt.title('Evaluation accuracy over training\n' + self.experiment_name)
        plt.savefig(self.logs_dir + os.path.sep + 'accuracy.jpg')
        plt.figure()

    def update(self, train_loss, eval_loss, eval_accuracy):
        # due to a bug in huggingface trainer, the training loss is zeroed after resuming from checkpoints,
        # and thus a manual calculation is required
        if self.train_loss:
            self.train_loss.append(((self.train_loss[-1] * len(self.train_loss)) /
                                   (len(self.train_loss) + 1))
                                   + train_loss)
        else:
            self.train_loss.append(train_loss)
        self.eval_loss.append(eval_loss)
        self.eval_accuracy.append(eval_accuracy)
        if eval_accuracy > max(self.eval_accuracy):
            self.best_model_epoch = len(eval_accuracy)

        self.write_json()
        self.draw_train_graphs()
