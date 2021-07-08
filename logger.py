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

    def update(self, train_loss, eval_loss, eval_accuracy):
        # due to a bug in huggingface trainer, the training loss is zeroed after resuming from checkpoints,
        # and thus a manual calculation is required
        if self.train_loss:
            self.train_loss.append(train_loss * (len(self.train_loss) + 1))
        else:
            self.train_loss.append(train_loss)
        self.eval_loss.append(eval_loss)
        self.eval_accuracy.append(eval_accuracy)
        if eval_accuracy > max(self.eval_accuracy):
            self.best_model_epoch = len(eval_accuracy)

        write_json(self.train_loss, range(1, len(self.train_loss) + 1), self.eval_loss, range(1, len(self.eval_loss) + 1), self.eval_accuracy, self.logs_dir)
        draw_train_graphs(self.train_loss, range(1, len(self.train_loss) + 1), self.eval_loss, range(1, len(self.eval_loss) + 1), self.eval_accuracy, self.experiment_name, self.logs_dir)


def write_json(train_loss, train_steps, eval_loss, eval_steps, eval_accuracy, experiment_name, logs_dir):
    data_dict = {'train_loss': train_loss, 'train_steps': train_steps, 'eval_loss': eval_loss, 'eval_steps': eval_steps, 'eval_accuracy': eval_accuracy, 'experiment_name': experiment_name}

    with open(logs_dir + os.path.sep + 'log.json', 'w') as json_file:
        json.dump(data_dict, json_file)


def draw_train_graphs(train_loss, train_steps, eval_loss, eval_steps, eval_accuracy, experiment_name, logs_dir):
    plt.plot(train_steps, train_loss, label='train')
    plt.plot(eval_steps, eval_loss, label='evaluation')
    plt.xticks(train_steps)
    plt.legend(loc='upper right')
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title('Train and evaluation loss over training\n' + experiment_name)
    plt.savefig(logs_dir + os.path.sep + 'loss.jpg')
    plt.close()

    plt.plot(eval_steps, eval_accuracy, label='evaluation accuracy')
    plt.xticks(train_steps)
    plt.legend(loc='upper right')
    plt.title('Evaluation accuracy over training\n' + experiment_name)
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.savefig(logs_dir + os.path.sep + 'accuracy.jpg')
    plt.close()


def log_from_trainer_state(trainer_state, model_dir):
    logs_dir = model_dir + os.path.sep + 'logs'
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)
        print("creating dataset directory " + logs_dir)

    experiment_name = os.path.basename(os.path.normpath(model_dir))
    train_loss = []
    train_steps = []
    eval_loss = []
    eval_accuracy = []
    eval_steps = []

    log_history = trainer_state.log_history
    for index, log in enumerate(log_history):
        if 'loss' in log:
            train_loss.append(log['loss'])
            train_steps.append(log['step'])
        if 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])
            eval_accuracy.append(log['eval_accuracy'])
            eval_steps.append(log['step'])

    draw_train_graphs(train_loss, train_steps, eval_loss, eval_steps, eval_accuracy, experiment_name, logs_dir)
    write_json(train_loss, train_steps, eval_loss, eval_steps, eval_accuracy, experiment_name, logs_dir)


def log_from_trainer_state_file(trainer_state_file, model_dir):
    with open(trainer_state_file, 'r') as trainer_state_file:
        trainer_state = json.load(trainer_state_file)
    log_from_trainer_state(trainer_state, model_dir)
