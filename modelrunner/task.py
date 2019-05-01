from .configuration import ModelConfiguration
from .cache import Cache
from shortuuid import uuid
from colorama import Fore


###########################################
# run_many([
#   Task(...), Task(...)
# ])
###########################################


def log(message):
    print(Fore.MAGENTA + "task.py: " + Fore.RESET + message)


class Task:
    def __init__(self, model, loader, evaluate, config):
        self.model = model
        self.loader = loader
        self.evaluate = evaluate
        self.config = config
        self.id = uuid()

    def pretty(self):
        return "id={};".format(self.id) + self.config.pretty()


def _run_one(task: Task):
    config = task.config
    model = task.model(**config.hyperparams)
    loader = task.loader
    evaluate = task.evaluate

    train_data = loader(config.data['train_x'], config.data['train_y'], **config.hyperparams)
    dev_data = loader(config.data['dev_x'], config.data['dev_y'], **config.hyperparams)
    test_data = loader(config.data['test_x'], **config.hyperparams)

    model.train(train_data)

    score = evaluate(dev_data, model)

    # Predict tags for the test set
    if config.save_results:
        test_predictions = model.predict(test_data)
        Cache.save(Cache.Types.PREDICTION, task.id, test_predictions)
    return score


def run(tasks: [Task]):
    sequence_id = uuid()
    results_filename = Cache.filename(Cache.Types.TASK_RESULT, sequence_id)
    log("Sequence {} started.".format(sequence_id))
    with open(results_filename, 'a') as file:
        file.write("score, task\n")
    for task in tasks:
        score = _run_one(task)
        log("Task {} finished. Score={}".format(task.id, score))
        with open(results_filename, 'a') as file:
            file.write("{}, {}\n".format(score, task.pretty()))
    log("Sequence {} finished.".format(sequence_id))
