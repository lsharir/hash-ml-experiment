class ModelConfiguration:
    def __init__(self, **kwargs):
        self.hyperparams = {
            'learning_rate': 1,
            'batch_size': 2
        }

        self.hyperparams = {**self.hyperparams, **kwargs}

    def clone(self, **kwargs):
        return ModelConfiguration(**{**self.hyperparams, **kwargs})

    def get(self, key):
        return self.hyperparams[key] if key in self.hyperparams else None

    def pretty(self):
        output = ""
        for key, value in self.hyperparams.items():
            output += "{}={};".format(key, value)
        return output
