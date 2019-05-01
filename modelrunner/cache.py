from os import path, makedirs
from .utils import ConstantDict
from enum import Enum
import numpy as np
from colorama import Fore
import pandas as pd


###########################################
# Cache.save(Cache.Types.PREDICTION, id)
# Cache.filename(Cache.Types.PREDICTION, id)
# Cache.load(Cache.Types.PREDICTION, id)
###########################################

root = path.abspath(path.join(path.dirname(path.realpath(__file__)), "../"))
cache_root = path.abspath(path.join(root, "./cache"))


def log(message):
	print(Fore.CYAN + "cache.py: " + Fore.RESET + message)


def cache_path(cache_name):
	cache_abs_path = path.abspath(path.join(cache_root, "./" + cache_name))
	if not path.exists(cache_abs_path):
		makedirs(cache_abs_path)
	return cache_abs_path


class CacheDirs(ConstantDict):
	ROOT = cache_path("")
	TASKS_RESULTS = cache_path("tasks_results")
	MODELS = cache_path("models")
	PREDICTIONS = cache_path("predictions")


CACHE_DIRS = CacheDirs()


class Cache:
	class Types(Enum):
		TASK_RESULT = 1
		MODEL = 2
		PREDICTION = 3

	@staticmethod
	def filename(filetype: Types, uid):
		return cache_filenames[filetype](uid)

	@staticmethod
	def save(filetype: Types, uid, data):
		filename = Cache.filename(filetype, uid)

		if filetype == Cache.Types.TASK_RESULT:
			np.savetxt(filename, data)
			log("Saved at {}".format(filename))

		if filetype == Cache.Types.PREDICTION:
			# Prediction: data should be a list of predicted tags
			df = pd.DataFrame(data)
			df.to_csv(filename, header=["tag"], index_label="id")
			log("Saved at {}".format(filename))

		if filetype == Cache.Types.MODEL:
			log("TODO handle model save")

		return filename

	@staticmethod
	def load(filetype: Types, uid, data):
		filename = Cache.filename(filetype, uid)

		if filetype == Cache.Types.TASK_RESULT:
			np.loadtxt(filename, data)
			log("Loaded from {}".format(filename))

		if filetype == Cache.Types.PREDICTION:
			np.loadtxt(filename, data)
			log("Loaded from {}".format(filename))

		if filetype == Cache.Types.MODEL:
			log("TODO handle model load")

		return filename


cache_filenames = {
	Cache.Types.TASK_RESULT: lambda uid: CACHE_DIRS.TASKS_RESULTS + "/sequence_" + uid + ".csv",
	Cache.Types.MODEL: lambda uid: CACHE_DIRS.MODELS + "/model_" + uid + ".pth",
	Cache.Types.PREDICTION: lambda uid: CACHE_DIRS.PREDICTIONS + "/prediction_" + uid + ".csv"
}
