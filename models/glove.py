from src.utils.utils import load_object, save_object
from src.utils.logger import Logger
import src.constants.const as const

import pandas as pd
import numpy as np
import os.path
import csv


class Glove:
    def __init__(self, file_name=const.GLOVE, load_if_exists=False):
        logger = Logger()
        logger.start_timer('Loading GLoVe embeddings..')
        if load_if_exists and os.path.exists(const.GLOVE_SAVE):
            self.words = load_object(const.GLOVE_SAVE)
        else:
            self.words = pd.read_table(file_name, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            save_object(self.words, const.GLOVE_SAVE)
        logger.end_timer()
        del logger
        self.length = self.words.shape[1]

    def vector(self, word):
        try:
            vec = self.words.loc[word].as_matrix()
        except KeyError:
            vec = np.zeros([self.length, ])
        return vec
