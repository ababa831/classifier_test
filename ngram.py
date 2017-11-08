# coding: utf-8

import collections
import json
import numpy as np


class MyCountVectorizer:

    def __init__(self):
        self.vocaburaly_ = None

    def transform(self, raw_documents):
        mat = np.zeros(shape=[len(raw_documents), len(self.vocaburaly_)])
        for row, d in enumerate(raw_documents):
            # if type(d) is not unicode:
            #     d = d.decode("utf-8")
            counter = collections.Counter(d)
            for c in counter:
                if c in self.vocaburaly_:
                    mat[row, self.vocaburaly_[c]] = counter[c]
        return mat

    def load(self, vocab):
        self.vocaburaly_ = vocab
