# coding: utf-8

import numpy as np
from sklearn.naive_bayes import MultinomialNB

class MyClassifier:

    def __init__(self):
        self.learned_model = None

    def classify(self, mat):
        classified_labels = self.learned_model.predict(mat)
        attributes = ["company" if i==0 else "bank" if i==2 else "adress" if i==1 else "" for i in classified_labels]
        return attributes

    def load(self, learned_model):
        self.learned_model = learned_model 