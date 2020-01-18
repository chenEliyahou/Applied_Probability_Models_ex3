from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class EM:
    def __init__(self, documents, categories):
        self.k = 10
        self.documents = documents
        self.categories = categories
        self.word_frequency = defaultdict(Counter)
        self.parse_words(documents)
        self.filter_rare_word()

    def parse_words(self, documents):
        for index, doc in enumerate(documents):
            for word in doc:
                self.word_frequency[word][index] += 1

    def filter_rare_word(self, threshold=3):
        self.word_frequency = {x: counter for x, counter in self.word_frequency.items() if
                               sum(counter.values()) >= threshold}

    def start(self):
        eps = 0.1
        alpha, p = self.initialize()
        likelihoods = list()
        while True:
            start = datetime.now()
            w, likelihood = self.cal_w(alpha, p)
            print('w running time: {0}'.format(datetime.now() - start))
            likelihoods.append(likelihood)

            start = datetime.now()
            alpha = self.cal_alpha(w)
            print('alpha running time: {0}'.format(datetime.now() - start))

            start = datetime.now()
            p = self.cal_p(w)
            print('p running time: {0}'.format(datetime.now() - start))
            print('likelihoods: ', likelihoods[-1])
            if len(likelihoods) > 1 and abs(likelihoods[-1] - likelihoods[-2]) <= eps:
                break
        EM.plot_likelihoods(likelihoods)

    def initialize(self):
        w = self.initialize_w()
        alpha = self.cal_alpha(w)
        p = self.cal_p(w)
        return alpha, p

    def initialize_w(self):
        w = np.zeros((len(self.documents), len(self.categories)))
        for t in range(len(self.documents)):
            mod = t % len(self.categories)
            w[t][mod] = 1
        return w

    def cal_alpha(self, w):
        alpha = np.zeros(len(self.categories))  # could be hard coded values according to % mod
        for i in range(len(self.categories)):
            for t in range(len(self.documents)):
                alpha[i] += w[t][i]
            alpha[i] /= len(self.documents)
        return EM.smoothing_alpha(alpha)

    def cal_p(self, w, lamda=0.0001):
        vocabulary_size = len(self.word_frequency.keys())
        p = np.zeros((len(self.categories), len(self.word_frequency.keys())))
        for i in range(len(self.categories)):
            for k, word in enumerate(self.word_frequency.keys()):
                up_sum, down_sum = 0, 0
                for t in range(len(self.documents)):
                    up_sum += w[t][i] * self.word_frequency[word][t]
                    down_sum += w[t][i] * len(self.documents[t])
                p[i][k] = (up_sum + lamda) / (down_sum + lamda * vocabulary_size)
        return p

    def cal_w(self, alpha, p):
        ln_likelihood = 0
        w = np.zeros((len(self.documents), len(self.categories)))
        for t in range(len(self.documents)):
            z = [self.cal_z(alpha, p, j, t) for j in range(len(self.categories))]
            m = max(z)
            indexes = list()
            x = list()
            for i in range(len(self.categories)):
                if z[i] - m >= -self.k:
                    indexes.append(i)
                    x.append(z[i] - m)
            x = np.exp(x)
            d = sum(x)
            for i, exp in zip(indexes, x):
                w[t][i] = exp / d
            ln_likelihood += m + np.math.log(d)
        return w, ln_likelihood

    def cal_z(self, alpha, p, i, t):
        return np.math.log(alpha[i], np.math.e) + \
               sum([self.word_frequency[word][t] * np.math.log(p[i][k], np.math.e) for k, word in
                    enumerate(self.word_frequency.keys())])

    # def cal_ln_likelihood(self, alpha, p):
    #     ln_likelihood = 0
    #     for t in range(len(self.documents)):
    #         z = [self.cal_z(alpha, p, j, t) for j in range(len(self.categories))]
    #         m = max(z)
    #         x = [z[i] - m for i in range(len(self.categories)) if z[i] - m >= -self.k]
    #         ln_likelihood += m + np.math.log(sum(np.exp(x)))
    #     return ln_likelihood

    @staticmethod
    def smoothing_alpha(alpha, eps=0.001):
        for i in range(len(alpha)):
            if alpha[i] < eps:
                alpha[i] = eps
        normal = sum(alpha)
        return [a / normal for a in alpha]

    @staticmethod
    def plot_likelihoods(likelihoods):
        plt.plot(range(len(likelihoods)), likelihoods, label='linear')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
