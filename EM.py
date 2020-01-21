from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


class EM:
    def __init__(self, documents, categories):
        self.k = 10
        self.vocabulary_size = 0
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
        word_frequency = defaultdict(lambda: defaultdict(int))
        for word, frequencies in self.word_frequency.items():
            if sum(frequencies.values()) > threshold:
                self.vocabulary_size += 1
                for doc, frequency in frequencies.items():
                    word_frequency[doc][word] = frequency
        self.word_frequency = word_frequency

    def start(self):
        threshold = 0.01
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
            if len(likelihoods) > 1 and abs(likelihoods[-1] - likelihoods[-2]) <= threshold:
                break
        EM.plot_likelihoods(likelihoods)
        return self.cal_clusters(w)

    def cal_clusters(self, w):
        clusters = defaultdict(list)
        for t in range(len(self.documents)):
            cluster_index = np.argmax(w[t])
            clusters[cluster_index].append(t)
        return clusters

    def initialize(self):
        w = self.initialize_w()
        alpha = self.cal_alpha(w)
        p = self.cal_p(w)
        return alpha, p

    def initialize_w(self):
        w = np.zeros((len(self.word_frequency), len(self.categories)))
        for t in self.word_frequency.keys():
            mod = t % len(self.categories)
            w[t][mod] = 1
        return w

    def cal_alpha(self, w):
        alpha = list()  # could be hard coded values according to % mod
        for i in range(len(self.categories)):
            alpha.append(0)
            for t in self.word_frequency.keys():
                alpha[i] += w[t][i]
            alpha[i] /= len(self.word_frequency)
        return EM.smoothing_alpha(alpha)

    def cal_p(self, w, lamda=0.0001):
        p = defaultdict(lambda: defaultdict(float))

        down_sum = list()
        for i in range(len(self.categories)):
            down_sum.append(sum([w[t][i] * len(self.documents[t]) for t in self.word_frequency.keys()]))

        for i in range(len(self.categories)):
            for t in self.word_frequency.keys():
                for word, frequency in self.word_frequency[t].items():
                    p[i][word] += w[t][i] * frequency

        for i in range(len(self.categories)):
            for word, current_value in p[i].items():
                p[i][word] = (current_value + lamda) / (down_sum[i] + lamda * self.vocabulary_size)

        return p

    def cal_w(self, alpha, p):
        ln_likelihood = 0
        w = defaultdict(lambda: defaultdict(float))
        for t in self.word_frequency.keys():
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
        return np.math.log(alpha[i], np.math.e) + sum(
            [frequency * np.math.log(p[i][word], np.math.e) for word, frequency in self.word_frequency[t].items()])

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
        smoothing = False
        for i in range(len(alpha)):
            if alpha[i] < eps:
                smoothing = True
                alpha[i] = eps
        if not smoothing:
            return alpha
        normal = sum(alpha)
        a = [a / normal for a in alpha]
        return a

    @staticmethod
    def plot_likelihoods(likelihoods):
        plt.plot(range(len(likelihoods)), likelihoods, label='linear')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
