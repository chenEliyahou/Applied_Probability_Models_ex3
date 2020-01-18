from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


class EM_Algorithm:
    def __init__(self, total_words_size, documents_words_freq, categories_list, documents_list, init_wti):
        self.__total_words_size = total_words_size
        self.__documents_words_freq = documents_words_freq
        self.__categories_list = categories_list
        self.__documents_list = documents_list
        self.__alpha = []
        self.__total_documents = len(documents_list)
        self.__total_categories = len(categories_list)
        self.__W = init_wti
        self.__p = []
        self.__likelihoods = []

    def EM_algorithm(self, eps=0.01):
        # init
        # start = datetime.now()
        self.__alpha = self.compute_probs_for_all_categories()
        # print('alpha running time: {0}'.format(datetime.now() - start))
        # start = datetime.now()
        self.__p = self.compute_p_level()
        # print('P running time: {0}'.format(datetime.now() - start))

        self.__likelihoods.append(self.compute_likelihood())

        while True:
            # E level
            # start = datetime.now()

            # wti = self.get_probs_all_docs_in_all_categories()
            W = self.e_level()
            # print('E running time: {0}'.format(datetime.now() - start))

            # M level
            # start = datetime.now()
            self.__alpha = self.compute_probs_for_all_categories()
            # print('alpha running time: {0}'.format(datetime.now() - start))

            # start = datetime.now()
            self.__p = self.compute_p_level()
            # print('P running time: {0}'.format(datetime.now() - start))

            # Likelihood
            # start = datetime.now()
            self.__likelihoods.append(self.compute_likelihood())
            # print('likelihood running time: {0}'.format(datetime.now() - start))
            print("likelihood = " + str(self.__likelihoods[-1]))
            if abs(self.__likelihoods[-1] - self.__likelihoods[-2]) <= eps:
                break

        self.plot_likelihoods(self.__likelihoods)

    # **************** E level *****************

    def e_level(self):
        for doc in range(self.__total_documents):
            z_categories = self.get_z_categories(doc)
            m = max(z_categories)
            for category in range(self.__total_categories):
                wti = self.compute_wti(z_categories[category], m, z_categories)
                self.__W[doc][category] = wti

    def compute_wti(self, z_i, m, z_categories, k=10):
        return 0 if (z_i - m) < -k else (
                np.exp(z_i - m) / sum(np.exp(z_j - m) for z_j in z_categories if (z_j - m) >= -k))

    def get_z_categories(self, document):
        z_categories = list()
        for category in range(self.__total_categories):
            sum = 0
            words_counter = self.__documents_words_freq[document]
            for word, word_freq in words_counter.items():
                sum += word_freq * np.log(self.__p[word][category])
            z_categories.append(np.log(self.__alpha[category]) + sum)
        return z_categories

    #  **************** M level *****************

    # **************************** alpha level **********************************88

    def compute_probs_for_all_categories(self, eps=0.000045):
        probs_all_categories = list()
        for c_i in range(len(self.__categories_list)):
            prob_categoty = self.compute_prob_to_category(c_i)
            probs_all_categories.append(prob_categoty)

        fix_probs = False
        for i in range(len(probs_all_categories)):
            if probs_all_categories[i] < eps:
                fix_probs = True
                probs_all_categories[i] += eps
        if fix_probs:
            self.fix_alpha(probs_all_categories)

        return probs_all_categories

    # αi = P(Ci) = 1/N*∑(wti)
    def compute_prob_to_category(self, category):
        sum_probs = 0
        for wti in self.__W:
            sum_probs += self.__W[wti][category]
        return (1 / self.__total_documents) * (sum_probs)

    def fix_alpha(self, probs_for_categories):
        sum_probs_categories = sum(map(np.array, probs_for_categories))
        probs_for_categories[:] = [(prob_i / sum_probs_categories) for prob_i in probs_for_categories]

    # **************************** P level **********************************88

    def compute_p_level(self, lamda=0.001):
        p = defaultdict(lambda: defaultdict(float))

        for doc in range(self.__total_documents):
            for word, word_freq in self.__documents_words_freq[doc].items():
                for category in range(len(self.__categories_list)):
                    p[word][category] += self.__W[doc][category] * word_freq

        # denominator calculate
        denominator = np.zeros(len(self.__categories_list))
        for doc in range(self.__total_documents):
            for category in range(len(self.__categories_list)):
                denominator[category] += self.__W[doc][category] * len(self.__documents_list[doc])

        # calc P with smoothing
        for word in p:
            for category in range(len(self.__categories_list)):
                p[word][category] = (p[word][category] + lamda) / (
                        denominator[category] + (self.__total_words_size * lamda))

        return p

    # *********************** Likelihood ***********************

    def compute_likelihood(self):
        likelihood_sum = 0
        for doc in range(self.__total_documents):
            w_probs_categories_for_doc = 0
            z_categories = self.get_z_categories(doc)
            m = max(z_categories)
            for category in range(self.__total_categories):
                w_probs_categories_for_doc += self.compute_wti_likelihood(z_categories[category], m)
            likelihood_sum += m + np.log(w_probs_categories_for_doc)
        return likelihood_sum

    def compute_wti_likelihood(self, z_i, m, k=10):
        if (z_i - m) >= -k:
            return np.exp(z_i - m)
        else:
            return 0

    def plot_likelihoods(self, likelihoods):
        plt.plot(range(len(likelihoods)), likelihoods, label='linear')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
