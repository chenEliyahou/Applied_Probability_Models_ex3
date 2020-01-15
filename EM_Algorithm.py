from datetime import datetime
import numpy as np



class EM_Algorithm:
    def __init__(self, total_words_size, words_freq, categories_list, documents_list, init_wti):
        self.__total_words_size = total_words_size
        self.__words_freq = words_freq
        self.__categories_list = categories_list
        self.__documents_list = documents_list
        self.__alpha = []
        self.__p_words_for_categories = []
        self.__total_documents = len(documents_list)
        self.__total_categories = len(categories_list)
        self.__W = init_wti

    def EM_algorithm(self):
        # init
        self.__alpha = self.compute_probs_for_all_categories()
        start = datetime.now()
        self.__p_words_for_categories = self.compute_all_words_for_categories()
        print('P running time: {0}'.format(datetime.now() - start))

        # likelihood =  self.compute_likelihood(alpha, p)


        for i in range(5):
            # E level
            start = datetime.now()

            # wti = self.get_probs_all_docs_in_all_categories()
            W = self.e_level()
            print('E running time: {0}'.format(datetime.now() - start))

            # M level
            start = datetime.now()
            self.__alpha = self.compute_probs_for_all_categories()
            print('alpha running time: {0}'.format(datetime.now() - start))

            start = datetime.now()
            p = self.compute_all_words_for_categories()
            print('P running time: {0}'.format(datetime.now() - start))


            # Likelihood

            # start = datetime.now()
            # new_likelihood = self.compute_likelihood(alpha, p)
            # print('likelihood running time: {0}'.format(datetime.now() - start))
            # print(new_likelihood)

    #  **************** E level *****************

    def e_level(self):
        for doc in range(self.__total_documents):
            print("t= " + str(doc))
            z_categories = []
            for category in range(self.__total_categories):
                start = datetime.now()
                z_categories = self.get_z_categories(category, doc)
                print('get_z_categories: {0}'.format(datetime.now() - start))

                m = max(z_categories)
                start = datetime.now()
                wti = self.compute_wti(z_categories[category], m, z_categories)
                print('compute_wti: {0}'.format(datetime.now() - start))

                self.__W[doc][category] = wti

    def compute_wti(self, z_i, m, z_categories, k=10):
        return 0 if (z_i - m) < -k else np.exp(z_i - m) / sum(np.exp(z_j - m) for z_j in z_categories if (z_j - m) < -k)

    def get_z_categories(self, category, document):
        z_categories = list()
        for probs_words_for_category in self.__p_words_for_categories:
            # z_categories.append(self.compute_z_for_category_i(category, document, probs_word_for_category))
            z_categories.append((np.log(self.__alpha[category])) +
                                (sum(word_freq[document] * np.log(prob_word_for_category))
                                 for word_freq, prob_word_for_category in
                                   zip(self.__words_freq.values(), probs_words_for_category)))
        return z_categories

    def compute_z_for_category_i(self, category, document, probs_words_for_category):
        tmp = 0
        for word_freq, prob_word_for_category in zip(self.__words_freq.values(), probs_words_for_category):
             tmp += word_freq[document] * np.log(prob_word_for_category)
        return np.log(self.__alpha[category]) + tmp
        #return (np.log(self.__alpha[category])) + (sum(word_freq[document] * np.log(prob_word_for_category)) for word_freq, prob_word_for_category in zip(self.__words_freq.values(), probs_words_for_category))




    #  **************** M level *****************

    def compute_probs_for_all_categories(self, eps = 0.000045):
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
        probs_for_categories[:] = [(prob_i/sum_probs_categories) for prob_i in probs_for_categories]





    def compute_all_words_for_categories(self):
        probs_words_for_categories = list()
        for category in range(len(self.__categories_list)):
            probs_words_for_categories.append(self.compute_all_words_for_category(category))
        return probs_words_for_categories

    def compute_all_words_for_category(self, category):
        probs_words_for_category = list()
        for freq_word in self.__words_freq:
            probs_words_for_category.append(self.compute_probs_word_for_category(self.__words_freq[freq_word], category))
        return probs_words_for_category


    # Pik = ∑t(wti*ntk) + λ
    #       ∑t(wti*nt) + |V |λ
    def compute_probs_word_for_category(self, freq_word, category, lamda = 0.01):
        first_calc = 0
        for wti, freq in zip(self.__W, freq_word):
            first_calc += self.__W[wti][category]*freq
        second_calc = 0
        for wti, doc in zip(self.__W, self.__documents_list):
            second_calc += self.__W[wti][category] * doc[1]

        return (first_calc + lamda) / (second_calc + self.__total_words_size*lamda)


    # **************** Likelihood *****************

    # def compute_likelihood(self, probs_all_categories, probs_all_words_for_categories, likelihood = True):
    #     likelihood_sum = 0
    #     for doc in self.__documents_list:
    #         w_doc_categories = self.get_probs_doc_in_categories(probs_all_categories, probs_all_words_for_categories, doc, likelihood)
    #         likelihood_sum += sum(w_doc_categories)
    #     return
    #
    #
    # def compute_wti_likelihood(self, z_i, m, z_categories, k=10):
    #     if (z_i - m) < -k:
    #         return 0
    #     else:
    #         return pow(np.e, z_i - m)
    #
    #
    #
