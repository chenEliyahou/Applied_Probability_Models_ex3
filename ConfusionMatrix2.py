# Chen Eliyahou 312490675 Noam Simon 208388850

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
import seaborn as sn


class ConfusionMatrix:
    def __init__(self, clusters, docs_topics, topics_list):
        self.__clusters = clusters
        self.__topics = topics_list
        self.__confusion_matrix = []
        self.__not_sorted_confusion_matrix = []
        self.__docs_topics = docs_topics
        self.__dominant_topic = list()

    def compute_confusion_matrix(self):
        confusion_matrix = np.zeros((len(self.__clusters), len(self.__topics) + 1))
        for cluster_index, docs in self.__clusters.items():
            confusion_matrix[cluster_index][-1] = len(docs)
            for doc_index in docs:
                for topic in self.__docs_topics[doc_index]:
                    confusion_matrix[cluster_index][self.__topics.index(topic)] += 1

        self.__confusion_matrix = sorted(confusion_matrix, key=lambda row: row[-1], reverse=True)

        self.__not_sorted_confusion_matrix = confusion_matrix

    def print_confusion_matrix(self):
        columns = self.__topics + ['Size']
        df = pn.DataFrame(data=self.__confusion_matrix, columns=columns)
        df.plot()
        plt.show()

    def dominant_topic_for_clusters(self):
        for cluster in self.__not_sorted_confusion_matrix:
            topic_index = np.argmax(cluster[:-2])
            self.__dominant_topic.append(self.__topics[topic_index])

    def accuracy(self):
        success = 0
        total = 0
        for cluster_index, docs in self.__clusters.items():
            for doc_index in docs:
                total += 1
                if self.__dominant_topic[cluster_index] in self.__docs_topics[doc_index]:
                    success += 1
        return success / total
