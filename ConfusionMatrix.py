# Chen Eliyahou 312490675 Noam Simon 208388850

import matplotlib.pyplot as plt
import numpy as np
import pandas as pn
import seaborn as sn


class ConfusionMatrix:
    def __init__(self, docs_clusters, topics_for_docs, topics_list):
        self.__docs_clusters = docs_clusters
        self.__topics = topics_list
        self.__confusion_matrix = []
        self.__topics_for_docs = topics_for_docs
        self.__dominant_topic = list()

    def compute_confusion_matrix(self):
        cluster_sizes = np.zeros(9)
        confusion_matrix = np.zeros((9, 10))
        for doc_index, cluster in enumerate(self.__docs_clusters):
            cluster_sizes[cluster] += 1
            for topic in self.__topics_for_docs[doc_index]:
                confusion_matrix[cluster][self.__topics.index(topic)] += 1

        for cluster_index, cluster_size in enumerate(cluster_sizes):
            confusion_matrix[cluster_index][-1] = cluster_size

        # Rows are sorted by cluster size in descending order.
        confusion_matrix = sorted(confusion_matrix, key=lambda row: row[-1], reverse=True)

        self.__confusion_matrix = confusion_matrix

    def print_confusion_matrix(self):
        columns = self.__topics + ['Size']

        df = pn.DataFrame(data=self.__confusion_matrix, columns=columns)
        print(df)
        print()
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df, annot=True, annot_kws={"size": 16})  # font size
        plt.show()

    def dominant_topic_for_clusters(self):
        for cluster in self.__confusion_matrix:
            topic_index = np.argmax(cluster[:-2])
            self.__dominant_topic.append(self.__topics[topic_index])

    def accuracy(self):
        success = 0
        total = 0
        for doc_index, cluster in enumerate(self.__docs_clusters):
            total += 1
            if self.__dominant_topic[cluster] in self.__topics_for_docs[doc_index]:
                success += 1
        return success / total
