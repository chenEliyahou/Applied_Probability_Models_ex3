# Chen Eliyahou 312490675 Noam Simon 208388850

import numpy as np
import pandas as pn
import seaborn as sn
import matplotlib.pyplot as plt


class Confusion_Matrix:
    def __init__(self, topics_for_docs):
        self.__confusion_matrix = []
        self.__topics_for_docs = topics_for_docs

    def compute_confusion_matrix(self, docs_clusters, topics):
        cluster_sizes = np.zeros(9)
        confusion_matrix = np.zeros((9, 10))
        for doc_index, cluster in enumerate(docs_clusters):
            cluster_sizes[cluster] += 1
            for topic in self.__topics_for_docs[doc_index]:
                confusion_matrix[cluster][topics.index(topic)] += 1

        for cluster_index, cluster_size in enumerate(cluster_sizes):
            confusion_matrix[cluster_index][-1] = cluster_size

        # Rows are sorted by cluster size in descending order.
        confusion_matrix = sorted(confusion_matrix, key = lambda row: row[-1], reverse = True)
        # np.argsort(confusion_matrix[:, -1])
        # reversed(self.compute_confusion_matrix)

        self.__confusion_matrix = confusion_matrix

    def print_confusion_matrix(self, topics):
        columns = topics + ['Size']

        df = pn.DataFrame(data=self.__confusion_matrix, columns=columns)
        print(df)
        print()
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df, annot=True, annot_kws={"size": 16})  # font size
        plt.show()
