# Chen Eliyahou 312490675 Noam Simon 208388850

import sys
from datetime import datetime

import Utils
from Clusters import Clusters
from ConfusionMatrix import ConfusionMatrix
from EM_Algorithm import EM_Algorithm


def main(develop_file, topics_file):
    lines = Utils.read_lines(develop_file)
    # the dictionary |V|
    dev = Utils.get_words_list(lines)
    filter_word_dictionary = Utils.get_filter_dict(Utils.create_counter_dict(dev))
    print("create dictionary")
    # create documents list
    documents_list = Utils.get_documents_list(lines)
    print("create documents")
    # topics list from topics_file
    topics_list = Utils.create_list_from_topic_file(topics_file)
    print("categories list")

    # calculate frequency to every word in each document [word k -> frequency(n1k, ... ntk... nNk)]
    documents_words_freq = Utils.documents_words_frequency(documents_list)
    print("calculate frequency to every word")
    # init wti
    W = Utils.first_init_documents(documents_list, topics_list)

    em_algorithm = EM_Algorithm(len(filter_word_dictionary), documents_words_freq, topics_list, documents_list, W)

    start = datetime.now()
    W = em_algorithm.EM_algorithm()
    print('EM_algorithm running time: {0}'.format(datetime.now() - start))

    # get topics list for the documents
    topics_for_docs = Utils.get_topics_list(lines)

    # get the max cluster for each doc
    clusters = Clusters()
    docs_clusters = clusters.max_clusters_for_docs(W)

    # calculate confusion matrix
    confusion_matrix = ConfusionMatrix(docs_clusters, topics_for_docs, topics_list)
    confusion_matrix.compute_confusion_matrix()
    confusion_matrix.print_confusion_matrix()
    confusion_matrix.dominant_topic_for_clusters()
    print(confusion_matrix.accuracy())

    # write_outputs(outputs, output_filename)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2])
    print('Total running time: {0}'.format(datetime.now() - start))
