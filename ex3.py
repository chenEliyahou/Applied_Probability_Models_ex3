from datetime import datetime
import sys
import Utils
from EM_Algorithm import EM_Algorithm

def main(develop_file, topics_file, test_file, output_filename):

    lines = Utils.read_lines(develop_file)
    # the dictionary |V|
    dev = Utils.get_words_list(lines)
    filter_word_dictionary = Utils.get_filter_dict(Utils.create_counter_dict(dev))
    print("create dictionary")
    # create documents list
    documents_list = Utils.get_documents_list(lines)
    print("create documents")
    # topics list from topics_file
    categories_list = Utils.create_list_from_topic_file(topics_file)
    print("categories list")

    # !!!!!!!!!!!!!!!!!!!!!!!!!
    # calculate frequency to every word in each document [word k -> frequency(n1k, ... ntk... nNk)]
    words_freq = Utils.words_frequency_in_documents(filter_word_dictionary, documents_list)
    print("calculate frequency to every word")
    # init wti
    wti = Utils.first_init_documents(documents_list, categories_list)

    em_algorithm = EM_Algorithm(len(filter_word_dictionary), words_freq, categories_list, documents_list, wti)

    start = datetime.now()
    em_algorithm.EM_algorithm()
    print('EM_algorithm running time: {0}'.format(datetime.now() - start))

    # probs_all_categories = EM_Algorithm.compute_probs_for_all_categories(wti, topics_list)

    # probs_all_words_for_categories = EM_Algorithm.compute_all_words_for_categories(wti, words_freq_ntk, documents_list, V, topics_list)


# test = utils.get_words_list()
    # topics_test = utils.get_topics_list()


    #write_outputs(outputs, output_filename)


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print('Total running time: {0}'.format(datetime.now() - start))
