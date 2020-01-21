# Chen Eliyahou 312490675 Noam Simon 208388850

import numpy as np
from collections import Counter

def create_counter_dict(words):
    counter_dict = dict()
    for word in words:
        increase_counter(counter_dict, word)
    return counter_dict

# Chen Eliyahou 312490675 Noam Simon 208388850

def increase_counter(words_counter_dict, word):
    words_counter_dict[word] = words_counter_dict.get(word, 0) + 1


def read_lines(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


def get_words_list(lines):
    words_list = list()
    for i in range(1, len(lines), 2):
        for word in lines[i].split():
            words_list.append(word)

    return words_list

def get_documents_list(lines):
    docs_list = list()
    for i in range(1, len(lines), 2):
        words = lines[i].split()
        docs_list.append(words)

    return docs_list


def get_topics_list(lines):
        topics_list = list()
        for i in range(0, len(lines), 2):
            topics_list.append(lines[i].replace('<', '').replace('>', '').split('\t')[2:])
        return topics_list

def get_filter_dict(words_counter_dict, k=3):
    counted = dict(words_counter_dict)
    return [word for word in words_counter_dict if counted[word] > k]


def create_list_from_topic_file(topic_file):
    topics_list = read_lines(topic_file)
    return topics_list

def documents_words_frequency(documents_list):
    documents_words_freq = list()
    for doc in documents_list:
        documents_words_freq.append(Counter(doc))
    return documents_words_freq

def first_init_documents(documents_list, topics_list):
    prop_docs_in_categories_wti = dict()
    for i in range(len(documents_list)):
        topics_array = np.zeros(len(topics_list))
        mod = i % len(topics_list)
        topics_array[mod] = 1
        prop_docs_in_categories_wti[i] = topics_array
    return prop_docs_in_categories_wti