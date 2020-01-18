import sys
from datetime import datetime

import Utils
from EM import EM


def main(develop_file, topics_file):
    docs = Utils.read_lines(develop_file)
    categories = Utils.read_lines(topics_file)
    em = EM(docs, categories)
    em.start()


if __name__ == "__main__":
    start = datetime.now()
    main(sys.argv[1], sys.argv[2])
    print('Total running time: {0}'.format(datetime.now() - start))
