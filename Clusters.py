# Chen Eliyahou 312490675 Noam Simon 208388850


class Clusters:
    def __init__(self):
        pass

    def max_clusters_for_docs(self, W):
        docs_clusters = list()
        for doc in W.values():
            tuple = max((v, i) for i, v in enumerate(doc))
            docs_clusters.append(tuple[1])
        return docs_clusters
