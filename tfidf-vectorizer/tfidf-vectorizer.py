import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    N = len(documents)

    tokenized_docs = [docs.split() for docs in documents]
    vocab = sorted(set(word for docs in tokenized_docs for word in docs))
    word_index = {word: i for i, word in enumerate(vocab)}

    df = Counter()
    for doc in tokenized_docs:
        unique = set(doc)
        for word in unique:
            df[word] += 1

    idf = {word: math.log(N/df[word]) for word in vocab}
    
    tfidf_matrix = np.zeros((N, len(vocab)))

    for i, doc in enumerate(tokenized_docs):
        tf = Counter(doc)
        doc_len = len(doc)

        for word in tf:
            j = word_index[word]
            tf_value = tf[word] / doc_len
            tfidf_matrix[i,j] = tf_value * idf[word]

    return tfidf_matrix, vocab