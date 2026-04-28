import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    if len(docs) == 0:
        return np.array([])
        
    N = len(docs)
    vocab = set(word for doc in docs for word in doc)
    
    df = Counter()
    for doc in docs:
        unique_word = set(doc)
        for word in unique_word:
            df[word] += 1

    idf = {word: math.log(((N - df[word] + 0.5) / (df[word] + 0.5))+1) for word in vocab}

    bm25_scores = np.zeros(N)
    avgdl = sum(len(doc) for doc in docs) / N
    if avgdl == 0:
        return np.zeros(N)

    for i, doc in enumerate(docs):
        doc_len = len(doc)
        for word in query_tokens:
            if word in doc:
                tf = doc.count(word)
                idf_value = idf[word]
                bm25_scores[i] += idf_value * ((tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len / avgdl))))
                
    return bm25_scores