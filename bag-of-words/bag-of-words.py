import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    word_set = [word for word in vocab]
    vector = np.zeros(len(word_set), dtype=int)

    for token in tokens:
        if token in word_set:
            index = word_set.index(token)
            vector[index] += 1
    return vector