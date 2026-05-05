def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    V = list(set(tokens))
    V_size = len(V)

    unigram_counts = {}
    for i in range(0, len(tokens) - 1):
        w1, w2 = tokens[i] , tokens[i+1]
        unigram_counts[(w1, w2)] = unigram_counts.get((w1, w2), 0) + 1

    context_counts = {}
    for (w1, w2), c in unigram_counts.items():
        context_counts[w1] = context_counts.get(w1, 0) + c

    probs = {}
    for w1 in V:
        denom = context_counts.get(w1, 0) + V_size
        for w2 in V:
            c = unigram_counts.get((w1, w2), 0)
            probs[(w1, w2)] = (c + 1) / denom

    return unigram_counts, probs