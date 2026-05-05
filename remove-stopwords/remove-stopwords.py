def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    stopwords_set = set(stopwords)
    results = [token for token in tokens if token not in stopwords_set]
    return results