def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k_list = recommended[:k]

    hit = list(set(top_k_list) & set(relevant))
    precision_k = len(hit) / k
    recall_k = len(hit) / len(relevant)

    return [precision_k, recall_k]