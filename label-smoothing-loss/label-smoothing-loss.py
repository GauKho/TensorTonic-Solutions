def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    import numpy as np

    predictions = np.array(predictions, dtype=float)
    k = len(predictions)
    
    # Shape with value epsilon/k
    q = np.full(k, epsilon / k)
    q[target] = (1 - epsilon) + (epsilon/k)

    loss = -np.sum(q * np.log(predictions + 1e-12))

    return loss