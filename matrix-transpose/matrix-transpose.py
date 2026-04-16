import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    row, col = A.shape

    result = np.zeros((col, row), dtype=A.dtype)

    for i in range(row):
        for j in range(col):
            result[j,i] = A[i,j]

    return result