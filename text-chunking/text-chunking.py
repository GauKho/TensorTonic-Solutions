def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    if len(tokens) <= chunk_size:
        return [tokens] if tokens else []
    
    chunk_store = []

    for i in range(0, len(tokens) - chunk_size + 1, chunk_size - overlap):
        chunk_store.append(tokens[i : i + chunk_size])

    return chunk_store