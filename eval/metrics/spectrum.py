import numpy as np
import itertools
from tqdm import tqdm
import torch

amino_acid_alphabet = 'ARNDCEQGHILKMFPSTWYV'

def make_kmer_trie(k):
    '''
    For efficient lookup of k-mers.
    '''
    kmers = [''.join(i) for i in itertools.product(amino_acid_alphabet, repeat = k)]
    kmer_trie = {}
    for i,kmer in enumerate(kmers):
        tmp_trie = kmer_trie
        for aa in kmer:
            if aa not in tmp_trie:
                tmp_trie[aa] = {}
            if 'kmers' not in tmp_trie[aa]:
                tmp_trie[aa]['kmers'] = []
            tmp_trie[aa]['kmers'].append(i)
            tmp_trie = tmp_trie[aa]
    return kmer_trie

three_mer_trie = make_kmer_trie(3)

def spectrum_map(sequences, k=3, mode='count', normalize=True, progress=False):
    '''
    Maps a set of sequences to k-mer vector representation.
    '''

    if isinstance(sequences, str):
        sequences = [sequences]
    if k==3:
        trie = three_mer_trie
    else:
        trie = make_kmer_trie(k)

    def matches(substring):
        d = trie
        for letter in substring:
            try:
                d = d[letter]
            except KeyError:
                return []
        return d['kmers']

    def map(sequence):
        vector = np.zeros(len(amino_acid_alphabet)**k)
        for i in range(len((sequence))-k+1):
            for j in matches(sequence[i:i+k]):
                if mode == 'count':
                    vector[j] += 1
                elif mode == 'indicate':
                    vector[j] = 1
        feat = np.array(vector)
        if normalize:
            norm = np.sqrt(np.dot(feat,feat))
            if norm != 0:
                feat /= norm
        return feat

    it = tqdm(sequences) if progress else sequences
    return np.array([map(seq) for seq in it], dtype=np.float32)

def esm_embedding_map(sequences, model, alphabet, mode='cls', layer=33, progress=False):
    '''
    Maps a set of protein sequences to ESM embedding representation.
    
    Parameters:
    -----------
    sequences : list of str or str
        Protein sequences to embed
    model : esm.pretrained model
        Pretrained ESM model
    alphabet : esm.data.Alphabet
        ESM alphabet
    mode : str, optional (default='mean')
        How to pool token embeddings: 'mean', 'sum', 'cls', or 'pooler'
    layer : int, optional (default=None)
        Which layer to extract embeddings from (None = last layer)
    progress : bool, optional (default=False)
        Whether to show progress bar
    
    Returns:
    --------
    numpy.ndarray
        Array of shape (n_sequences, embedding_dim)
    '''
    
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Set model to evaluation mode
    model.eval()
    
    # Get device
    device = next(model.parameters()).device
    
    # Prepare batch converter
    batch_converter = alphabet.get_batch_converter()
    
    def get_embeddings(sequence):
        # Prepare data
        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Get embeddings
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer] if layer is not None else [model.num_layers], return_contacts=False)
            embeddings = results["representations"][layer if layer is not None else model.num_layers]
        
        # Remove batch dimension and special tokens
        embeddings = embeddings[0, 1:-1, :]  # Remove [CLS] and [EOS] tokens
        
        # Pool embeddings
        if mode == 'mean':
            pooled = embeddings.mean(dim=0)
        elif mode == 'sum':
            pooled = embeddings.sum(dim=0)
        elif mode == 'cls':
            pooled = results["representations"][layer if layer is not None else model.num_layers][0, 0, :]  # [CLS] token
        elif mode == 'pooler' and hasattr(model, 'pooler'):
            pooled = model.pooler(embeddings.unsqueeze(0)).squeeze(0)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        
        return pooled.cpu().numpy()
    
    # Process sequences
    it = tqdm(sequences) if progress else sequences
    embeddings = []
    
    for seq in it:
        try:
            emb = get_embeddings(seq)
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing sequence: {e}")
            # Add zero vector as fallback
            embeddings.append(np.zeros(model.embed_dim if hasattr(model, 'embed_dim') else 1280))
    
    return np.array(embeddings, dtype=np.float32)