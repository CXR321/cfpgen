import numpy as np
from .spectrum import spectrum_map,esm_embedding_map
from tqdm import tqdm
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
import esm
from functools import partial

def median_heuristic(emb):
    """
    计算 pairwise 欧几里得距离的中位数，并计算合适的 gamma。
    """
    pairwise_dists = pairwise_distances(emb, metric='euclidean')
    sigma = np.median(pairwise_dists)  # 取 pairwise 距离的中位数
    if sigma == 0:
        sigma = 1.0  # 避免 sigma 为 0
    gamma = 1.0 / (2 * sigma ** 2)  # RBF 核的 gamma
    return gamma

def mmd(seq1=None, seq2=None, emb1=None, emb2=None, mean1=None, mean2=None, embedding='esm', kernel='linear', kernel_args={}, return_pvalue=False, progress=False, **kwargs):
    '''
    Calculates MMD between two sets of sequences. Optionally takes embeddings or mean embeddings of sequences if these have been precomputed for efficiency. If <return_pvalue> is true, a Monte-Carlo estimate (1000 iterations) of the p-value is returned. Note that this is compute-intensive and only implemented for the linear kernel.
    '''

    if not mean1 is None and not mean2 is None:
        MMD = np.sqrt(np.dot(mean1,mean1) + np.dot(mean2,mean2) - 2*np.dot(mean1,mean2))
        return MMD

    if embedding == 'spectrum':
        embed = spectrum_map
    if embedding == 'profet':
        raise NotImplementedError
    if embedding == 'unirep':
        raise NotImplementedError
    if embedding == 'esm':
        DEVICE = "cuda:0"
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.to(DEVICE)
        model.eval()
        embed = partial(esm_embedding_map, 
                   model=model, 
                   alphabet=alphabet, 
                   layer=33, 
                   mode='cls')
 
    if mean1 is None and emb1 is None:
        emb1 = embed(sequences=seq1, progress=progress, **kwargs)
    if mean2 is None and emb2 is None:
        emb2 = embed(sequences=seq2, progress=progress, **kwargs)


    if kernel == 'linear':
        x = np.mean(emb1, axis=0)
        y = np.mean(emb2, axis=0)
        MMD = np.sqrt(np.dot(x,x) + np.dot(y,y) - 2*np.dot(x,y))
    elif kernel == 'gaussian':

        x = np.array(emb1)
        y = np.array(emb2)
        m = x.shape[0]
        n = y.shape[0]

        if 'gamma' not in kernel_args:
            gamma = median_heuristic(np.vstack([x, y]))
            kernel_args['gamma'] = gamma  # 更新 gamma

        Kxx = rbf_kernel(x,x, **kernel_args)#.numpy()
        Kxy = rbf_kernel(x,y, **kernel_args)#.numpy()
        Kyy = rbf_kernel(y,y, **kernel_args)#.numpy()
        MMD = np.sqrt(
            np.sum(Kxx) / (m**2)
            - 2 * np.sum(Kxy) / (m*n)
            + np.sum(Kyy) / (n**2)
        )

    if return_pvalue:
        agg = np.concatenate((emb1,emb2),axis=0)
        mmds = []
        it = tqdm(range(1000)) if progress else range(1000)
        for i in it:
            np.random.shuffle(agg)
            _emb1 = agg[:m]
            _emb2 = agg[m:]
            mmds.append(mmd(emb1=_emb1, emb2=_emb2, kernel=kernel, kernel_args=kernel_args))
        rank = float(sum([x<=MMD for x in mmds]))+1
        pval = (1000+1-rank)/(1000+1)
        return MMD, pval
    else:
        return MMD



