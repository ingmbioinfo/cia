import numpy as np
import pandas as pd
import itertools 
import scipy

def signatures_similarity(signatures_dict, show='J'):
    """
    Computes the similarity between gene signatures.

    Parameters
    ----------
    signatures_dict : dict
        A dictionary having as keys the signature names and as values the lists of gene names (gene signatures).
    show : str, optional
        Specifies the metric for showing similarities: 'J' for Jaccard index or '%' for percentages of intersection.
        Default is 'J'.

    Returns
    -------
    similarity : pandas.DataFrame
        A DataFrame containing the similarity of each pair of signatures, with signatures as both rows and columns.

    Raises
    ------
    ValueError
        If 'show' is different from 'J' or '%'.

    Example
    -------
    >>> signatures = {
    >>>     'signature1': ['gene1', 'gene2', 'gene3'],
    >>>     'signature2': ['gene2', 'gene3', 'gene4'],
    >>>     'signature3': ['gene1', 'gene5']
    >>> }
    >>> similarity = signatures_similarity(signatures, show='J')
    >>> print(similarity)
    """
    if show not in ['J', '%']:
        raise ValueError('show must be "J" or "%".')
    
    signature_names = list(signatures_dict.keys())
    n = len(signature_names)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            intersec = len(np.intersect1d(signatures_dict[signature_names[i]], signatures_dict[signature_names[j]]))
            if show == 'J':
                union = len(np.union1d(signatures_dict[signature_names[i]], signatures_dict[signature_names[j]]))
                similarity = intersec / union
            elif show == '%':
                similarity = round(100 * intersec / len(signatures_dict[signature_names[i]]), 2)
            
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    similarity = pd.DataFrame(similarity_matrix, index=signature_names, columns=signature_names)
    return similarity

import pandas as pd
import numpy as np
import scipy.sparse

def filter_degs(data, groupby, uns_key='rank_genes_groups', direction='up', logFC=0, scores=None, perc=0, mean=0):
    """
    Filters differentially expressed genes (DEGs) obtained with scanpy.tl.rank_genes_groups based on given thresholds.
    
    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the analysis results.
    groupby : str
        Column in AnnData.obs containing cell group labels.
    uns_key : str
        Key in AnnData.uns where differential expression analysis results are stored.
    direction : str
        Specifies if filtering for upregulated ('up') or downregulated ('down') genes.
    logFC : float
        Log fold change threshold to filter genes.
    scores : float, optional
        Z score threshold to filter genes.
    perc : float
        Percentage of cells expressing the gene threshold.
    mean : float
        Mean expression threshold to filter genes.
    
    Returns
    -------
    signatures_dict : dict
        Dictionary with cell group names as keys and lists of filtered gene names as values.
    
    Raises
    ------
    ValueError
        If 'direction' is not 'up' or 'down'.
    
    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> sc.tl.rank_genes_groups(adata, 'louvain', method='t-test')
    >>> filtered_genes = filter_degs(adata, 'louvain', direction='up', logFC=1, perc=10, mean=0.1)
    >>> print(filtered_genes['0'])  # Show filtered genes for the first group
    """

    signatures_dict={}
    for group in data.obs[groupby].cat.categories:
    #for group in data.uns[uns_key]['names'].dtype.names:
        degs=data.uns[uns_key]['names'][group] 


        n_cells=sum(data.obs[groupby]==group)


        
        if direction=='up':
            
            order=pd.DataFrame(data.uns[uns_key]['logfoldchanges'][group]).sort_values(by=0,ascending=False).index
            degs= degs[order]

            if scipy.sparse.issparse(data.raw.X):
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.todense() > 0).sum(axis=0)/n_cells*100)
            else:
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X > 0).sum(axis=0)/n_cells*100)
            cells =(cells >= perc)
            
            gene_mean = np.ravel(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.mean(0))
            gene_mean = (gene_mean >= mean )
            
            lfc= data.uns[uns_key]['logfoldchanges'][group]
            lfc= (lfc[order] >=logFC)
            
            filters=[cells, gene_mean, lfc]
            
            if scores!=None:
                s= data.uns[uns_key]['scores'][group]
                s= (s[order] >=scores)
                filters.append(s)
            

            filters=np.bitwise_and.reduce(filters)
            signatures_dict[group]= degs[filters].tolist()
            
            

            
        elif direction=='down':

            order=pd.DataFrame(data.uns[uns_key]['logfoldchanges'][group]).sort_values(by=0,ascending=False).index
            degs= degs[order]
            
            if scipy.sparse.issparse(data.raw.X):
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.todense() > 0).sum(axis=0)/n_cells*100)
            else:
                cells = (np.array(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X> 0).sum(axis=0)/n_cells*100)
            cells =(cells <= perc)
            
            gene_mean = np.ravel(data.raw[data.obs[groupby].isin([group])][:,degs.tolist()].X.mean(0))
            gene_mean = (gene_mean <= mean )
            
            lfc= data.uns[uns_key]['logfoldchanges'][group]
            lfc= (lfc[order] <=logFC)
            
            filters=[cells, gene_mean, lfc]
            
            if scores!=None:
                s= data.uns[uns_key]['scores'][group]
                s= (s[order] <=scores)
                filters.append(s)
            

            filters=np.bitwise_and.reduce(filters)
            signatures_dict[group]= degs[filters].tolist()
            
        else:
            raise ValueError('direction must be "up" or "down".')
        
    return signatures_dict

def save_gmt(signatures_dict, file):
    """
    A function to convert a dictionary of signatures in a gmt file correctly formatted for signature_score and signature_based_classification functions.
    
    Parameters
    ----------
     
    signatures_dict: dict
        a dictionary having as keys the signature names and as values the gene signatures (lists of gene names).
    file: str
        filepath of gmt file. See pandas.DataFrame.to_csv documentation.  
    """
    pd.DataFrame.from_dict(signatures_dict, orient='index').to_csv(file, sep='\t', header=None)
