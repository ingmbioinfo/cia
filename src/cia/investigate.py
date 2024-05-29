import numpy as np
import pandas as pd
from anndata import AnnData
import time 
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_signatures(signatures_input):
    """
    Load gene signatures from a given source.

    This function loads gene signatures from either a local file path, a URL, or directly from a dictionary. If a file path or URL is provided, the file should be in tab-separated format with the first column as keys and subsequent columns as values.

    Parameters
    ----------
    signatures_input : str or dict
        The source of the gene signatures. This can be a path to a tab-separated file, a URL pointing to such a file, or a dictionary where keys are signature names and values are lists of gene names.

    Returns
    -------
    dict
        A dictionary where each key is a signature name and each value is a list of gene names associated with that signature.

    Raises
    ------
    TypeError
        If `signatures_input` is neither a string (for file paths or URLs) nor a dictionary.

    Examples
    --------
    >>> signatures = load_signatures('signatures.tsv')
    >>> print(signatures['signature1'])
    ['gene1', 'gene2', 'gene3']

    >>> signatures_dict = {'signature1': ['gene1', 'gene2'], 'signature2': ['gene3', 'gene4']}
    >>> signatures = load_signatures(signatures_dict)
    >>> print(signatures['signature1'])
    ['gene1', 'gene2']
    """
    if isinstance(signatures_input, str):
        df = pd.read_csv(signatures_input, sep='\t', header=None)
        signatures = {row[0]: row[1:].dropna().tolist() for _, row in df.iterrows()}
    elif isinstance(signatures_input, dict):
        signatures = signatures_input
    else:
        raise TypeError("signatures_input must be either a dict or a string path/URL to a GMT file.")
    return signatures

def score_signature(data, geneset):
    """
    Compute signature scores (from https://doi.org/10.1038/s41467-021-22544-y) for a single gene set against the provided dataset.

    This function calculates the signature scores based on the presence (count) and expression (exp) of genes in the specified gene set within the dataset. 
    The score is the product of the count of genes expressed in a given cell and the sum of their expression levels, normalized by the total expression detected in the cell.
    
    Parameters
    ----------
    data : AnnData
        An AnnData object containing the dataset to compute scores for. It is expected to have an attribute `raw` containing an `X` matrix (observations x variables) and `var_names` (gene names).
    geneset : array_like
        A list or array of gene names for which to compute the signature scores.

    Returns
    -------
    numpy.ndarray
        An array of signature scores, one per observation (cell) in `data`.

    Notes
    -----
    The function first intersects the provided gene set with the gene names available in `data.raw.var_names` to ensure only relevant genes are considered. If no genes from the gene set are found in the data, the function returns an array of zeros.

    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.pbmc68k_reduced()
    >>> geneset = ['CD3D', 'CD3E', 'CD3G', 'CD4', 'CD8A', 'CD8B']
    >>> scores = compute_signature_scores(adata, geneset)
    >>> print(scores.shape)
    (700,)

    Raises
    ------
    AttributeError
        If `data` does not have the required `raw` attribute or if `raw` does not have the `X` and `var_names` attributes.
    """
    geneset = np.intersect1d(geneset, data.raw.var_names)
    if len(geneset) == 0:
        return np.zeros(data.shape[0])
    count = (data.raw[:, geneset].X > 0).sum(axis=1)
    exp = data.raw[:, geneset].X.sum(axis=1) / data.raw.X.sum(axis=1)
    return (np.array(count) * np.array(exp)).reshape(-1)



def score_all_signatures(data, signatures_input, score_mode='raw', return_df=False, n_cpus=None):
    """
    Compute signature scores for a given dataset and a set of gene signatures.

    This function checks which genes from the signatures are present in the dataset,
    computes the signature scores for each cell in the dataset, and can return the scores
    as a DataFrame or add them to the `data.obs`.

    Parameters
    ----------
    data : AnnData
        An AnnData object containing the dataset to compute scores for, expected to have a `raw`
        attribute containing a matrix (`X`) and `var_names`.
    signatures_input : str or dict
        Path to a file or a dictionary containing gene signatures. If a string is provided,
        it should be the file path or URL to the signature file.
    score_mode : str, optional
        The mode of score calculation. Options are 'raw', 'scaled', 'log', 'log2', 'log10'.
        Defaults to 'raw'.
    return_df : bool, optional
        If True, the function returns a DataFrame with signature scores. Otherwise, it adds
        the scores to `data.obs`. Defaults to False.
    n_cpus : int, optional
        Number of CPU cores to use for parallel processing. If None, uses all available cores.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the signature scores if `return_df` is True. Otherwise, the function
        adds the scores to `data.obs` and returns None.

    Notes
    -----
    The function parallelizes the computation of signature scores across the specified number of CPU cores.
    It prints the number of genes found in both the signatures and the dataset for each signature.

    Examples
    --------
    >>> data = sc.read_h5ad('path/to/your/data.h5ad')  # Assume sc is Scanpy and data is loaded
    >>> signatures_input = 'path/to/signatures.txt'
    >>> signature_scores = signature_score(data, signatures_input, score_mode='scaled', return_df=True)
    >>> signature_scores.head()
    """
    signatures = load_signatures(signatures_input)
    print('Checking if genes are in AnnData.var_names...\n')
    signatures_summary = [f"{sig}: {len(np.intersect1d(genes, data.raw.var_names))}/{len(genes)}"
                      for sig, genes in signatures.items()]

    print('\n'.join(signatures_summary), '\n')
    # Define a function that will be executed in parallel
    def compute_score(name_geneset_tuple):
        name, geneset = name_geneset_tuple
        return name, score_signature(data, geneset)

    # Use ThreadPoolExecutor to parallelize the score computation
    scores = {}
    with ThreadPoolExecutor(max_workers=n_cpus) as executor:
        # Submit all tasks to the executor
        futures = [executor.submit(compute_score, item) for item in signatures.items()]
        for future in as_completed(futures):
            name, score = future.result()
            scores[name] = score
        executor.shutdown(wait=True)
        del futures

    scores_df = pd.DataFrame(scores)

    if score_mode == 'raw':
        pass  # scores_df is already in 'raw' format
    elif score_mode == 'scaled':
        scores_df = scores_df / scores_df.max()
    elif score_mode in ['log', 'log2', 'log10']:
        min_val = np.nextafter(0, 1)
        log_func = {'log': np.log, 'log2': np.log2, 'log10': np.log10}[score_mode]
        scores_df = log_func(scores_df + min_val)
    else:
        raise ValueError("Invalid score_mode. Must be one of: ['raw', 'scaled', 'log', 'log2', 'log10']")
    for col in scores_df.columns:
        data.obs[col] = scores_df[col].values
    if return_df:
        return scores_df

            
def CIA_classify(data, signatures_input, n_cpus=None, similarity_threshold=0, label_column='CIA_prediction', score_mode='scaled', unassigned_label='Unassigned'):
    """
    Classify cells in `data` based on gene signature scores.

    This function computes scaled signature scores for the provided data against a set of gene signatures.
    It then classifies each cell based on the highest score unless the top two scores are too similar,
    in which case it assigns an 'Unassigned' label.

    Parameters
    ----------
    data : AnnData
        An AnnData object containing the dataset to compute scores for, expected to have a `raw`
        attribute containing a matrix (`X`) and `var_names`.
    signatures_input : str or dict
        Path to a file or a dictionary containing gene signatures. If a string is provided,
        it should be the file path or URL to the signature file.
    n_cpus : int, optional
        Number of CPU cores to use for parallel computation. If None, all available cores are used.
    similarity_threshold : float, optional
        The threshold below which the top two scores are considered too similar, resulting in an 'Unassigned' label.
        Defaults to 0.1 (difference < 10%).
    label_column : str, optional
        The column name in `data.obs` where the classification labels will be stored. Defaults to 'CIA prediction'.
    unassigned_label : str, optional
        The label to assign when the top two scores are too similar. Defaults to 'Unassigned'.

    Returns
    -------
    None
        The function directly modifies the `data` object by adding classification labels to `data.obs`.

    Notes
    -----
    The function calculates signature scores using the `score_all_signatures` function. The highest score is used for
    classification unless it is within the `similarity_threshold` of the second-highest score.

    Examples
    --------
    >>> data = sc.read_h5ad('path/to/your/data.h5ad')  # Assume sc is Scanpy and data is loaded
    >>> signatures_input = 'path/to/signatures.txt'
    >>> CIA_classify(data, signatures_input, similarity_threshold=0.1)
    >>> data.obs['CIA prediction']
    """
    start_time=time.time()
    scores_df = score_all_signatures(data, signatures_input, score_mode=score_mode, return_df=True, n_cpus=n_cpus)

    # Identify the indices of the highest and second highest scores
    sorted_scores_idx = np.argsort(-scores_df.values, axis=1)
    top_score_idx = sorted_scores_idx[:, 0]
    second_top_score_idx = sorted_scores_idx[:, 1]
    
    # Calculate the scores for the highest and second highest scores
    top_scores = np.take_along_axis(scores_df.values, top_score_idx[:, None], axis=1).flatten()
    second_top_scores = np.take_along_axis(scores_df.values, second_top_score_idx[:, None], axis=1).flatten()
    
    # Determine if the top score is too similar to the second top score
    score_difference = top_scores - second_top_scores
    too_similar = score_difference <= similarity_threshold
    
    # Generate labels based on the highest score, assign 'Unassigned' if scores are too similar
    labels = [scores_df.columns[i] if not sim else unassigned_label for i, sim in zip(top_score_idx, too_similar)]
    
    # Update the data object with new labels
    data.obs[label_column] = labels

  
    end_time = time.time()  # Capture end time
    print(f'Classification complete! Start time: {time.strftime("%H:%M:%S", time.gmtime(start_time))}, '
          f'End time: {time.strftime("%H:%M:%S", time.gmtime(end_time))}, '
          f'Results stored in AnnData.obs["{label_column}"]')
