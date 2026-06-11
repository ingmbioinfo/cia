import os
import io
import requests
import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor, as_completed


def _fetch_url_gmt(url: str) -> dict:
    """Fetches a GMT file from a URL and parses it into a dictionary.

    Parameters
    ----------
    url : str
        The URL of the GMT file.

    Returns
    -------
    dict
        A dictionary where keys are gene set names and values are lists of associated genes.

    Raises
    ------
    ValueError
        If the request to fetch the GMT file fails.
    """

    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch GMT file. HTTP Status Code: {response.status_code}"
        )

    gene_sets = {}
    file_content = io.StringIO(response.text)

    for line in file_content:
        data = line.strip().split("\t")
        if len(data) < 3:
            continue
        gene_set_name = data[0]
        genes = data[2:]
        gene_sets[gene_set_name] = genes

    return gene_sets


def signatures_similarity(signatures_dict, show="J"):
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

    if show not in ["J", "%"]:
        raise ValueError('show must be "J" or "%".')

    signature_names = list(signatures_dict.keys())
    n = len(signature_names)
    similarity_matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            intersec = len(
                np.intersect1d(
                    signatures_dict[signature_names[i]],
                    signatures_dict[signature_names[j]],
                )
            )

            if show == "J":
                union = len(
                    np.union1d(
                        signatures_dict[signature_names[i]],
                        signatures_dict[signature_names[j]],
                    )
                )
                similarity = intersec / union if union > 0 else 0.0
            else:
                denom = len(signatures_dict[signature_names[i]])
                similarity = round(100 * intersec / denom, 2) if denom > 0 else 0.0

            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity

    return pd.DataFrame(
        similarity_matrix, index=signature_names, columns=signature_names
    )


def filter_degs(
    data,
    groupby,
    uns_key="rank_genes_groups",
    direction="up",
    logFC=0,
    scores=None,
    perc=0,
    mean=0,
):
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

    signatures_dict = {}

    for group in data.obs[groupby].cat.categories:
        degs = data.uns[uns_key]["names"][group]
        n_cells = sum(data.obs[groupby] == group)

        if direction == "up":
            order = (
                pd.DataFrame(data.uns[uns_key]["logfoldchanges"][group])
                .sort_values(by=0, ascending=False)
                .index
            )
            degs = degs[order]

            if scipy.sparse.issparse(data.raw.X):
                cells = (
                    np.array(
                        data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X.todense() > 0
                    ).sum(axis=0)
                    / n_cells
                    * 100
                )
            else:
                cells = (
                    np.array(
                        data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X > 0
                    ).sum(axis=0)
                    / n_cells
                    * 100
                )
            cells = cells >= perc

            gene_mean = np.ravel(
                data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X.mean(0)
            )
            gene_mean = gene_mean >= mean

            lfc = data.uns[uns_key]["logfoldchanges"][group]
            lfc = lfc[order] >= logFC

            filters = [cells, gene_mean, lfc]

            if scores is not None:
                s = data.uns[uns_key]["scores"][group]
                s = s[order] >= scores
                filters.append(s)

            filters = np.bitwise_and.reduce(filters)
            signatures_dict[group] = degs[filters].tolist()

        elif direction == "down":
            order = (
                pd.DataFrame(data.uns[uns_key]["logfoldchanges"][group])
                .sort_values(by=0, ascending=False)
                .index
            )
            degs = degs[order]

            if scipy.sparse.issparse(data.raw.X):
                cells = (
                    np.array(
                        data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X.todense() > 0
                    ).sum(axis=0)
                    / n_cells
                    * 100
                )
            else:
                cells = (
                    np.array(
                        data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X > 0
                    ).sum(axis=0)
                    / n_cells
                    * 100
                )
            cells = cells <= perc

            gene_mean = np.ravel(
                data.raw[data.obs[groupby].isin([group])][:, degs.tolist()].X.mean(0)
            )
            gene_mean = gene_mean <= mean

            lfc = data.uns[uns_key]["logfoldchanges"][group]
            lfc = lfc[order] <= logFC

            filters = [cells, gene_mean, lfc]

            if scores is not None:
                s = data.uns[uns_key]["scores"][group]
                s = s[order] <= scores
                filters.append(s)

            filters = np.bitwise_and.reduce(filters)
            signatures_dict[group] = degs[filters].tolist()

        else:
            raise ValueError('direction must be "up" or "down".')

    return signatures_dict


def _cia_exact_scores_matrix(adata, genes):
    """
    Compute cumulative CIA exact scores for a set of genes.

    For each cell (rows) and for each prefix of the provided `genes` list (columns),
    this function computes a cumulative count of expressed genes and the cumulative
    expression, then combines them into a raw CIA score which is scaled per column.

    Parameters
    ----------
    adata : AnnData
        Annotated data object. The function expects expression data to be available
        in `adata.raw`.
    genes : list-like
        Sequence of gene names (order matters). The returned matrix has as many
        columns as the length of `genes` and column k contains the score computed
        using the first k+1 genes.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_cells, n_genes) with scaled cumulative scores. Any NaNs
        or infinite values are converted to 0.0 before returning.

    Notes
    -----
    - The function handles both sparse and dense matrices by converting sparse
      inputs to dense arrays where needed.
    - It relies on `adata.raw` being present; if your code uses a local variable
      `raw` you should ensure it references `adata.raw`.
    """

    #raw = adata.raw

    #X = raw[:, genes].X
    X = adata.raw[:, genes].X
    total_expr = adata.raw.X.sum(axis=1)

    if scipy.sparse.issparse(total_expr):
        total_expr = total_expr.A1
    else:
        total_expr = np.asarray(total_expr).ravel()

    if scipy.sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    B = (X > 0).astype(np.float32)

    cum_count = np.cumsum(B, axis=1)
    cum_expr = np.cumsum(X, axis=1)

    safe_total_expr = np.where(total_expr == 0, np.nan, total_expr)
    raw_scores = cum_count * (cum_expr / safe_total_expr[:, None])

    max_per_col = np.nanmax(raw_scores, axis=0)
    max_per_col[max_per_col == 0] = np.nan

    scaled_scores = raw_scores / max_per_col
    scaled_scores = np.nan_to_num(scaled_scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scaled_scores


def _empty_group_result():
    return {
        "markers": [],
        "log": {
            "raw_best_n": 0,
            "raw_best_auc": np.nan,
            "selected_n": 0,
            "selected_auc": np.nan,
            "tested": {},
        },
    }


def _build_candidate_ns(n_genes, min_n=1, step=5, refine_window=5):
    coarse = list(range(min_n, n_genes + 1, step))
    if n_genes not in coarse:
        coarse.append(n_genes)
    coarse = sorted(set(coarse))
    return coarse


def _refine_ns(best_n, n_genes, min_n=1, refine_window=5):
    lo = max(min_n, best_n - refine_window)
    hi = min(n_genes, best_n + refine_window)
    return list(range(lo, hi + 1))


def _process_single_group(
    adata,
    expr,
    rg,
    padj_key,
    groupby,
    group,
    logFC,
    score,
    pct1,
    pct2,
    mean,
    padj,
    auc_tolerance,
    min_n,
    max_genes,
    step,
    refine_window,
):
    # instead of extract as dataframe columns extract for each group the properties as arrays
    genes = np.asarray(rg["names"][group]).astype(str)
    scores = np.asarray(rg["scores"][group], dtype=float)
    logfcs = np.asarray(rg["logfoldchanges"][group], dtype=float)
    padjs = np.asarray(rg[padj_key][group], dtype=float)

    # Check consistency of gene names with expression matrix and filter out genes not present in the expression matrix
    valid = np.isin(genes, expr.var_names) # Q: filter out genes not in the expression matrix, why? Do we expect this could happens?
    genes = genes[valid]
    scores = scores[valid]
    logfcs = logfcs[valid]
    padjs = padjs[valid]

    mask_group = adata.obs[groupby].values == group
    mask_rest = ~mask_group

    # number of postiitve cells for the group and the rest, used for percentage calculations
    n_group = int(mask_group.sum())
    n_rest = int(mask_rest.sum())

    if n_group == 0:
        out = _empty_group_result()
        return group, out["markers"], out["log"]

    Xg = expr[mask_group, genes].X
    Xr = expr[mask_rest, genes].X

    if scipy.sparse.issparse(Xg):
        pct1_expr = np.asarray((Xg > 0).sum(axis=0)).ravel() / n_group * 100
        mean_expr = np.asarray(Xg.mean(axis=0)).ravel()
    else:
        Xg = np.asarray(Xg)
        pct1_expr = (Xg > 0).sum(axis=0) / n_group * 100
        mean_expr = np.asarray(Xg.mean(axis=0)).ravel()

    if n_rest > 0:
        if scipy.sparse.issparse(Xr):
            pct2_expr = np.asarray((Xr > 0).sum(axis=0)).ravel() / n_rest * 100
        else:
            Xr = np.asarray(Xr)
            pct2_expr = (Xr > 0).sum(axis=0) / n_rest * 100
    else:
        pct2_expr = np.zeros(len(genes), dtype=float)

    # NOTE: we could avoid to create a dataframe and compute the percentage, 
    # actually scapy already do this using pts=True in rank_genes_groups, 
    # but we want to be sure to have the same percentage definition used for 
    # filtering and for the final output, so we compute it here again.
    # Eventually we could manage this in the parental function, check if pts 
    # is present and if not compute it and add to the uns, but for now we compute 
    # it here to be sure to have the same definition.
    df = pd.DataFrame(
        {
            "gene": genes,
            "score": scores,
            "logFC": logfcs,
            "padj": padjs,
            "pct1": pct1_expr,
            "pct2": pct2_expr,
            "mean": mean_expr,
        }
    )

    mask = (
        (df["logFC"] >= logFC)
        & (df["pct1"] >= pct1)
        & (df["pct2"] <= pct2)
        & (df["mean"] >= mean)
        & (df["padj"] <= padj)
    )

    if score is not None:
        mask &= df["score"] >= score

    # keep only the top max_genes genes after filtering, sorted by score and logFC, to limit the number of genes tested for AUC optimization
    df = df.loc[mask].sort_values(["score", "logFC"], ascending=[False, False])
    ranked_genes = df["gene"].drop_duplicates().tolist()[:max_genes] # Q: Why drop duplicates? Do we expect duplicated genes in the DE results?

    if len(ranked_genes) < min_n:
        out = _empty_group_result()
        return group, out["markers"], out["log"]

    y = (adata.obs[groupby].values == group).astype(int) # put group to 1 and the rest to 0, used for AUC calculation

    # score cumulation for all genes and all prefixes of the gene list, to avoid recomputing the scores for each tested n
    score_matrix = _cia_exact_scores_matrix(adata, ranked_genes)
    n_genes = len(ranked_genes)


    coarse_ns = _build_candidate_ns(
        n_genes=n_genes,
        min_n=min_n,
        step=step,
        refine_window=refine_window,
    )

    tested = {}

    for n in coarse_ns:
        auc = roc_auc_score(y, score_matrix[:, n - 1])
        if not np.isnan(auc):
            tested[int(n)] = float(auc)

    if len(tested) == 0:
        out = _empty_group_result()
        return group, out["markers"], out["log"]

    coarse_best_auc = max(tested.values())
    coarse_best_n = min(n for n, v in tested.items() if v == coarse_best_auc)

    refine_ns = _refine_ns(
        best_n=coarse_best_n,
        n_genes=n_genes,
        min_n=min_n,
        refine_window=refine_window,
    )

    for n in refine_ns:
        if n in tested:
            continue
        auc = roc_auc_score(y, score_matrix[:, n - 1])
        if not np.isnan(auc):
            tested[int(n)] = float(auc)

    if len(tested) == 0:
        out = _empty_group_result()
        return group, out["markers"], out["log"]

    raw_best_auc = max(tested.values())
    raw_best_n = min(n for n, v in tested.items() if v == raw_best_auc)

    if auc_tolerance is not None:
        selected_n = min(n for n, v in tested.items() if v >= raw_best_auc - auc_tolerance)
    else:
        selected_n = raw_best_n

    selected_auc = tested[selected_n]
    markers = ranked_genes[:selected_n]

    log_dict = {
        "raw_best_n": int(raw_best_n),
        "raw_best_auc": float(raw_best_auc),
        "selected_n": int(selected_n),
        "selected_auc": float(selected_auc),
        "tested": {int(k): float(v) for k, v in sorted(tested.items())},
        "n_cells_used_for_auc": int(adata.n_obs),
        "n_ranked_genes_after_cap": int(len(ranked_genes)),
    }

    return group, markers, log_dict


def retrieve_optimal_markers(
    adata,
    groupby,
    uns_key="rank_genes_groups",
    logFC=0.25,
    score=None,
    pct1=10,
    pct2=100,
    mean=0.1,
    padj=0.05,
    auc_tolerance=None,
    use_raw=True,
    verbose=False,
    uns_key_added="retrieve_optimal_markers",
    min_n=1,
    n_jobs=None,
    max_genes=250,
    step=5,
    refine_window=5,
):
    """
    Retrieve optimal markers for each group using CIA scoring with AUC optimization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    groupby : str
        Column name in adata.obs to use for grouping.
    uns_key : str, optional
        Key in adata.uns containing rank_genes_groups results. Default is "rank_genes_groups".
    logFC : float, optional
        Log fold change threshold. Default is 0.25.
    score : str, optional
        Scoring method. Default is None.
    pct1 : float, optional
        Percentage threshold 1. Default is 10.
    pct2 : float, optional
        Percentage threshold 2. Default is 100.
    mean : float, optional
        Mean expression threshold. Default is 0.1.
    padj : float, optional
        Adjusted p-value threshold. Default is 0.05.
    auc_tolerance : float, optional
        Tolerance for AUC selection. Default is None.
    use_raw : bool, optional
        Whether to use raw data. Must be True. Default is True.
    verbose : bool, optional
        If True, print progress information. Default is False.
    uns_key_added : str, optional
        Key to store results in adata.uns. Default is "retrieve_optimal_markers".
    min_n : int, optional
        Minimum number of genes to consider. Default is 1.
    n_jobs : int, optional
        Number of parallel jobs. If None, uses minimum of number of groups and CPU count. Default is None.
    max_genes : int, optional
        Maximum number of genes to test. Default is 250.
    step : int, optional
        Step size for testing different gene numbers. Default is 5.
    refine_window : int, optional
        Window size for refinement around optimal n. Default is 5.

    Returns
    -------
    dict
        Dictionary mapping group names to lists of optimal marker genes.

    Raises
    ------
    ValueError
        If groupby not in adata.obs, uns_key not in adata.uns, use_raw is False, adata.raw is None,
        or if parameter values are invalid.
    """
    if groupby not in adata.obs.columns:
        raise ValueError(f"{groupby!r} not found in adata.obs")
    if uns_key not in adata.uns:
        raise ValueError(f"{uns_key!r} not found in adata.uns")
    if not use_raw:
        raise ValueError("Exact CIA scoring requires use_raw=True.")
    if adata.raw is None:
        raise ValueError("use_raw=True but adata.raw is None.")
    if max_genes < 1:
        raise ValueError("max_genes must be >= 1.")
    if min_n < 1:
        raise ValueError("min_n must be >= 1.")
    if step < 1:
        raise ValueError("step must be >= 1.")
    if refine_window < 0:
        raise ValueError("refine_window must be >= 0.")

    rg = adata.uns[uns_key]
    expr = adata.raw.to_adata()

    if "pvals_adj" in rg:
        padj_key = "pvals_adj"
    elif "pvals" in rg:
        padj_key = "pvals"
    else:
        raise ValueError("Neither 'pvals_adj' nor 'pvals' found in DE results.")

    col = adata.obs[groupby]
    if isinstance(col.dtype, pd.CategoricalDtype):
        groups = list(col.cat.categories)
    else:
        groups = list(pd.unique(col))

    # Determine number of jobs for parallel processing based on the number of free core at the moment
    if n_jobs is None:
        n_jobs = min(len(groups), os.cpu_count() or 1)
    n_jobs = max(1, int(n_jobs))

    optimal_markers = {}
    logs = {
        "params": {
            "groupby": groupby,
            "uns_key": uns_key,
            "logFC": logFC,
            "score": score,
            "pct1": pct1,
            "pct2": pct2,
            "mean": mean,
            "padj": padj,
            "auc_tolerance": auc_tolerance,
            "use_raw": use_raw,
            "min_n": min_n,
            "n_jobs": n_jobs,
            "max_genes": max_genes,
            "step": step,
            "refine_window": refine_window,
        },
        "groups": {},
    }

    futures = {}
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for group in groups:
            future = executor.submit(
                _process_single_group,
                adata=adata,
                expr=expr,
                rg=rg,
                padj_key=padj_key,
                groupby=groupby,
                group=group,
                logFC=logFC,
                score=score,
                pct1=pct1,
                pct2=pct2,
                mean=mean,
                padj=padj,
                auc_tolerance=auc_tolerance,
                min_n=min_n,
                max_genes=max_genes,
                step=step,
                refine_window=refine_window,
            )
            futures[future] = group #it works like a pointer, it doesn't block the execution

        for future in as_completed(futures): 
            group = futures[future]
            try:
                group_name, markers, group_log = future.result() # get the result for the ended thread
                optimal_markers[group_name] = markers
                logs["groups"][group_name] = group_log

                if verbose:
                    print(
                        f"{group_name}: selected n = {group_log['selected_n']} | "
                        f"selected AUC = {group_log['selected_auc']:.6f} | "
                        f"raw best n = {group_log['raw_best_n']} | "
                        f"raw best AUC = {group_log['raw_best_auc']:.6f} | "
                        f"cells used = {group_log['n_cells_used_for_auc']} | "
                        f"genes used = {group_log['n_ranked_genes_after_cap']}"
                    )

            except Exception as e:
                optimal_markers[group] = []
                logs["groups"][group] = {
                    "raw_best_n": 0,
                    "raw_best_auc": np.nan,
                    "selected_n": 0,
                    "selected_auc": np.nan,
                    "tested": {},
                    "error": repr(e),
                }
                if verbose:
                    print(f"{group}: ERROR -> {repr(e)}")

    optimal_markers = {group: optimal_markers.get(group, []) for group in groups}
    logs["groups"] = {
        group: logs["groups"].get(group, _empty_group_result()["log"]) for group in groups
    }

    adata.uns[uns_key_added] = logs
    return optimal_markers


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
    
    with open(file, "w") as f:
        for key, values in signatures_dict.items():
            line = key + "\t" + key + "\t" + "\t".join(values) + "\n"
            f.write(line)