import numpy as np
import pandas as pd
import scanpy as sc

def celltypist_majority_vote(data, classification_obs, groups_obs=None, min_prop=0, unassigned_label='Unassigned'):
    """
    A function that wraps Celltypist majority voting (DOI: 10.1126/science.abl5197).
    Assigns cell group labels based on the majority voting of cell type predictions within each group.

    If no reference cell groups are provided, an over-clustering step is performed using the Leiden algorithm.

    Parameters
    ----------
    data : anndata.AnnData
        An AnnData object containing the cell data and, optionally, previous clustering results.
    classification_obs : str or list of str
        The AnnData.obs column(s) where the cell type predictions (labels) are stored.
    groups_obs : str, optional
        The AnnData.obs column where the reference group labels are stored. If None, an over-clustering with the
        Leiden algorithm is performed based on the dataset size.
    min_prop : float, optional
        The minimum proportion of cells required to assign a majority vote label to a group. If the largest
        cell type in a group doesn't reach this proportion, the group is labeled as 'Unassigned'.
    unassigned_label : str, optional
        The label to assign to cell groups where no cell type reaches the minimum proportion. Default is 'Unassigned'.

    Notes
    -----
    The function automatically adjusts the resolution for the Leiden algorithm based on the number of observations in the data.
    Results of majority voting are stored back in the AnnData.obs, adding a column for each classification considered.
    """
    # Determine resolution for Leiden clustering based on data size if groups_obs is not provided
    if groups_obs is None:
        resolution = 5 + 5 * (data.n_obs // 20000)  # Increasing resolution in steps based on data size
        print(f'Reference annotation not selected. Computing over-clustering with Leiden algorithm (resolution={resolution}) ...')
        sc.tl.leiden(data, resolution=resolution, key_added=f'leiden_{resolution}')
        groups_obs = f'leiden_{resolution}'
        print(f'Dataset has been divided into {len(data.obs[groups_obs].cat.categories)} groups according to transcriptional similarities.')
        print(f'Over-clustering result saved in AnnData.obs["{groups_obs}"].')
    else:
        print(f'AnnData.obs["{groups_obs}"] selected as reference annotation.')

    print('Extending the more represented cell type label to each cell group...\n')
    groups = data.obs[groups_obs]

    # Ensure classification_obs is a list
    classification_obs = [classification_obs] if isinstance(classification_obs, str) else classification_obs

    for classification in classification_obs:
        votes = pd.crosstab(data.obs[classification], groups)
        majority = votes.idxmax()
        freqs = votes.max() / votes.sum()

        # Apply minimum proportion threshold to assign labels
        majority_labels = majority.where(freqs >= min_prop, unassigned_label)
        data.obs[f'{classification}_majority_voting'] = groups.map(majority_labels).astype('category')

        print(f'New classification labels have been stored in AnnData.obs["{classification}_majority_voting"].')
